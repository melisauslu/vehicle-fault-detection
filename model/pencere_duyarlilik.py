import json
import os
import subprocess
import sys
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

BURASI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURASI)
from ortak import (ANA_KLASOR, bolmeler, kat_sabitleri, fizik_ekle, ozellik_listesi, model_kur,
                   denetimli_egit_tahmin, degerlendir, en_iyi_parametreleri_oku)

# PENCERE BOYU DUYARLILIK ANALİZİ — "neden 30 saniye?" sorusu için.
# Literatürde pencere boyu uygulamaya göre denenerek seçiliyor (ör. Feng vd. 2025, Sci. Rep.: 4–8 s denenmiş,
# 6 s en iyi). Burada aynı şeyi yapıyoruz:
#   - Pencere boyu dışında HER ŞEY sabit: adım 5 s, aynı arızalı dosyalar, aynı 5 kat, aynı özellik seti
#     (ham+fizik+gecmis), aynı model (ayarlı XGBoost — ayarlar 30 s'de seçildi, bu yüzden 30 s'ye biraz avantaj
#     var; tezde belirtilmeli).
#   - YALNIZCA çapraz doğrulama (28 eğitim sürüşü). Test sürüşlerine dokunulmaz; ana sonuçlar DEĞİŞMEZ.
#   - Alarm kuralı her pencere boyu için alarm_kurali.py ile AYNI ölçütle ayrıca seçilir (N pencere boyuna göre
#     farklı çıkabilir).
# KALDIĞI YERDEN DEVAM: her (pencere, kat) bittiği anda tahminleri dosyaya yazılır; tekrar çalıştırınca atlanır.

# ============================================================================
# AYARLAR
# ============================================================================
PENCERELER_S = [10, 20, 30, 45, 60]
MODEL = "XGBoost"
SET = ["ham_istatistik", "fizik_dusuk", "uzun_gecmis"]       # ana set: ham+fizik+gecmis
TESPIT_HEDEFI, YANLIS_ALARM_HEDEFI = 0.95, 1.0              # alarm_kurali.py ile aynı
N_ADAYLARI = list(range(1, 31))

CIKTI = os.path.join(ANA_KLASOR, "data", "model", "pencere_duyarlilik")
RAPOR = os.path.join(ANA_KLASOR, "reports", "model")
os.makedirs(CIKTI, exist_ok=True)
os.makedirs(RAPOR, exist_ok=True)


def veri_klasoru(p):
    # 30 s için mevcut ana veri kullanılır (zaten 30 s ile hazırlandı); diğerleri ayrı klasöre hazırlanır
    ana = os.path.join(ANA_KLASOR, "data", "model")
    if p == 30 and os.path.exists(os.path.join(ana, "bilgi.json")):
        return ana
    return os.path.join(ANA_KLASOR, "data", f"model_pencere{p}s")


def veriyi_hazirla(p):
    klasor = veri_klasoru(p)
    bilgi_yolu = os.path.join(klasor, "bilgi.json")
    if os.path.exists(bilgi_yolu):
        with open(bilgi_yolu, encoding="utf-8") as f:
            if json.load(f)["pencere_satir"] == p * 10:
                return klasor
    print(f"--- {p} s pencereyle veri hazırlanıyor (veri_hazirla.py, birkaç dakika sürebilir) ---", flush=True)
    ortam = {**os.environ, "PENCERE_S": str(p), "VERI_KLASORU": klasor, "DL_KAYDET": "0"}
    subprocess.run([sys.executable, os.path.join(BURASI, "veri_hazirla.py")], env=ortam, check=True,
                   stdout=subprocess.DEVNULL)
    return klasor


def oku(klasor):
    oz = pd.read_csv(os.path.join(klasor, "pencereler_ozellik.csv"), sep=";", low_memory=False)
    with open(os.path.join(klasor, "bilgi.json"), encoding="utf-8") as f:
        bilgi = json.load(f)
    return oz, bilgi


def ardisik_say(a):
    # alarm_kurali.py ile aynı: her pencerede o ana kadar kesintisiz kaç pencere "arıza" dendi
    c = np.zeros(len(a), dtype=int)
    for i, v in enumerate(a):
        c[i] = (c[i - 1] + 1 if i else 1) if v else 0
    return c


def alarm_kurali(t, adim_s):
    # alarm_kurali.py'deki ölçütün aynısı: tespit ≥ %95 olan N'ler içinde yanlış alarm olayı/saat ≤ 1 olan en
    # küçük N; yoksa tespiti sağlayan en büyük N
    t = t.sort_values(["dosya", "pencere_bitis_s"])
    dosyalar = []
    for dosya, d in t.groupby("dosya", sort=False):
        tahmin = d["tahmin"].to_numpy()
        dosyalar.append({"bolme": d["bolme"].iloc[0], "tip": d["ariza_tipi"].iloc[0], "tahmin": tahmin,
                         "c": ardisik_say(tahmin != "saglam"), "sure": d["ariza_suresi_s"].to_numpy()})
    satirlar = []
    for N in N_ADAYLARI:
        kat, tespit, gecikme, teshis = {}, [], [], []
        for f in dosyalar:
            if f["tip"] == "saglam":
                k = kat.setdefault(f["bolme"], {"olay": 0, "saat": 0.0})
                k["olay"] += int(np.sum(f["c"] == N))
                k["saat"] += len(f["c"]) * adim_s / 3600
                continue
            i = np.flatnonzero(f["c"] >= N)
            tespit.append(len(i) > 0)
            if len(i):
                gecikme.append(f["sure"][i[0]])
                teshis.append(Counter(f["tahmin"][i[0] - N + 1:i[0] + 1]).most_common(1)[0][0] == f["tip"])
        satirlar.append({"N": N, "olay_saatte": np.mean([k["olay"] / k["saat"] for k in kat.values()]),
                         "tespit": np.mean(tespit), "gecikme_s": float(np.median(gecikme)) if gecikme else np.nan,
                         "teshis": float(np.mean(teshis)) if teshis else np.nan})
    s = pd.DataFrame(satirlar)
    uygun = s[s["tespit"] >= TESPIT_HEDEFI]
    hedefte = uygun[uygun["olay_saatte"] <= YANLIS_ALARM_HEDEFI]
    if len(hedefte):
        return s[s["N"] == hedefte["N"].min()].iloc[0], "hedefte"
    if len(uygun):
        return s[s["N"] == uygun["N"].max()].iloc[0], "hedef tutmadı"
    return None, "tespit < %95"


# ============================================================================
# ÇALIŞTIR
# ============================================================================
parametre = en_iyi_parametreleri_oku().get(MODEL)
print(f"Model: {MODEL} (ayarlı: {json.dumps(parametre)})   Set: ham+fizik+gecmis   Adım: 5 s   "
      f"Yalnızca 5 katlı çapraz doğrulama (test sürüşleri kullanılmaz)\n")
ozet, sinif_satirlari = [], []
for p in PENCERELER_S:
    klasor = veriyi_hazirla(p)
    oz, bilgi = oku(klasor)
    assert bilgi["pencere_satir"] == p * 10, f"{klasor} {p} s ile hazırlanmamış"
    cv, _ = bolmeler(oz)
    adim_s = bilgi["adim_satir"] * 0.1
    siniflar = sorted(oz["ariza_tipi"].unique())
    kolonlar = ozellik_listesi(bilgi, SET)
    parcalar = []
    for i, (egitim, dogrulama) in enumerate(cv):
        yol = os.path.join(CIKTI, f"p{p}s_kat{i}.csv")
        if not os.path.exists(yol):
            sab = kat_sabitleri(oz, egitim, bilgi)          # sabitler yalnızca bu katın eğitiminden
            o = fizik_ekle(oz, sab)
            y_tr = o.loc[egitim, "ariza_tipi"].to_numpy()
            tahmin, _, _ = denetimli_egit_tahmin(MODEL, model_kur(MODEL, parametre), o.loc[egitim, kolonlar], y_tr,
                                                 o.loc[dogrulama, kolonlar])
            m = o.loc[dogrulama, ["dosya", "ariza_tipi", "ariza_suresi_s", "pencere_bitis_s"]]
            m.assign(bolme=f"kat{i}", tahmin=tahmin).to_csv(yol + ".tmp", sep=";", index=False)
            os.replace(yol + ".tmp", yol)                  # yarım dosya kalmasın
        parcalar.append(pd.read_csv(yol, sep=";"))
    t = pd.concat(parcalar, ignore_index=True)

    katlar = []
    for b, x in t.groupby("bolme"):
        y, yh = x["ariza_tipi"].to_numpy(), x["tahmin"].to_numpy()
        katlar.append({"macro_f1": f1_score(y, yh, labels=siniflar, average="macro", zero_division=0),
                       **degerlendir(x, y, yh, adim_s)})
        f1s = f1_score(y, yh, labels=siniflar, average=None, zero_division=0)
        sinif_satirlari += [{"pencere_s": p, "bolme": b, "sinif": c, "f1": f} for c, f in zip(siniflar, f1s)]
    k = pd.DataFrame(katlar)
    a, durum = alarm_kurali(t, adim_s)
    r = {"pencere_s": p, "macro_f1": k["macro_f1"].mean(), "macro_f1_std": k["macro_f1"].std(ddof=0),
         "tespit_f1": k["tespit_f1"].mean(), "yanlis_alarm_pencere_saatte": k["yanlis_alarm_saatte"].mean(),
         "gecikme_kuralsiz_s": k["tespit_gecikmesi_ortanca_s"].mean(),
         "N": np.nan if a is None else int(a["N"]), "alarm_durumu": durum,
         "olay_saatte_N": np.nan if a is None else a["olay_saatte"],
         "tespit_N": np.nan if a is None else a["tespit"],
         "gecikme_N_s": np.nan if a is None else a["gecikme_s"],
         "teshis_N": np.nan if a is None else a["teshis"]}
    ozet.append(r)
    print(f"  {p:2d} s  macro-F1 {r['macro_f1']:.3f} ± {r['macro_f1_std']:.3f}  tespit-F1 {r['tespit_f1']:.3f}  "
          f"| alarm N={r['N']:.0f} ({durum}): yanlış alarm {r['olay_saatte_N']:.2f}/saat, "
          f"tespit %{100 * r['tespit_N']:.1f}, gecikme {r['gecikme_N_s']:.0f} s, teşhis %{100 * r['teshis_N']:.1f}",
          flush=True)

# ============================================================================
# ÖZET + KAYDET + GRAFİK
# ============================================================================
s = pd.DataFrame(ozet)
sf = pd.DataFrame(sinif_satirlari).groupby(["sinif", "pencere_s"])["f1"].mean().unstack("pencere_s")
s.to_csv(os.path.join(RAPOR, "pencere_duyarlilik.csv"), sep=";", index=False)
sf.to_csv(os.path.join(RAPOR, "pencere_duyarlilik_sinif_f1.csv"), sep=";")

pd.set_option("display.width", 250)
print("\n=== ÖZET: pencere boyuna göre (5 katın ortalaması; yalnızca çapraz doğrulama) ===")
goster = s.copy()
for c in ["tespit_N", "teshis_N"]:
    goster[c] = (100 * goster[c]).round(1)
print(goster.round(3).to_string(index=False))
print("\n=== Arıza tipine göre F1 (sütunlar: pencere boyu, saniye) ===")
print(sf.round(2).to_string())

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].errorbar(s["pencere_s"], s["macro_f1"], yerr=s["macro_f1_std"], marker="o", capsize=4, color="#2a6fbb")
ax[0].set_ylabel("macro-F1 (5 kat ort. ± std)")
ax[0].set_title("Teşhis başarısı", fontsize=10)
ax[1].plot(s["pencere_s"], s["gecikme_N_s"], marker="o", color="#c23b3b")
ax[1].set_ylabel("alarm gecikmesi, ortanca (s)")
ax[1].set_title("Alarm kuralıyla gecikme", fontsize=10)
ax[2].plot(s["pencere_s"], s["olay_saatte_N"], marker="o", color="#3a9a5b")
ax[2].axhline(YANLIS_ALARM_HEDEFI, color="#999999", linestyle="--", linewidth=1)
ax[2].set_ylabel("yanlış alarm olayı / saat")
ax[2].set_title("Alarm kuralıyla yanlış alarm", fontsize=10)
for a_ in ax:
    a_.set_xlabel("pencere boyu (s)")
    a_.set_xticks(PENCERELER_S)
    a_.axvline(30, color="#bbbbbb", linewidth=1, zorder=0)
    a_.grid(color="#e5e5e5", linewidth=0.6)
    a_.set_axisbelow(True)
    for side in ("top", "right"):
        a_.spines[side].set_visible(False)
fig.suptitle(f"Pencere boyu duyarlılığı — {MODEL}, ham+fizik+gecmis, çapraz doğrulama (dikey çizgi: seçilen 30 s)",
             fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RAPOR, "pencere_duyarlilik.png"), dpi=120)
plt.close(fig)
print("\nKaydedildi: reports/model/pencere_duyarlilik.csv, pencere_duyarlilik_sinif_f1.csv, pencere_duyarlilik.png")
