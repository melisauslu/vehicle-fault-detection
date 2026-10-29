import json
import os
import sys
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import ANA_KLASOR, veriyi_oku

RAPOR = os.path.join(ANA_KLASOR, "reports", "model")
TAHMIN = os.path.join(ANA_KLASOR, "data", "model")

# ALARM KURALI: "art arda N pencere arıza derse alarm ver"
# Tek pencerelik kararlar yerine kalıcılık (persistence) şartı: kısa süreli tek tük yanlış tahminler alarm
# üretmez; gerçek (kalıcı) arızalar ise birkaç pencere üst üste arıza dendiği için yine yakalanır.
# BİZİM KARARIMIZ:
#   - N, ÖNCEDEN belirlenen iki adımlı ölçütle seçilir:
#       1) Koşul: arızalı dosyaların en az TESPIT_HEDEFI kadarı yakalanmalı (arızaların %5'inden fazlası kaçmasın)
#       2) Bu koşulu sağlayan N'ler içinde, yanlış alarm olayı/saat ≤ YANLIS_ALARM_HEDEFI olan EN KÜÇÜK N
#          (gereğinden büyük N gecikmeyi boşuna artırır); hedefe ulaşılamıyorsa koşulu sağlayan EN BÜYÜK N.
#     İlk ölçüt ("koşulu sağlayan en büyük N") sentetik veride, yanlış alarm sıfırlandıktan sonra bile N'yi
#     aralığın ucuna itip gecikmeyi gereksiz büyüttüğü için, gerçek sonuçlar görülmeden bu hale getirildi.
#   - Her model kendi N'ini aynı kuralla alır (eşitlik).
#   - Yalnızca çapraz doğrulama tahminleri kullanılır (test sürüşlerine dokunulmaz). Seçilen N'ler kaydedilir;
#     son testte AYNI N'ler kullanılır.
#   - "Arıza var mı" düzeyinde: üst üste N pencerenin hepsi "sağlam değil" ise alarm (arıza tipleri farklı olabilir).
#     Alarm anındaki teşhis = o N penceredeki en sık arıza tipi.
# Basitleştirme: arızalı dosyalarda yalnızca arıza başladıktan SONRAKİ pencereler var (veri_hazirla.py); arıza
#   öncesi kısım temiz sürüşle aynı olduğundan oradaki yanlış alarmlar temiz sürüş üzerinden sayılır.

# ============================================================================
# AYARLAR
# ============================================================================
AYARLI = True                       # True → ml_tahminler_ayarli.csv (ayarlı modeller), False → varsayılan
SET = "ham+fizik+gecmis"            # ana set
TESPIT_HEDEFI = 0.95                # arızalı dosyaların en az bu kadarı yakalanmalı (BİZİM KARARIMIZ)
YANLIS_ALARM_HEDEFI = 1.0           # saatte en fazla bu kadar yanlış alarm OLAYI (BİZİM KARARIMIZ, ürün hedefi)
N_ADAYLARI = list(range(1, 31))     # 1–30 pencere (5 s adımla: ilk alarm en erken (N−1)·5 s sonra)

oz, bilgi = veriyi_oku()
adim_s = bilgi["adim_satir"] * 0.1
ek = "_ayarli" if AYARLI else ""
t = pd.read_csv(os.path.join(TAHMIN, f"ml_tahminler{ek}.csv"), sep=";", low_memory=False)
t = t[t["ozellik_seti"] == SET].sort_values(["model", "dosya", "pencere_bitis_s"])
modeller = sorted(t["model"].unique())


def ardisik_say(a):
    # a: her pencerenin "arıza" kararı (True/False) → her pencerede o ana kadar kesintisiz kaç pencere arıza dendi
    c = np.zeros(len(a), dtype=int)
    for i, v in enumerate(a):
        c[i] = (c[i - 1] + 1 if i else 1) if v else 0
    return c


satirlar, tip_satirlari = [], []
for model in modeller:
    m = t[t["model"] == model]
    dosyalar = []
    for dosya, d in m.groupby("dosya", sort=False):
        tahmin = d["tahmin"].to_numpy()
        dosyalar.append({"dosya": dosya, "bolme": d["bolme"].iloc[0], "tip": d["ariza_tipi"].iloc[0],
                         "saglam": d["ariza_tipi"].iloc[0] == "saglam", "tahmin": tahmin,
                         "c": ardisik_say(tahmin != "saglam"), "sure": d["ariza_suresi_s"].to_numpy()})
    for N in N_ADAYLARI:
        # --- sağlam sürüşler: yanlış alarm (her katta ayrı, sonra ortalama — ml_modeller.py ile aynı) ---
        kat = {}
        for f in dosyalar:
            if f["saglam"]:
                k = kat.setdefault(f["bolme"], {"pencere": 0, "olay": 0, "saat": 0.0})
                k["pencere"] += int(np.sum(f["c"] >= N))      # alarm durumunda geçen pencere
                k["olay"] += int(np.sum(f["c"] == N))         # ayrı alarm OLAYI (her kesintisiz dizi bir kez)
                k["saat"] += len(f["c"]) * adim_s / 3600
        # --- arızalı dosyalar: tespit, gecikme, alarm anındaki teşhis ---
        tespit, gecikme, dogru_teshis = [], [], []
        for f in dosyalar:
            if f["saglam"]:
                continue
            i = np.flatnonzero(f["c"] >= N)
            tespit.append(len(i) > 0)
            if len(i):
                j = i[0]
                gecikme.append(f["sure"][j])
                teshis = Counter(f["tahmin"][j - N + 1:j + 1]).most_common(1)[0][0]
                dogru_teshis.append(teshis == f["tip"])
                tip_satirlari.append({"model": model, "N": N, "ariza_tipi": f["tip"], "tespit": True,
                                      "gecikme_s": f["sure"][j], "dogru_teshis": teshis == f["tip"]})
            else:
                tip_satirlari.append({"model": model, "N": N, "ariza_tipi": f["tip"], "tespit": False,
                                      "gecikme_s": np.nan, "dogru_teshis": False})
        satirlar.append({
            "model": model, "N": N, "gecikme_alt_siniri_s": (N - 1) * adim_s,
            "yanlis_alarm_olay_saatte": np.mean([k["olay"] / k["saat"] for k in kat.values()]),
            "yanlis_alarm_pencere_saatte": np.mean([k["pencere"] / k["saat"] for k in kat.values()]),
            "tespit_edilen_dosya_orani": np.mean(tespit),
            "tespit_gecikmesi_ortanca_s": float(np.median(gecikme)) if gecikme else np.nan,
            "alarm_aninda_dogru_teshis": float(np.mean(dogru_teshis)) if dogru_teshis else np.nan,
        })

s = pd.DataFrame(satirlar)
tip = pd.DataFrame(tip_satirlari)

# ============================================================================
# N SEÇİMİ (önceden belirlenen ölçüt)
# ============================================================================
secim, gerekce = {}, {}
for model in modeller:
    uygun = s[(s["model"] == model) & (s["tespit_edilen_dosya_orani"] >= TESPIT_HEDEFI)]
    hedefte = uygun[uygun["yanlis_alarm_olay_saatte"] <= YANLIS_ALARM_HEDEFI]
    if len(hedefte):
        secim[model], gerekce[model] = int(hedefte["N"].min()), "yanlış alarm hedefine ulaşan en küçük N"
    elif len(uygun):
        secim[model], gerekce[model] = int(uygun["N"].max()), "yanlış alarm hedefine ulaşılamadı → tespit koşulunu sağlayan en büyük N"
    else:
        secim[model], gerekce[model] = None, f"hiçbir N'de %{100 * TESPIT_HEDEFI:.0f} tespit koşulu sağlanamadı → N seçilmedi"

pd.set_option("display.width", 250)
print(f"=== Alarm kuralı — {'ayarlı' if AYARLI else 'varsayılan'} modeller, {SET}, çapraz doğrulama ===")
print(f"Ölçüt: arızalı dosyaların en az %{100 * TESPIT_HEDEFI:.0f}'i yakalanırken, yanlış alarm olayı/saat ≤ "
      f"{YANLIS_ALARM_HEDEFI} olan en küçük N (ulaşılamazsa koşulu sağlayan en büyük N)\n")
print("Kontrol: N = 1 (alarm kuralı yok) → 'yanlis_alarm_pencere_saatte' ml_ozet'teki yanlış alarm/saat ile aynı olmalı\n")
kolon = ["N", "gecikme_alt_siniri_s", "yanlis_alarm_olay_saatte", "yanlis_alarm_pencere_saatte",
         "tespit_edilen_dosya_orani", "tespit_gecikmesi_ortanca_s", "alarm_aninda_dogru_teshis"]
for model in modeller:
    m = s[s["model"] == model].set_index("N")
    goster = sorted({1, 2, 3, 4, 6, 8, 12, 18, 24, 30} | ({secim[model]} if secim[model] else set()))
    print(f"--- {model}  (seçilen N = {secim[model]}: {gerekce[model]}) ---")
    print(m.loc[goster, kolon[1:]].round(3).to_string(), "\n")

print("=== ÖZET: alarm kuralı yok (N = 1) → seçilen N ===")
ozet = []
for model in modeller:
    a, b = s[(s["model"] == model) & (s["N"] == 1)].iloc[0], None
    if secim[model]:
        b = s[(s["model"] == model) & (s["N"] == secim[model])].iloc[0]
    ozet.append({"model": model, "secilen_N": secim[model],
                 "olay_saatte_N1": a["yanlis_alarm_olay_saatte"],
                 "olay_saatte_secilen": b["yanlis_alarm_olay_saatte"] if b is not None else np.nan,
                 "tespit_N1": a["tespit_edilen_dosya_orani"],
                 "tespit_secilen": b["tespit_edilen_dosya_orani"] if b is not None else np.nan,
                 "gecikme_N1_s": a["tespit_gecikmesi_ortanca_s"],
                 "gecikme_secilen_s": b["tespit_gecikmesi_ortanca_s"] if b is not None else np.nan,
                 "dogru_teshis_secilen": b["alarm_aninda_dogru_teshis"] if b is not None else np.nan})
print(pd.DataFrame(ozet).round(3).to_string(index=False))

print("\n=== Seçilen N'de arıza tipine göre: tespit oranı / ortanca gecikme (s) / alarm anında doğru teşhis ===")
for model in modeller:
    if not secim[model]:
        continue
    x = tip[(tip["model"] == model) & (tip["N"] == secim[model])]
    g = x.groupby("ariza_tipi").agg(dosya=("tespit", "size"), tespit=("tespit", "mean"),
                                    gecikme_s=("gecikme_s", "median"), dogru_teshis=("dogru_teshis", "mean"))
    if model == "IsolationForest":
        g = g.drop(columns="dogru_teshis")                 # arıza tipi söyleyemez
    print(f"--- {model} (N = {secim[model]}) ---")
    print(g.round(2).to_string(), "\n")

# ============================================================================
# KAYDET + GRAFİK
# ============================================================================
s.to_csv(os.path.join(RAPOR, f"alarm_kurali{ek}.csv"), sep=";", index=False)
with open(os.path.join(RAPOR, f"alarm_kurali_secim{ek}.json"), "w", encoding="utf-8") as f:
    json.dump({"tespit_hedefi": TESPIT_HEDEFI, "yanlis_alarm_hedefi": YANLIS_ALARM_HEDEFI, "set": SET,
               "N": secim, "gerekce": gerekce}, f, ensure_ascii=False, indent=1)

renk = {"XGBoost": "#2a6fbb", "RandomForest": "#e08a1e", "SVM": "#3a9a5b", "IsolationForest": "#8a5cc2"}
isaret = {"XGBoost": "o", "RandomForest": "s", "SVM": "^", "IsolationForest": "D"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for model in modeller:
    m = s[s["model"] == model]
    for a_, y, olcek in ((ax[0], m["yanlis_alarm_olay_saatte"], 1), (ax[1], m["tespit_edilen_dosya_orani"], 100)):
        a_.plot(m["N"], y * olcek, color=renk.get(model), marker=isaret.get(model), markersize=4,
                linewidth=1.8, label=model)
        if secim[model]:
            k = m[m["N"] == secim[model]]
            a_.plot(k["N"], k[y.name] * olcek, marker=isaret.get(model), markersize=10, color=renk.get(model),
                    markeredgecolor="black", linestyle="none")
ax[0].set_yscale("symlog", linthresh=1)            # 0 da görünsün diye: 0–1 doğrusal, üstü logaritmik
ax[0].axhline(YANLIS_ALARM_HEDEFI, color="#999999", linestyle="--", linewidth=1)
ax[0].set_ylabel("yanlış alarm olayı / saat (0–1 doğrusal, üstü log)")
ax[0].set_title("Yanlış alarm (sağlam sürüşler)", fontsize=10)
ax[1].axhline(100 * TESPIT_HEDEFI, color="#999999", linestyle="--", linewidth=1)
ax[1].text(N_ADAYLARI[0], 100 * TESPIT_HEDEFI - 1, f"hedef %{100 * TESPIT_HEDEFI:.0f}", ha="left", va="top",
           fontsize=8, color="#555555")
ax[1].set_ylabel("tespit edilen arızalı dosya (%)")
ax[1].set_title("Arıza tespiti (arızalı dosyalar)", fontsize=10)
for a_ in ax:
    a_.set_xlabel("N (art arda arıza denen pencere sayısı)")
    a_.grid(color="#e5e5e5", linewidth=0.6)
    a_.set_axisbelow(True)
    for side in ("top", "right"):
        a_.spines[side].set_visible(False)
ax[1].legend(fontsize=8, frameon=False)
fig.suptitle(f"Alarm kuralı — büyük işaret: seçilen N ({'ayarlı' if AYARLI else 'varsayılan'} modeller, {SET})",
             fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RAPOR, f"alarm_kurali{ek}.png"), dpi=120)
plt.close(fig)
print(f"\nKaydedildi: reports/model/alarm_kurali{ek}.csv, alarm_kurali_secim{ek}.json, alarm_kurali{ek}.png")
