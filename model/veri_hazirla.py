import glob
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import skew, kurtosis

ANA_KLASOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMIZ = os.path.join(ANA_KLASOR, "data", "clean_B")
ARIZALI = os.path.join(ANA_KLASOR, "data", "faulty")
SPLIT = os.path.join(ANA_KLASOR, "split.csv")
CIKTI = os.path.join(ANA_KLASOR, "data", "model")

# ============================================================================
# AYARLAR
# ============================================================================
DT = 0.1                 # 10 Hz
PENCERE = 300            # 30 s  (BİZİM KARARIMIZ: başlangıç değeri; 10 s ve 60 s de denenecek)
ADIM = 50                # 5 s
DL_SEYREKLESTIR = 5      # derin öğrenme için her 5. satır (2 Hz): sensörler zaten 0.2–1 s'de güncelleniyor
KAT_SAYISI = 5           # eğitim sürüşlerinde sürüş bazında 5 katlı çapraz doğrulama
SEED = 42

V_BAT, I_BAT, T_BAT, SOC = "Battery Voltage [V]", "Battery Current [A]", "Battery Temperature [°C]", "SoC [%]"
ISI_P, ISTENEN = "Heating Power LIN [W]", "Requested Heating Power [W]"
ISI_V, ISI_I = "Heater Voltage [V]", "Heater Current [A]"
GIRIS, CIKIS = "Temperature Coolant Heater Inlet [°C]", "Temperature Coolant Heater Outlet [°C]"
DEBI, HIZ, DIS = "Coolant Volume Flow +500 [l/h]", "Velocity [km/h]", "Ambient Temperature [°C]"

# Modele girecek ham sinyaller. Etiket sütunları (ariza, ariza_tipi, ...) KESİNLİKLE girmez.
# Kabin sıcaklığı alınmadı: arızalar onu değiştirmiyor (sınırlılık), model bu tutarsızlığı kullanmasın
HAM = [V_BAT, I_BAT, T_BAT, SOC, ISI_P, ISTENEN, ISI_V, ISI_I, GIRIS, CIKIS, DEBI, HIZ, DIS]

# Satır bazında yeni sütunlar (kalıntılar) — her biri sağlam veride doğrulanmış bir fizik ilişkisi
# risk: "dusuk" = arıza eklerken kullanılmayan ilişki; "riskli" = arıza enjeksiyonundaki sabitle aynı
YENI = {
    "bat_guc":           "dusuk",   # V·I [Tez-B Eq 3.8] — bağlam bilgisi
    "gerilim_farki":     "dusuk",   # V_ısıtıcı − V_bat: aynı hattaki iki sensör (oran 1.007 doğrulandı)
    "isitici_VI_farki":  "dusuk",   # V_ısıtıcı·I_ısıtıcı − P [Eq 3.8] (oran 0.999, inspect_signals)
    "isitici_guc_farki": "dusuk",   # istenen − gerçek güç [Meckel Eq 3: sağlıklıdan sapma] (check_heater)
    "su_isinmasi":       "dusuk",   # T_çıkış − T_giriş: iki ham sinyalin farkı
}
# Bir SABİTE bağlı kalıntılar (K, Q, beklenen direnç). VERİ SIZINTISI olmasın diye burada HESAPLANMAZ:
# sabitler her katta yalnızca o katın eğitim pencerelerinden öğrenilmeli → model kodunda ortak.py/fizik_ekle().
# Burada yalnızca yapı taşları kaydedilir (yardımcı sütunlar, modele doğrudan girmez):
#   su_isinmasi_farki__ort = su_isinmasi__ort − isitici_guc_pozitif__ort / K   [Tez-O Eq 4.26]   (riskli)
#   soc_farki              = soc_degisim − 100·akim_Ah / Q                      [Tez-B Eq 3.26]   (düşük risk)
#   direnc_orani           = vi_egimi / (a·T_bat + b)                           [Rint]            (riskli)
KAT_BAGIMLI = {"su_isinmasi_farki__ort": "riskli", "soc_farki": "dusuk", "direnc_orani": "riskli"}
YARDIMCI = ["isitici_guc_pozitif__ort", "soc_degisim", "akim_Ah", "vi_egimi", "soc_degisim_5dk", "akim_Ah_5dk"]

# UZUN GEÇMİŞ ÖZELLİKLERİ — pencerenin bitişinden GERİYE doğru son 5 dakikaya bakar (yalnızca geçmiş veri;
# araçta o anda elde olan bilgi). Gerekçe: zor_analiz.py → 30 s pencerede sürüklenme sabit kayma gibi görünüyor
# (30 s'deki artış k·30 s gürültünün altında), takılı sensör de sağlam sensörden ayırt edilemiyor.
# BİZİM KARARIMIZ: 5 dk (3000 satır). Sürüşün ilk 5 dakikasında eldeki geçmiş kullanılır (en az 30 s, pencere
#   boyu) — böylece boş değer oluşmaz; "boş = sürüş başı = sağlam" gibi bir kısayol öğrenilemez.
#   gerilim_farki__egim_5dk  V_ısıtıcı − V_bat farkının 5 dk eğimi [V/s]. Aynı hattaki iki sensör: sabit kaymada
#                            fark sabit kalır (eğim ≈ 0), sürüklenmede zamanla büyür [Wang §2.3: x + k·t]
#   soc_farki_5dk            5 dk'da ΔSoC − 100·∫I dt / Q [Tez-B Eq 3.26]. Akım sensörü hatası 30 s'de küçük,
#                            5 dk'da birikir. Q kata bağlı → ortak.py/fizik_ekle() hesaplar (yapı taşları YARDIMCI'da)
#   <sensör>__degismeden_s   V, I, T için değerin son değiştiği andan beri geçen süre [s], en fazla 300 s
#                            [Wang §2.3: takılma x̂ = x(t0) → değer hiç değişmez]
GECMIS = 3000
DEGISMEDEN = [V_BAT, I_BAT, T_BAT]
UZUN = [f"{kn}__degismeden_s" for kn in DEGISMEDEN] + ["gerilim_farki__egim_5dk"]
UZUN_KAT_BAGIMLI = ["soc_farki_5dk"]

# ============================================================================
# VERİYİ OKU
# ============================================================================
split = pd.read_csv(SPLIT, sep=";")
grup = dict(zip(split["dosya"], split["grup"]))
liste = pd.read_csv(os.path.join(ARIZALI, "ariza_listesi.csv"), sep=";")
liste = liste[liste["kullanilabilir"]]                      # görünmeyen arızalar çıkarılır (A seçeneği)

saglam = {os.path.basename(f): pd.read_csv(f, sep=";") for f in sorted(glob.glob(os.path.join(TEMIZ, "*.csv")))}
egitim_surusleri = sorted(a for a in saglam if grup.get(a) == "egitim")
test_surusleri = sorted(a for a in saglam if grup.get(a) == "test")

# Sürüş bazında 5 kat: yalnızca eğitim sürüşleri bölünür; test sürüşleri (split.csv) en sona saklanır
rng = np.random.default_rng(SEED)
karisik = list(rng.permutation(egitim_surusleri))
kat = {ad: i % KAT_SAYISI for i, ad in enumerate(karisik)}
kat.update({ad: -1 for ad in test_surusleri})              # -1 = son test

def yeni_sutunlar(d):
    # Satır bazında kalıntılar (formüllerin kaynağı yukarıdaki YENI sözlüğünde)
    return pd.DataFrame({
        "bat_guc": d[V_BAT] * d[I_BAT],
        "gerilim_farki": d[ISI_V] - d[V_BAT],
        "isitici_VI_farki": d[ISI_V] * d[ISI_I] - d[ISI_P],
        "isitici_guc_farki": d[ISTENEN] - d[ISI_P],
        "su_isinmasi": d[CIKIS] - d[GIRIS],
    })


def pencereler(x, baslangiclar):
    # x: (satır, kanal) → (pencere, PENCERE, kanal); kopyasız görünüm
    v = sliding_window_view(x, PENCERE, axis=0)            # (satır−PENCERE+1, kanal, PENCERE)
    return np.transpose(v[baslangiclar], (0, 2, 1))


def egim(W):
    # her kanal için pencere boyunca doğrusal eğim [birim / s] (en küçük kareler)
    t = (np.arange(PENCERE) - (PENCERE - 1) / 2) * DT
    return np.einsum("t,ntc->nc", t, W - W.mean(axis=1, keepdims=True)) / np.sum(t ** 2)


# Pencerede gerilimin akıma göre eğimi (Rint'teki R'nin pencere tahmini). Akım yeterince değişmiyorsa hesaplanmaz.
MIN_AKIM_STD = 10.0      # BİZİM KARARIMIZ: pencerede akım std < 10 A ise eğim güvenilmez → NaN


def vi_egimi(W, i_v, i_i):
    V, I = W[:, :, i_v], W[:, :, i_i]
    Vc, Ic = V - V.mean(1, keepdims=True), I - I.mean(1, keepdims=True)
    e = np.sum(Vc * Ic, 1) / np.maximum(np.sum(Ic ** 2, 1), 1e-9)
    return np.where(I.std(1) >= MIN_AKIM_STD, e, np.nan)


kanallar = HAM + list(YENI)
iv, ii, it = kanallar.index(V_BAT), kanallar.index(I_BAT), kanallar.index(T_BAT)


def pencere_ozellikleri(W):
    # W: (pencere, PENCERE, kanal) tam çözünürlük. Her kanal için istatistikler + pencere kalıntıları
    # İstatistikler Chen vd. 2026'daki zaman alanı özelliklerine benzer (ortalama, std, tepe, çarpıklık, basıklık)
    o = {}
    fark = np.diff(W, axis=1)
    istat = {
        "ort": W.mean(1), "std": W.std(1), "min": W.min(1), "max": W.max(1),
        "egim": egim(W),                                   # sürüklenme
        "son_ilk": W[:, -1] - W[:, 0],
        "degisim_orani": (np.abs(fark) > 1e-9).mean(1),    # takılma → 0
        "ort_mutlak_fark": np.abs(fark).mean(1),           # doğruluk kaybı (gürültü)
    }
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")                    # neredeyse sabit pencerelerde scipy uyarısı
        istat["carpiklik"] = np.nan_to_num(skew(W, axis=1))     # sabit pencerede tanımsız → 0
        istat["basiklik"] = np.nan_to_num(kurtosis(W, axis=1))
    for ad, deger in istat.items():
        for c, kn in enumerate(kanallar):
            o[f"{kn}__{ad}"] = deger[:, c]
    # sabite bağlı kalıntıların yapı taşları (yukarıdaki KAT_BAGIMLI açıklamasına bak)
    soc, akim = W[:, :, kanallar.index(SOC)], W[:, :, ii]
    o["isitici_guc_pozitif__ort"] = np.clip(W[:, :, kanallar.index(ISI_P)], 0, None).mean(1)
    o["soc_degisim"] = soc[:, -1] - soc[:, 0]
    o["akim_Ah"] = akim.sum(1) * DT / 3600
    o["vi_egimi"] = vi_egimi(W, iv, ii)
    return o


def uzun_gecmis(x, son):
    # x: (satır, kanal) bütün dosya; son: pencerelerin bitiş satırları. Açıklama yukarıda (UZUN).
    bas = np.maximum(son - GECMIS + 1, 0)                  # geçmişin ilk satırı (sürüş başında daha kısa)
    say = son - bas + 1

    def topla(v):                                          # [bas, son] aralığının toplamı (kümülatif toplamla)
        c = np.concatenate([[0.0], np.cumsum(v)])
        return c[son + 1] - c[bas]

    o = {}
    # en küçük kareler eğimi: (Σty − Σt·Σy/n) / (Σt² − (Σt)²/n); t pencere sonuna göre (sayısal doğruluk için)
    y = x[:, kanallar.index("gerilim_farki")]
    egimler = np.empty(len(son))
    for j, (b, s) in enumerate(zip(bas, son)):
        t = (np.arange(b, s + 1) - s) * DT
        tc = t - t.mean()
        egimler[j] = np.dot(tc, y[b:s + 1] - y[b:s + 1].mean()) / np.dot(tc, tc)
    o["gerilim_farki__egim_5dk"] = egimler
    # SoC ve akım yapı taşları (30 s'lik soc_degisim / akim_Ah ile aynı tanım, 5 dk için)
    soc = x[:, kanallar.index(SOC)]
    o["soc_degisim_5dk"] = soc[son] - soc[bas]
    o["akim_Ah_5dk"] = topla(x[:, ii]) * DT / 3600
    # değerin son değiştiği satır → o andan beri geçen süre
    n = np.arange(len(x))
    for kn in DEGISMEDEN:
        v = x[:, kanallar.index(kn)]
        degisti = np.r_[True, np.abs(np.diff(v)) > 1e-9]
        son_degisim = np.maximum.accumulate(np.where(degisti, n, 0))
        o[f"{kn}__degismeden_s"] = np.minimum(son - son_degisim[son], say - 1) * DT
    return o


# ============================================================================
# PENCERELERİ OLUŞTUR
# Sağlam pencereler: yalnızca temiz sürüşlerden. Arızalı dosyaların arıza öncesi kısmı temiz sürüşün
# birebir kopyası olduğu için tekrar alınmaz; arızalı dosyalardan yalnızca SONU arıza başladıktan sonra
# olan pencereler alınır (etiket = pencerenin son anındaki durum).
# ============================================================================
oz_parcalar, meta_parcalar, X_ham, X_fizik = [], [], [], []
ham_idx = list(range(len(HAM)))
fizik_idx = list(range(len(HAM), len(kanallar)))


def isle(d, ad, kaynak, tip, sinyal, siddet, n0):
    x = pd.concat([d[HAM], yeni_sutunlar(d)], axis=1).to_numpy(float)
    bas = np.arange(0, len(x) - PENCERE + 1, ADIM)
    if n0 is not None:
        bas = bas[bas + PENCERE - 1 >= n0]
    W = pencereler(x, bas)
    son = bas + PENCERE - 1
    oz_parcalar.append(pd.DataFrame({**pencere_ozellikleri(W), **uzun_gecmis(x, son)}))
    meta_parcalar.append(pd.DataFrame({
        "dosya": ad, "surus": kaynak, "grup": grup.get(kaynak, "?"), "kat": kat[kaynak],
        "pencere_bitis_s": np.round(son * DT, 1),
        "ariza": 0 if n0 is None else 1,
        "ariza_tipi": "saglam" if n0 is None else tip,
        "ariza_sinyali": "" if n0 is None else sinyal,
        "ariza_siddeti": 0.0 if n0 is None else siddet,
        "ariza_suresi_s": 0.0 if n0 is None else np.round((son - n0 + 1) * DT, 1),   # arıza ne zamandır var
    }))
    Wd = W[:, ::DL_SEYREKLESTIR].astype(np.float32)        # derin öğrenme için 2 Hz
    X_ham.append(Wd[:, :, ham_idx])
    X_fizik.append(Wd[:, :, fizik_idx])


for ad, d in saglam.items():
    isle(d, ad, ad, None, None, None, None)
for _, r in liste.iterrows():
    d = pd.read_csv(os.path.join(ARIZALI, r["dosya"]), sep=";", usecols=HAM + ["ariza"])   # etiket metin sütunları gereksiz
    n0 = int(d["ariza"].to_numpy().argmax())               # arızanın başladığı satır
    isle(d, r["dosya"], r["kaynak"], r["ariza_tipi"], r["ariza_sinyali"], r["ariza_siddeti"], n0)

oz = pd.concat(oz_parcalar, ignore_index=True)
meta = pd.concat(meta_parcalar, ignore_index=True)
X_ham, X_fizik = np.concatenate(X_ham), np.concatenate(X_fizik)

# ============================================================================
# KAYDET
# ============================================================================
os.makedirs(CIKTI, exist_ok=True)
pd.concat([meta, oz], axis=1).to_csv(os.path.join(CIKTI, "pencereler_ozellik.csv"), sep=";", index=False)
np.savez_compressed(os.path.join(CIKTI, "pencereler_ham.npz"), X_ham=X_ham, X_fizik=X_fizik)

ozellik_gruplari = {
    "ham_istatistik": [c for c in oz.columns if c.split("__")[0] in HAM and c not in YARDIMCI + UZUN],
    "fizik_dusuk": [c for c in oz.columns if c.split("__")[0] in YENI and c not in YARDIMCI + UZUN]
                   + [k for k, v in KAT_BAGIMLI.items() if v == "dusuk"],
    "fizik_riskli": [k for k, v in KAT_BAGIMLI.items() if v == "riskli"],
    "uzun_gecmis": UZUN + UZUN_KAT_BAGIMLI,
    "yardimci": YARDIMCI,
}
bilgi = {
    "pencere_satir": PENCERE, "adim_satir": ADIM, "dl_seyreklestir": DL_SEYREKLESTIR,
    "ham_kanallar": HAM, "fizik_kanallar": list(YENI), "ozellik_gruplari": ozellik_gruplari,
    "kat": kat, "kat_bagimli": KAT_BAGIMLI, "min_akim_std": MIN_AKIM_STD, "gecmis_satir": GECMIS,
}
with open(os.path.join(CIKTI, "bilgi.json"), "w", encoding="utf-8") as f:
    json.dump(bilgi, f, ensure_ascii=False, indent=1)

# ============================================================================
# ÖZET
# ============================================================================
pd.set_option("display.width", 200)
print(f"Toplam pencere: {len(meta)}")
print(f"Özellik sayısı — ham istatistik: {len(ozellik_gruplari['ham_istatistik'])}, fizik (düşük risk): "
      f"{len(ozellik_gruplari['fizik_dusuk'])}, fizik (riskli): {len(ozellik_gruplari['fizik_riskli'])}, "
      f"uzun geçmiş: {len(ozellik_gruplari['uzun_gecmis'])}"
      f"  (soc_farki, soc_farki_5dk, su_isinmasi_farki, direnc_orani her katta model kodunda hesaplanır)")
print(f"Derin öğrenme dizileri: X_ham {X_ham.shape}, X_fizik {X_fizik.shape}  (pencere, zaman adımı, kanal)\n")
print("Pencere sayısı, arıza tipi × grup:")
print(pd.crosstab(meta["ariza_tipi"], meta["grup"], margins=True, margins_name="toplam").to_string())
print("\nKatlara göre sürüş ve pencere sayısı (−1 = son test):")
kt = meta.groupby("kat").agg(surus=("surus", "nunique"), pencere=("surus", "size"),
                             arizali_oran=("ariza", "mean")).round(2)
print(kt.to_string())
print(f"\nV–I eğimi hesaplanamayan pencere (akım std < {MIN_AKIM_STD} A): %{100 * oz['vi_egimi'].isna().mean():.0f}")
# Kontrol amaçlı: sabitler eğitim sürüşlerinin sağlam pencerelerinden (model kodunda her katta ayrıca hesaplanır)
from ortak import kat_sabitleri, fizik_ekle
tum = pd.concat([meta, oz], axis=1)
sab = kat_sabitleri(tum, tum["kat"] >= 0, bilgi)
print(f"Kontrol sabitleri (28 eğitim sürüşü): K = {sab['K']:.1f} W/°C, Q = {sab['Q']:.1f} Ah, "
      f"beklenen V–I eğimi = {sab['R_fit'][0]:.4f}·T + {sab['R_fit'][1]:.4f} ohm")
print("\nSağlam pencerelerde kalıntılar (ortanca ve %95 aralığı) — sıfıra/bire yakın olmalı:")
s = fizik_ekle(tum, sab)[meta["ariza"] == 0]
for c in ["gerilim_farki__ort", "isitici_VI_farki__ort", "isitici_guc_farki__ort", "su_isinmasi_farki__ort",
          "soc_farki", "direnc_orani"]:
    q = s[c].quantile([0.025, 0.5, 0.975])
    print(f"  {c:28s} ortanca {q[0.5]:9.3f}   [%2.5: {q[0.025]:9.3f}, %97.5: {q[0.975]:9.3f}]")
# Uzun geçmiş özellikleri işe yarıyor mu? Arıza tipine göre ortanca |değer| — sağlamdan belirgin farklıysa ayırt edici
u = fizik_ekle(tum, sab)
print("\nUzun geçmiş özellikleri, arıza tipine göre ortanca |değer| (sağlam satırıyla karşılaştır):")
print(u.groupby(meta["ariza_tipi"])[ozellik_gruplari["uzun_gecmis"]].agg(lambda v: v.abs().median())
      .rename(columns=lambda c: c.replace("Battery ", "").replace(" [V]", "").replace(" [A]", "")
              .replace(" [°C]", "")).to_string(float_format=lambda v: f"{v:.4g}"))
kayit_mb = os.path.getsize(os.path.join(CIKTI, "pencereler_ham.npz")) / 1e6
print(f"\nKaydedildi: data/model/pencereler_ozellik.csv, pencereler_ham.npz ({kayit_mb:.0f} MB), bilgi.json")
