import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import ANA_KLASOR, SEED, veriyi_oku, bolmeler
from dl_ortak import (CIHAZ, VARSAYILAN, DL_EN_IYI_PARAMETRELER, ham_dizileri_oku, kat_verisi, egit_ve_tahmin_et,
                      atomik_csv)

RAPOR = os.path.join(ANA_KLASOR, "reports", "model")
KATLAR = os.path.join(RAPOR, "dl_hiperparametre_katlar.csv")          # her (model, deneme, kat) bir satır
DENEMELER = os.path.join(RAPOR, "dl_hiperparametre_denemeleri.csv")   # katların ortalaması (her seferinde yeniden)

# DL İÇİN EŞİT BÜTÇELİ HİPERPARAMETRE AYARI — hiperparametre_ayari.py (ML) ile AYNI yöntem:
#   rastgele arama [Bergstra & Bengio 2012], model başına aynı deneme sayısı, deneme 0 = varsayılan,
#   ölçüt: 5 katın ortalama macro-F1'i (Autoencoder: AUC), ana set (ham+fizik+gecmis), test sürüşlerine dokunulmaz.
# KALDIĞI YERDEN DEVAM: her (model, deneme, kat) bittiği anda dl_hiperparametre_katlar.csv'ye bir satır eklenir.
#   Tekrar çalıştırınca var olan satırlar atlanır → kesilirse en fazla o an eğitilen TEK kat baştan yapılır.
# DİKKAT: ARAMA_UZAYI, DENEME_SAYISI veya SEED değiştirilirse deneme listesi değişir; o zaman eski CSV'ler
#   silinmeli/yeniden adlandırılmalı (yoksa eski sonuçlar yeni denemelerle karışır).

# ============================================================================
# AYARLAR
# ============================================================================
SET = "ham+fizik+gecmis"
MODELLER = ["CNN", "GRU", "LSTM", "Autoencoder"]     # ayrı ayrı da çalıştırılabilir, ör. ["CNN"]
DENEME_SAYISI = 20                                    # ML ile aynı (eşit bütçe)

# Arama uzayları — her mimarinin en çok ayarlanan parametreleri, yaygın aralıklar (BİZİM KARARIMIZ)
ARAMA_UZAYI = {
    "CNN": {"n_katman": [2, 3, 4], "filtre": [32, 64, 128], "cekirdek": [3, 5, 7], "dropout": [0.0, 0.2, 0.4],
            "lr": loguniform(1e-4, 3e-3), "paket": [64, 128, 256]},
    "GRU": {"gizli": [32, 64, 128], "katman": [1, 2], "cift_yonlu": [False, True], "dropout": [0.0, 0.2, 0.4],
            "lr": loguniform(1e-4, 3e-3), "paket": [64, 128, 256]},
    "LSTM": {"gizli": [32, 64, 128], "katman": [1, 2], "cift_yonlu": [False, True], "dropout": [0.0, 0.2, 0.4],
             "lr": loguniform(1e-4, 3e-3), "paket": [64, 128, 256]},
    "Autoencoder": {"filtre": [16, 32, 64], "gizil": [4, 8, 16], "dropout": [0.0, 0.1, 0.2],
                    "lr": loguniform(1e-4, 3e-3), "paket": [64, 128, 256]},
}


def denemeler(ad):
    # Her çalıştırmada AYNI liste (aynı tohum); deneme 0 = varsayılan
    r = ParameterSampler(ARAMA_UZAYI[ad], n_iter=DENEME_SAYISI - 1, random_state=SEED)
    return [VARSAYILAN[ad]] + [{k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()} for p in r]


oz, bilgi = veriyi_oku()
X_ham, X_fizik = ham_dizileri_oku()
cv, _ = bolmeler(oz)
siniflar = sorted(oz["ariza_tipi"].unique())
adim_saat = bilgi["adim_satir"] * 0.1 / 3600
os.makedirs(RAPOR, exist_ok=True)
onbellek = {}


def kat(i):
    # kat verisi bir kez hazırlanır, bütün denemeler aynısını kullanır (dl_modeller.py ile aynı tohum: SEED + kat)
    if i not in onbellek:
        tr, va = cv[i]
        onbellek[i] = kat_verisi(oz, X_ham, X_fizik, bilgi, tr.to_numpy(), va.to_numpy(), SET, siniflar, SEED + i)
    return onbellek[i]


def bitenler():
    if not os.path.exists(KATLAR):
        return set()
    d = pd.read_csv(KATLAR, sep=";", on_bad_lines="skip")
    return set(zip(d["model"], d["deneme"].astype(int), d["kat"].astype(int)))


print(f"Cihaz: {CIHAZ}   Set: {SET}   Model başına {DENEME_SAYISI} deneme × {len(cv)} kat")
print(f"Ara sonuçlar: {KATLAR} (kesilirse kaldığı yerden devam eder)\n")
biten = bitenler()
for ad in MODELLER:
    olcut = "AUC" if ad == "Autoencoder" else "macro-F1"
    print(f"=== {ad} ({olcut}) ===")
    for d_no, p in enumerate(denemeler(ad)):
        t0, calisti = time.time(), False
        for i in range(len(cv)):
            if (ad, d_no, i) in biten:
                continue
            calisti = True
            k = kat(i)
            tahmin, skor, b = egit_ve_tahmin_et(ad, k, p, siniflar, SEED + i)
            y_va = oz.loc[cv[i][1].to_numpy(), "ariza_tipi"].to_numpy()
            if skor is not None:
                deger = roc_auc_score(y_va != "saglam", skor)
            else:
                deger = f1_score(y_va, tahmin, labels=siniflar, average="macro", zero_division=0)
            g, t = y_va != "saglam", tahmin != "saglam"
            satir = {"model": ad, "deneme": d_no, "kat": i, "skor": deger, "tespit_f1": f1_score(g, t, zero_division=0),
                     "yanlis_alarm_saatte": np.sum(~g & t) / (np.sum(~g) * adim_saat), "egitim_s": b["egitim_s"],
                     "epoch": b["epoch"], "parametreler": json.dumps(p)}
            pd.DataFrame([satir]).to_csv(KATLAR, sep=";", index=False, mode="a", header=not os.path.exists(KATLAR))
            biten.add((ad, d_no, i))
        d = pd.read_csv(KATLAR, sep=";", on_bad_lines="skip")
        d = d[(d["model"] == ad) & (d["deneme"] == d_no)]
        if len(d) == len(cv) and calisti:                    # yalnızca bu oturumda (en az bir katı) biten denemeler
            print(f"  deneme {d_no:2d}/{DENEME_SAYISI - 1}  {olcut} {d['skor'].mean():.3f} ± {d['skor'].std(ddof=0):.3f}  "
                  f"tespit-F1 {d['tespit_f1'].mean():.3f}  yanlış alarm/saat {d['yanlis_alarm_saatte'].mean():6.1f}  "
                  f"({time.time() - t0:.0f} s)  {'varsayılan' if d_no == 0 else json.dumps(p)}", flush=True)
    print()

# ============================================================================
# DENEME ÖZETİ + EN İYİ AYARLAR (CSV'de denemesi tamamlanmış BÜTÜN modeller — ayrı ayrı çalıştırmaya uygun)
# ============================================================================
d = pd.read_csv(KATLAR, sep=";", on_bad_lines="skip")
ozet = (d.groupby(["model", "deneme"])
        .agg(kat_sayisi=("kat", "nunique"), skor_ort=("skor", "mean"), skor_std=("skor", lambda x: x.std(ddof=0)),
             tespit_f1_ort=("tespit_f1", "mean"), yanlis_alarm_saatte_ort=("yanlis_alarm_saatte", "mean"),
             egitim_s_toplam=("egitim_s", "sum"), parametreler=("parametreler", "first")).reset_index())
atomik_csv(ozet, DENEMELER)
tam = ozet[ozet["kat_sayisi"] == len(cv)]
en_iyi, satirlar = {}, []
for ad in [m for m in ARAMA_UZAYI if m in set(tam["model"])]:
    m = tam[tam["model"] == ad]
    if len(m) < DENEME_SAYISI:
        print(f"UYARI: {ad} için {len(m)}/{DENEME_SAYISI} deneme tamam — eşit bütçe için tamamlanmalı")
    b = m.loc[m["skor_ort"].idxmax()]
    v = m[m["deneme"] == 0]
    en_iyi[ad] = json.loads(b["parametreler"])
    satirlar.append({"model": ad, "olcut": "AUC" if ad == "Autoencoder" else "macro-F1",
                     "varsayilan": v["skor_ort"].iloc[0] if len(v) else np.nan, "ayarli": b["skor_ort"],
                     "ayarli_std": b["skor_std"], "en_iyi_deneme": int(b["deneme"]),
                     "ayarli_alarm_saatte": b["yanlis_alarm_saatte_ort"]})
with open(DL_EN_IYI_PARAMETRELER, "w", encoding="utf-8") as f:
    json.dump(en_iyi, f, ensure_ascii=False, indent=1)
pd.set_option("display.width", 250)
print("=== ÖZET: varsayılan → ayarlı (5 katın ortalaması; ayarlı skor biraz iyimser) ===")
print(pd.DataFrame(satirlar).round(3).to_string(index=False))
print("\nSeçilen ayarlar:")
for ad, p in en_iyi.items():
    print(f"  {ad}: {json.dumps(p)}")
print("\nKaydedildi: reports/model/dl_hiperparametre_katlar.csv, dl_hiperparametre_denemeleri.csv, "
      "dl_hiperparametre_en_iyi.json")
