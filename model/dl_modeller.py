import json
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import ANA_KLASOR, SEED, veriyi_oku, bolmeler, degerlendir
from dl_ortak import (CIHAZ, VARSAYILAN, ham_dizileri_oku, kat_verisi, egit_ve_tahmin_et, en_iyi_parametreleri_oku,
                      atomik_csv)

RAPOR = os.path.join(ANA_KLASOR, "reports", "model")
TAHMIN = os.path.join(ANA_KLASOR, "data", "model")

# DERİN ÖĞRENME MODELLERİ — ml_modeller.py'nin DL karşılığı (aynı katlar, aynı ölçütler: ortak.degerlendir)
# KALDIĞI YERDEN DEVAM: her (bölme, set, model) bittiği anda sonucu ayrı bir dosyaya yazılır
#   (data/model/dl_parcalar<ek>/). Kod tekrar çalıştırılınca biten parçalar atlanır. Bilgisayar/Colab kapanırsa
#   en fazla o an eğitilmekte olan TEK parça (bir model, bir kat) baştan yapılır.

# ============================================================================
# AYARLAR
# ============================================================================
OZELLIK_SETLERI = ["ham", "ham+fizik+gecmis"]          # ham: yalnızca sinyaller; ikincisi ML ana setinin karşılığı
DENETIMLI = ["CNN", "GRU", "LSTM"]                       # 10 sınıf (sağlam + 9 arıza)
DENETIMSIZ = ["Autoencoder"]                             # yalnızca sağlam pencerelerle; yalnızca "arıza var mı"
AYARLI = False                                           # True → dl_hiperparametre_ayari.py'nin seçtiği ayarlar
SON_TEST = False                                         # YALNIZCA bütün kararlar bittikten sonra, bir kez True

# ============================================================================
oz, bilgi = veriyi_oku()
X_ham, X_fizik = ham_dizileri_oku()
assert len(X_ham) == len(oz), "pencereler_ham.npz ile pencereler_ozellik.csv uyuşmuyor — veri_hazirla.py yeniden çalışmalı"
cv, son = bolmeler(oz)
adim_s = bilgi["adim_satir"] * 0.1
siniflar = sorted(oz["ariza_tipi"].unique())
ek = ("_ayarli" if AYARLI else "") + ("_son_test" if SON_TEST else "")
PARCA = os.path.join(TAHMIN, f"dl_parcalar{ek}")
os.makedirs(PARCA, exist_ok=True)
os.makedirs(RAPOR, exist_ok=True)

if SON_TEST:
    bolme_listesi = [("son_test", son[0].to_numpy(), son[1].to_numpy())]
    print("!!! SON TEST: 28 eğitim sürüşüyle eğitilip 7 test sürüşünde ölçülüyor !!!\n")
else:
    bolme_listesi = [(f"kat{i}", tr.to_numpy(), va.to_numpy()) for i, (tr, va) in enumerate(cv)]
parametreler = en_iyi_parametreleri_oku() if AYARLI else {}
modeller = DENETIMLI + DENETIMSIZ
print(f"Cihaz: {CIHAZ}   Model ayarları: {'AYARLI (dl_hiperparametre_en_iyi.json)' if AYARLI else 'varsayılan'}")
for ad in modeller:
    p = parametreler.get(ad)
    if AYARLI and p is None:
        print(f"  {ad}: !!! AYAR YOK — dl_hiperparametre_ayari.py bu model için çalışmadı, VARSAYILAN kullanılıyor")
    print(f"  {ad}: {json.dumps(p if p is not None else VARSAYILAN[ad])}")
print(f"Ara sonuçlar: {PARCA} (kesilirse kaldığı yerden devam eder)\n")


def parca_adi(bolme, set_adi, ad):
    return os.path.join(PARCA, f"{bolme}__{set_adi.replace('+', '-')}__{ad}")


meta_kolon = ["dosya", "ariza_tipi", "ariza_siddeti", "ariza_suresi_s", "pencere_bitis_s"]
for b_no, (bolme, egitim, dogrulama) in enumerate(bolme_listesi):
    for set_adi in OZELLIK_SETLERI:
        k = None                                          # kat verisi yalnızca gerekirse hazırlanır
        for ad in modeller:
            yol = parca_adi(bolme, set_adi, ad)
            if os.path.exists(yol + "_sonuc.csv") and os.path.exists(yol + "_tahmin.csv"):
                print(f"  {bolme} {set_adi:17s} {ad:12s} zaten var → atlandı")
                continue
            if k is None:
                k = kat_verisi(oz, X_ham, X_fizik, bilgi, egitim, dogrulama, set_adi, siniflar, SEED + b_no)
            p = parametreler.get(ad) or VARSAYILAN[ad]
            tahmin, skor, bilgi_ = egit_ve_tahmin_et(ad, k, p, siniflar, SEED + b_no)
            meta_va = oz.loc[dogrulama, meta_kolon].reset_index(drop=True)
            y_va = meta_va["ariza_tipi"].to_numpy()
            r = {"bolme": bolme, "model": ad, "ozellik_seti": set_adi,
                 "macro_f1": np.nan if skor is not None else
                 f1_score(y_va, tahmin, labels=siniflar, average="macro", zero_division=0),
                 **degerlendir(meta_va, y_va, tahmin, adim_s),
                 "auc": roc_auc_score(y_va != "saglam", skor) if skor is not None else np.nan,
                 **bilgi_, "parametreler": json.dumps(p)}
            # önce tahminler, en son sonuç satırı yazılır: sonuç dosyası varsa parça tamamdır
            atomik_csv(meta_va.assign(bolme=bolme, model=ad, ozellik_seti=set_adi, tahmin=tahmin), yol + "_tahmin.csv")
            atomik_csv(pd.DataFrame([r]), yol + "_sonuc.csv")
            olcut = f"AUC {r['auc']:.3f}      " if skor is not None else f"macro-F1 {r['macro_f1']:.3f}"
            print(f"  {bolme} {set_adi:17s} {ad:12s} {olcut}  tespit-F1 {r['tespit_f1']:.3f}  "
                  f"yanlış alarm/saat {r['yanlis_alarm_saatte']:6.1f}  ({r['egitim_s']:.0f} s, {r['epoch']} epoch)",
                  flush=True)
    print()

# ============================================================================
# BİRLEŞTİR, KAYDET, ÖZETLE (yalnızca bütün parçalar bittiyse)
# ============================================================================
beklenen = [parca_adi(b, s, a) for b, _, _ in bolme_listesi for s in OZELLIK_SETLERI for a in modeller]
eksik = [y for y in beklenen if not os.path.exists(y + "_sonuc.csv")]
if eksik:
    sys.exit(f"{len(eksik)} parça eksik — kodu tekrar çalıştır, kaldığı yerden devam eder.")

s = pd.concat([pd.read_csv(y + "_sonuc.csv", sep=";") for y in beklenen], ignore_index=True)
t = pd.concat([pd.read_csv(y + "_tahmin.csv", sep=";") for y in beklenen], ignore_index=True)
s.to_csv(os.path.join(RAPOR, f"dl_sonuclar{ek}.csv"), sep=";", index=False)
t.to_csv(os.path.join(TAHMIN, f"dl_tahminler{ek}.csv"), sep=";", index=False)

olcutler = ["macro_f1", "auc", "tespit_f1", "tespit_recall", "tespit_precision", "yanlis_alarm_saatte",
            "tespit_edilen_dosya_orani", "tespit_gecikmesi_ortanca_s", "egitim_s", "tahmin_ms_pencere", "epoch"]
s.groupby(["model", "ozellik_seti"])[olcutler].agg(["mean", "std"]).round(3).to_csv(
    os.path.join(RAPOR, f"dl_ozet{ek}.csv"), sep=";")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(f"=== ÖZET (katların ortalaması; süreler {s['cihaz'].iloc[0]} üzerinde) ===")
print(s.groupby(["model", "ozellik_seti"])[olcutler].mean().round(3).to_string())

sinif_f1 = []
for (b, ad, set_adi), x in t.groupby(["bolme", "model", "ozellik_seti"]):
    if ad in DENETIMLI:
        f1s = f1_score(x["ariza_tipi"], x["tahmin"], labels=siniflar, average=None, zero_division=0)
        sinif_f1 += [{"bolme": b, "model": ad, "ozellik_seti": set_adi, "sinif": c, "f1": f} for c, f in zip(siniflar, f1s)]
sf = pd.DataFrame(sinif_f1)
sf.to_csv(os.path.join(RAPOR, f"dl_sinif_f1{ek}.csv"), sep=";", index=False)
print("\n=== Arıza tipine göre F1 (katların ortalaması) ===")
print(sf.groupby(["model", "ozellik_seti", "sinif"])["f1"].mean().unstack("sinif").round(2).to_string())

for ad in DENETIMLI:
    for set_adi in OZELLIK_SETLERI:
        x = t[(t["model"] == ad) & (t["ozellik_seti"] == set_adi)]
        cm = confusion_matrix(x["ariza_tipi"], x["tahmin"], labels=siniflar, normalize="true")
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(siniflar)), siniflar, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(siniflar)), siniflar, fontsize=8)
        for i in range(len(siniflar)):
            for j in range(len(siniflar)):
                if cm[i, j] >= 0.01:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        ax.set_title(f"{ad} – {set_adi}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(os.path.join(RAPOR, f"karisiklik_{ad}_{set_adi}{ek}.png"), dpi=120)
        plt.close(fig)

print(f"\nKaydedildi: reports/model/ (dl_sonuclar{ek}, dl_ozet{ek}, dl_sinif_f1{ek}, karisiklik_*), "
      f"data/model/dl_tahminler{ek}.csv")
