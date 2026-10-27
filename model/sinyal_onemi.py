import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import ANA_KLASOR, veriyi_oku, bolmeler, kat_sabitleri, fizik_ekle, ozellik_listesi, sensoru_kullanir

RAPOR = os.path.join(ANA_KLASOR, "reports", "model")

# HANGİ SENSÖR NE KADAR ÖNEMLİ? — sensör bazında permütasyon önemi
# KAYNAKTAN [Breiman 2001, Random Forests, §10]: bir özelliğin değerleri karıştırılır (permütasyon),
#   modelin başarısı ne kadar düşerse özellik o kadar önemlidir.
# BİZİM KARARIMIZ: tek tek özellik yerine bir SENSÖRÜN BÜTÜN özellikleri BİRLİKTE karıştırılır
#   (ör. akımın ort, std, eğim... + akımı kullanan fizik kalıntıları). Aynı sensörün özellikleri birbirine
#   çok benzediğinden tek tek karıştırmak önemi küçük gösterir (biri karışınca model diğerini kullanır).
# Neden ml_modeller.py'deki "özellik önemi" (Gini) yetmiyor: Gini önemi eğitim verisinden hesaplanır ve
#   birbirine benzeyen özellikler arasında bölünür; permütasyon önemi DOĞRULAMA sürüşlerinde ölçülür.
# DİKKAT (tezde yazılmalı): sonuç "bizim enjekte ettiğimiz arızaları bulmak için hangi sensör önemli"
#   demektir. Arıza eklenen sensörler (akım, gerilim, sıcaklık, ısıtıcı gücü, debi) doğal olarak öne çıkar.
# Yalnızca çapraz doğrulama katları kullanılır (test sürüşlerine dokunulmaz).

# ============================================================================
# AYARLAR
# ============================================================================
MODEL = "XGBoost"           # "XGBoost" ya da "RandomForest" (ayarlar ml_modeller.py ile aynı)
OZELLIK_GRUPLARI = ["ham_istatistik", "fizik_dusuk", "uzun_gecmis"]   # ana sonuç seti
SET_ADI = "ham+fizik+gecmis"
TEKRAR = 3                  # her sensör kaç kez karıştırılıp ortalaması alınır
SEED = 42


def model_kur(ad):
    if ad == "RandomForest":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                    n_jobs=-1, random_state=SEED))
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                         n_jobs=-1, random_state=SEED)


oz, bilgi = veriyi_oku()
cv, _ = bolmeler(oz)
siniflar = sorted(oz["ariza_tipi"].unique())
kolonlar = ozellik_listesi(bilgi, OZELLIK_GRUPLARI)

# Her sensör için karıştırılacak sütunlar: kendi istatistikleri + onu kullanan kalıntılar
gruplar = {s: [c for c in kolonlar if sensoru_kullanir(c, [s])] for s in bilgi["ham_kanallar"]}

rng = np.random.default_rng(SEED)
satirlar = []
for k, (egitim, dogrulama) in enumerate(cv):
    o = fizik_ekle(oz, kat_sabitleri(oz, egitim, bilgi))
    X_tr, X_va = o.loc[egitim, kolonlar].to_numpy(), o.loc[dogrulama, kolonlar].to_numpy()
    y_tr, y_va = o.loc[egitim, "ariza_tipi"].to_numpy(), o.loc[dogrulama, "ariza_tipi"].to_numpy()
    model = model_kur(MODEL)
    if MODEL == "XGBoost":
        egitim_siniflari = np.unique(y_tr)                    # XGBoost sınıfları 0..k−1 sayıları ister
        model.fit(X_tr, np.searchsorted(egitim_siniflari, y_tr), sample_weight=compute_sample_weight("balanced", y_tr))
    else:
        model.fit(X_tr, y_tr)

    def tahmin(X):
        t = model.predict(X)
        return egitim_siniflari[t] if MODEL == "XGBoost" else t

    def skorlar(t):
        return (f1_score(y_va, t, labels=siniflar, average="macro", zero_division=0),
                f1_score(y_va, t, labels=siniflar, average=None, zero_division=0))

    taban_macro, taban_sinif = skorlar(tahmin(X_va))
    print(f"kat{k}: karıştırmadan macro-F1 = {taban_macro:.3f}")
    for s, cols in gruplar.items():
        idx = [kolonlar.index(c) for c in cols]
        for _ in range(TEKRAR):
            X = X_va.copy()
            X[:, idx] = X[rng.permutation(len(X))][:, idx]   # sütunlar birlikte (aynı sırayla) karıştırılır
            m, sf = skorlar(tahmin(X))
            satirlar.append({"kat": k, "sensor": s, "macro_dusus": taban_macro - m,
                             **{c: taban_sinif[i] - sf[i] for i, c in enumerate(siniflar)}})

d = pd.DataFrame(satirlar)
kat_ort = d.groupby(["sensor", "kat"]).mean().reset_index()      # önce her katta tekrarların ortalaması
ozet = (kat_ort.groupby("sensor")["macro_dusus"].agg(["mean", "std"])
        .sort_values("mean", ascending=False))
sinif = kat_ort.groupby("sensor")[siniflar].mean().loc[ozet.index]

pd.set_option("display.width", 250)
print(f"\n=== {MODEL}, {SET_ADI} — sensör karıştırılınca macro-F1 ne kadar düşüyor? ({len(cv)} katın ortalaması) ===")
print("    (büyük düşüş = model bu sensöre çok güveniyor; ~0 = bu sensör olmadan da aynı başarı)")
print(ozet.round(3).to_string())
print("\n=== Arıza tipine göre F1 düşüşü (hangi sensör hangi arıza için önemli?) ===")
print(sinif.round(2).to_string())

os.makedirs(RAPOR, exist_ok=True)
ozet.to_csv(os.path.join(RAPOR, f"sinyal_onemi_{MODEL}.csv"), sep=";")
sinif.to_csv(os.path.join(RAPOR, f"sinyal_onemi_sinif_{MODEL}.csv"), sep=";")

# Grafik 1: sensör sıralaması (tek renk, katlar arası standart sapma çizgisiyle)
fig, ax = plt.subplots(figsize=(8, 5.5))
y = np.arange(len(ozet))[::-1]
ax.barh(y, ozet["mean"], xerr=ozet["std"], color="#2a6fbb", height=0.6,
        error_kw={"ecolor": "#555555", "elinewidth": 1, "capsize": 2})
ax.set_yticks(y, ozet.index, fontsize=8)
ax.set_xlabel("macro-F1 düşüşü (sensör karıştırılınca)")
ax.set_title(f"Sensör önemi — {MODEL}, {SET_ADI} ({len(cv)} kat ortalaması ± std)", fontsize=10)
ax.axvline(0, color="#999999", linewidth=0.8)
ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RAPOR, f"sinyal_onemi_{MODEL}.png"), dpi=120)
plt.close(fig)

# Grafik 2: sensör × arıza tipi (tek renk tonu: açık = etkisiz, koyu = çok önemli)
fig, ax = plt.subplots(figsize=(10, 6))
deger = sinif.clip(lower=0).to_numpy()
im = ax.imshow(deger, cmap="Blues", vmin=0, vmax=max(deger.max(), 0.01), aspect="auto")
ax.set_xticks(range(len(siniflar)), siniflar, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(sinif)), sinif.index, fontsize=8)
for i in range(deger.shape[0]):
    for j in range(deger.shape[1]):
        if deger[i, j] >= 0.01:
            ax.text(j, i, f"{deger[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if deger[i, j] > 0.6 * deger.max() else "black")
ax.set_title(f"Hangi sensör hangi arıza için önemli? (F1 düşüşü) — {MODEL}, {SET_ADI}", fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.03, label="F1 düşüşü")
plt.tight_layout()
plt.savefig(os.path.join(RAPOR, f"sinyal_onemi_sinif_{MODEL}.png"), dpi=120)
plt.close(fig)

print(f"\nKaydedildi: reports/model/sinyal_onemi_{MODEL}.csv/.png, sinyal_onemi_sinif_{MODEL}.csv/.png")
