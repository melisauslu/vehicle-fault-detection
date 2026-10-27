import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import ANA_KLASOR, veriyi_oku, bolmeler, kat_sabitleri, fizik_ekle, ozellik_listesi

RAPOR = os.path.join(ANA_KLASOR, "reports", "model")
TAHMIN = os.path.join(ANA_KLASOR, "data", "model")

# ============================================================================
# AYARLAR
# ============================================================================
# Dört özellik seti — her grubun katkısı aynı çalıştırmada ölçülür:
#   fizik kalıntılarının katkısı (ham → ham+fizik), uzun geçmişin katkısı (→ +gecmis),
#   döngüsellik riski (→ +riskli; ana sonuç DEĞİL, üst sınır)
OZELLIK_SETLERI = {
    "ham": ["ham_istatistik"],
    "ham+fizik": ["ham_istatistik", "fizik_dusuk"],
    "ham+fizik+gecmis": ["ham_istatistik", "fizik_dusuk", "uzun_gecmis"],
    "ham+fizik+gecmis+riskli": ["ham_istatistik", "fizik_dusuk", "uzun_gecmis", "fizik_riskli"],
}
ANA_SET = "ham+fizik+gecmis"     # özellik önemi bu set için yazdırılır
DENETIMLI = ["RandomForest", "XGBoost", "SVM"]     # etiketli: arıza var mı + hangi arıza (10 sınıf)
DENETIMSIZ = ["IsolationForest"]                   # yalnızca sağlam pencerelerle eğitilir: arıza var mı
SEED = 42

# Sinyal seçimi deneyi: buraya yazılan sensörler modelden çıkarılır (kendi istatistikleri + onları kullanan
# fizik kalıntıları). Boş liste = bütün sensörler. Adaylar sinyal_onemi.py ile belirlenir; karar çapraz
# doğrulamayla verilir, son testte doğrulanır.
# Örnek: ["Velocity [km/h]", "Ambient Temperature [°C]", "Heater Current [A]"]
CIKARILAN_SINYALLER = []

# BİZİM KARARIMIZ: hiperparametre ayarı YAPILMADI; yaygın varsayılanlar sabit tutuldu.
# Böylece hiçbir ayar doğrulama/test sonucuna bakılarak seçilmedi (sızıntı kuralı 5).
# SVM eğitim süresi örnek sayısının ~karesiyle büyüdüğü için her katta en fazla bu kadar
# eğitim penceresiyle (sınıf oranları korunarak) eğitilir — laptopta makul süre için. Tezde belirtilmeli.
SVM_MAKS_ORNEK = 12000
# Isolation Forest alarm eşiği: sağlam kalibrasyon pencerelerinin %1'i alarm verecek şekilde (BİZİM KARARIMIZ)
IF_YANLIS_ALARM = 0.01

# Son test (7 sürüş) YALNIZCA bütün kararlar bittikten sonra, bir kez True yapılır (sızıntı kuralı 6)
SON_TEST = False


def denetimli_model(ad):
    if ad == "RandomForest":
        # boş değerler (ör. direnc_orani, akım az değişince) eğitimin ortancasıyla doldurulur + "boştu" işareti eklenir
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                    n_jobs=-1, random_state=SEED))
    if ad == "XGBoost":
        # XGBoost boş değerlerle kendisi çalışabilir
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                             n_jobs=-1, random_state=SEED)
    if ad == "SVM":
        # SVM ölçeğe duyarlı → normalizasyon (StandardScaler) yalnızca eğitimde öğrenilir (sızıntı kuralı 2)
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             SVC(C=10, gamma="scale", class_weight="balanced"))
    raise ValueError(ad)


def degerlendir(meta, gercek, tahmin, adim_s):
    # meta: doğrulama pencerelerinin bilgileri; gercek/tahmin: sınıf adları ("saglam" ya da arıza tipi)
    g_ariza, t_ariza = gercek != "saglam", tahmin != "saglam"
    yanlis_alarm = np.sum(~g_ariza & t_ariza)
    saglam_saat = np.sum(~g_ariza) * adim_s / 3600            # her pencere 5 s'lik yeni bir zamanı temsil eder
    # tespit gecikmesi: her arızalı dosyada ilk alarmın, arıza başladıktan kaç saniye sonra verildiği
    d = meta.assign(alarm=t_ariza)[g_ariza]
    ilk = d[d["alarm"]].groupby("dosya")["ariza_suresi_s"].min()
    return {
        "tespit_precision": precision_score(g_ariza, t_ariza, zero_division=0),
        "tespit_recall": recall_score(g_ariza, t_ariza, zero_division=0),
        "tespit_f1": f1_score(g_ariza, t_ariza, zero_division=0),
        "yanlis_alarm_orani": yanlis_alarm / max(np.sum(~g_ariza), 1),
        "yanlis_alarm_saatte": yanlis_alarm / max(saglam_saat, 1e-9),
        "tespit_edilen_dosya_orani": len(ilk) / max(d["dosya"].nunique(), 1),
        "tespit_gecikmesi_ortanca_s": float(ilk.median()) if len(ilk) else np.nan,
    }


# ============================================================================
# VERİ VE BÖLMELER
# ============================================================================
oz, bilgi = veriyi_oku()
cv, son = bolmeler(oz)
adim_s = bilgi["adim_satir"] * 0.1
siniflar = sorted(oz["ariza_tipi"].unique())             # 10 sınıf: sağlam + 9 arıza tipi
if SON_TEST:
    bolme_listesi = [("son_test", son[0], son[1])]
    print("!!! SON TEST: 28 eğitim sürüşüyle eğitilip 7 test sürüşünde ölçülüyor !!!\n")
else:
    bolme_listesi = [(f"kat{i}", tr, va) for i, (tr, va) in enumerate(cv)]
if CIKARILAN_SINYALLER:
    print(f"Modelden çıkarılan sinyaller: {CIKARILAN_SINYALLER}\n")

os.makedirs(RAPOR, exist_ok=True)
sonuclar, sinif_f1, tahminler, onem = [], [], [], []

for bolme, egitim, dogrulama in bolme_listesi:
    # Fizik sabitleri YALNIZCA bu bölmenin eğitim pencerelerinden (sızıntı kuralı 1)
    sab = kat_sabitleri(oz, egitim, bilgi)
    o = fizik_ekle(oz, sab)
    meta_va = o.loc[dogrulama, ["dosya", "ariza_tipi", "ariza_siddeti", "ariza_suresi_s", "pencere_bitis_s"]]
    y_tr, y_va = o.loc[egitim, "ariza_tipi"].to_numpy(), o.loc[dogrulama, "ariza_tipi"].to_numpy()
    print(f"=== {bolme}: eğitim {egitim.sum()} pencere ({o.loc[egitim, 'surus'].nunique()} sürüş), "
          f"doğrulama {dogrulama.sum()} pencere ({o.loc[dogrulama, 'surus'].nunique()} sürüş) ===")

    for set_adi, gruplar in OZELLIK_SETLERI.items():
        kolonlar = ozellik_listesi(bilgi, gruplar, CIKARILAN_SINYALLER)
        X_tr, X_va = o.loc[egitim, kolonlar], o.loc[dogrulama, kolonlar]

        # ---- DENETİMLİ: 10 sınıf (sağlam + 9 arıza); "arıza var mı" sonucu buradan türetilir ----
        for ad in DENETIMLI:
            model = denetimli_model(ad)
            t0 = time.time()
            if ad == "SVM" and len(X_tr) > SVM_MAKS_ORNEK:
                Xs, _, ys, _ = train_test_split(X_tr, y_tr, train_size=SVM_MAKS_ORNEK, stratify=y_tr, random_state=SEED)
                model.fit(Xs, ys)
            elif ad == "XGBoost":
                # XGBoost sütun adlarında "[ ]" kabul etmiyor → sayı dizisi olarak verilir (sıra aynı)
                # XGBoost sınıfları 0..k−1 sayıları ister → bu katın eğitimindeki sınıflar numaralanır
                egitim_siniflari = np.unique(y_tr)
                model.fit(X_tr.to_numpy(), np.searchsorted(egitim_siniflari, y_tr),
                          sample_weight=compute_sample_weight("balanced", y_tr))   # sınıf dengesizliği
            else:
                model.fit(X_tr, y_tr)
            egitim_s = time.time() - t0
            t0 = time.time()
            tahmin = model.predict(X_va.to_numpy() if ad == "XGBoost" else X_va)
            tahmin_ms = (time.time() - t0) / len(X_va) * 1000
            if ad == "XGBoost":
                tahmin = egitim_siniflari[tahmin]

            r = {"bolme": bolme, "model": ad, "ozellik_seti": set_adi, "ozellik_sayisi": len(kolonlar),
                 "macro_f1": f1_score(y_va, tahmin, labels=siniflar, average="macro", zero_division=0),
                 **degerlendir(meta_va, y_va, tahmin, adim_s),
                 "egitim_s": egitim_s, "tahmin_ms_pencere": tahmin_ms}
            sonuclar.append(r)
            f1s = f1_score(y_va, tahmin, labels=siniflar, average=None, zero_division=0)
            sinif_f1 += [{"bolme": bolme, "model": ad, "ozellik_seti": set_adi, "sinif": s, "f1": f}
                         for s, f in zip(siniflar, f1s)]
            tahminler.append(meta_va.assign(bolme=bolme, model=ad, ozellik_seti=set_adi, tahmin=tahmin))
            # hangi özellik ne kadar önemli (yalnızca ağaç modelleri)
            if ad in ("RandomForest", "XGBoost"):
                if ad == "RandomForest":
                    adlar = model[0].get_feature_names_out(kolonlar)
                    deger = model[-1].feature_importances_
                else:
                    adlar, deger = kolonlar, model.feature_importances_
                onem += [{"bolme": bolme, "model": ad, "ozellik_seti": set_adi, "ozellik": a, "onem": v}
                         for a, v in zip(adlar, deger)]
            print(f"  {ad:15s} {set_adi:23s} macro-F1 {r['macro_f1']:.3f}  tespit-F1 {r['tespit_f1']:.3f}  "
                  f"yanlış alarm/saat {r['yanlis_alarm_saatte']:6.1f}  ({egitim_s:.0f} s)")

        # ---- DENETİMSİZ: yalnızca SAĞLAM eğitim pencereleriyle; yalnızca "arıza var mı" ----
        for ad in DENETIMSIZ:
            # Eşik, eğitim katlarından birinin sağlam pencereleriyle belirlenir (doğrulamaya bakılmaz)
            egitim_katlari = sorted(o.loc[egitim, "kat"].unique())
            kalib = egitim & (o["kat"] == egitim_katlari[0]) & (o["ariza"] == 0)
            ogren = egitim & (o["kat"] != egitim_katlari[0]) & (o["ariza"] == 0)
            model = make_pipeline(SimpleImputer(strategy="median"),
                                  IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1))
            t0 = time.time()
            model.fit(o.loc[ogren, kolonlar])
            egitim_s = time.time() - t0
            esik = np.quantile(-model.decision_function(o.loc[kalib, kolonlar]), 1 - IF_YANLIS_ALARM)
            t0 = time.time()
            skor = -model.decision_function(X_va)                 # büyük = daha anormal
            tahmin_ms = (time.time() - t0) / len(X_va) * 1000
            tahmin = np.where(skor > esik, "ariza", "saglam")
            r = {"bolme": bolme, "model": ad, "ozellik_seti": set_adi, "ozellik_sayisi": len(kolonlar),
                 "macro_f1": np.nan,                               # hangi arıza olduğunu söyleyemez
                 **degerlendir(meta_va, y_va, tahmin, adim_s),
                 "auc": roc_auc_score(y_va != "saglam", skor),     # eşikten bağımsız ayırma başarısı
                 "egitim_s": egitim_s, "tahmin_ms_pencere": tahmin_ms}
            sonuclar.append(r)
            tahminler.append(meta_va.assign(bolme=bolme, model=ad, ozellik_seti=set_adi, tahmin=tahmin))
            print(f"  {ad:15s} {set_adi:23s} AUC {r['auc']:.3f}       tespit-F1 {r['tespit_f1']:.3f}  "
                  f"yanlış alarm/saat {r['yanlis_alarm_saatte']:6.1f}  ({egitim_s:.0f} s)")
    print()

# ============================================================================
# KAYDET VE ÖZETLE
# ============================================================================
ek = "_son_test" if SON_TEST else ""
s = pd.DataFrame(sonuclar)
s.to_csv(os.path.join(RAPOR, f"ml_sonuclar{ek}.csv"), sep=";", index=False)
pd.DataFrame(sinif_f1).to_csv(os.path.join(RAPOR, f"ml_sinif_f1{ek}.csv"), sep=";", index=False)
pd.concat(tahminler).to_csv(os.path.join(TAHMIN, f"ml_tahminler{ek}.csv"), sep=";", index=False)

olcutler = ["macro_f1", "tespit_f1", "tespit_recall", "tespit_precision", "yanlis_alarm_saatte",
            "tespit_edilen_dosya_orani", "tespit_gecikmesi_ortanca_s", "egitim_s", "tahmin_ms_pencere"]
ozet = s.groupby(["model", "ozellik_seti"])[olcutler].agg(["mean", "std"]).round(3)
ozet.to_csv(os.path.join(RAPOR, f"ml_ozet{ek}.csv"), sep=";")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("=== ÖZET (katların ortalaması) ===")
print(s.groupby(["model", "ozellik_seti"])[olcutler].mean().round(3).to_string())

sf = pd.DataFrame(sinif_f1).groupby(["model", "ozellik_seti", "sinif"])["f1"].mean().unstack("sinif").round(2)
print("\n=== Arıza tipine göre F1 (katların ortalaması) ===")
print(sf.to_string())

if onem:
    on = (pd.DataFrame(onem).groupby(["model", "ozellik_seti", "ozellik"])["onem"].mean().reset_index()
          .sort_values("onem", ascending=False))
    on.to_csv(os.path.join(RAPOR, f"ml_ozellik_onemi{ek}.csv"), sep=";", index=False)
    print(f"\n=== En önemli 10 özellik (RandomForest, {ANA_SET}) ===")
    print(on[(on["model"] == "RandomForest") & (on["ozellik_seti"] == ANA_SET)]
          .head(10)[["ozellik", "onem"]].round(4).to_string(index=False))

# Karışıklık matrisleri (bütün katların tahminleri birlikte, satır = gerçek, oranlar)
t = pd.concat(tahminler)
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

print(f"\nKaydedildi: reports/model/ (ml_sonuclar, ml_ozet, ml_sinif_f1, ml_ozellik_onemi, karisiklik_*.png), "
      f"data/model/ml_tahminler{ek}.csv")
