import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Bütün model kodlarının ortak kullandığı fonksiyonlar.
# VERİ SIZINTISI KURALI: veriden öğrenilen her şey (sabitler, normalizasyon, model) yalnızca o katın
# EĞİTİM pencerelerinden öğrenilir; doğrulama/test pencerelerine yalnızca uygulanır.

ANA_KLASOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERI = os.path.join(ANA_KLASOR, "data", "model")
EN_IYI_PARAMETRELER = os.path.join(ANA_KLASOR, "reports", "model", "hiperparametre_en_iyi.json")
DT = 0.1
SEED = 42
# SVM eğitim süresi örnek sayısının ~karesiyle büyüdüğü için her katta en fazla bu kadar eğitim penceresiyle
# (sınıf oranları korunarak) eğitilir — laptopta makul süre için (BİZİM KARARIMIZ, tezde belirtilmeli)
SVM_MAKS_ORNEK = 12000
# Isolation Forest alarm eşiği: sağlam kalibrasyon pencerelerinin %1'i alarm verecek şekilde (BİZİM KARARIMIZ)
IF_YANLIS_ALARM = 0.01


# ============================================================================
# MODELLER — ml_modeller.py, hiperparametre_ayari.py ve sinyal_onemi.py HEPSİ bunları kullanır,
# böylece her kod birebir aynı modeli aynı şekilde eğitir.
# ============================================================================
def model_kur(ad, parametreler=None):
    # Varsayılan ayarlar (yaygın değerler); parametreler verilirse (hiperparametre ayarından) üzerine yazılır.
    # Parametre adları set_params biçiminde: boru hattındaki adımın adı + "__" + parametre (ör. svc__C)
    if ad == "RandomForest":
        # boş değerler (ör. direnc_orani, akım az değişince) eğitimin ortancasıyla doldurulur + "boştu" işareti
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 n_jobs=-1, random_state=SEED))
    elif ad == "XGBoost":
        # XGBoost boş değerlerle kendisi çalışabilir
        m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                          n_jobs=-1, random_state=SEED)
    elif ad == "SVM":
        # SVM ölçeğe duyarlı → normalizasyon (StandardScaler) yalnızca eğitimde öğrenilir (sızıntı kuralı 2)
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          SVC(C=10, gamma="scale", class_weight="balanced"))
    elif ad == "IsolationForest":
        m = make_pipeline(SimpleImputer(strategy="median"),
                          IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1))
    else:
        raise ValueError(ad)
    if parametreler:
        m.set_params(**parametreler)
    return m


def denetimli_egit_tahmin(ad, model, X_tr, y_tr, X_va):
    # Denetimli modeli eğitir, doğrulama pencerelerini tahmin eder → (tahmin edilen sınıf adları, eğitim s, ms/pencere)
    t0 = time.time()
    if ad == "SVM" and len(X_tr) > SVM_MAKS_ORNEK:
        Xs, _, ys, _ = train_test_split(X_tr, y_tr, train_size=SVM_MAKS_ORNEK, stratify=y_tr, random_state=SEED)
        model.fit(Xs, ys)
    elif ad == "XGBoost":
        # XGBoost sütun adlarında "[ ]" kabul etmiyor → sayı dizisi olarak verilir (sıra aynı)
        # XGBoost sınıfları 0..k−1 sayıları ister → bu katın eğitimindeki sınıflar numaralanır
        egitim_siniflari = np.unique(y_tr)
        model.fit(np.asarray(X_tr), np.searchsorted(egitim_siniflari, y_tr),
                  sample_weight=compute_sample_weight("balanced", y_tr))   # sınıf dengesizliği
    else:
        model.fit(X_tr, y_tr)
    egitim_s = time.time() - t0
    t0 = time.time()
    tahmin = model.predict(np.asarray(X_va) if ad == "XGBoost" else X_va)
    tahmin_ms = (time.time() - t0) / len(X_va) * 1000
    if ad == "XGBoost":
        tahmin = egitim_siniflari[tahmin]
    return tahmin, egitim_s, tahmin_ms


def if_egit_skor(model, X_ogren, X_kalib, X_va):
    # Isolation Forest: yalnızca SAĞLAM pencerelerle öğrenir; eşik, ayrı sağlam kalibrasyon pencerelerinden
    # (doğrulamaya bakılmaz) → (eşik, doğrulama anormallik skorları [büyük = anormal], eğitim s, ms/pencere)
    t0 = time.time()
    model.fit(X_ogren)
    egitim_s = time.time() - t0
    esik = np.quantile(-model.decision_function(X_kalib), 1 - IF_YANLIS_ALARM)
    t0 = time.time()
    skor = -model.decision_function(X_va)
    tahmin_ms = (time.time() - t0) / len(X_va) * 1000
    return esik, skor, egitim_s, tahmin_ms


def if_kalibrasyon_bolmesi(o, egitim):
    # Isolation Forest için eğitim sürüşlerinin sağlam pencereleri ikiye ayrılır: ilk eğitim katı → eşik
    # kalibrasyonu, diğerleri → öğrenme (eşik, modeli öğrendiği pencerelerle belirlenmesin diye)
    egitim_katlari = sorted(o.loc[egitim, "kat"].unique())
    kalib = egitim & (o["kat"] == egitim_katlari[0]) & (o["ariza"] == 0)
    ogren = egitim & (o["kat"] != egitim_katlari[0]) & (o["ariza"] == 0)
    return ogren, kalib


def en_iyi_parametreleri_oku():
    # hiperparametre_ayari.py'nin seçtiği ayarlar (yoksa hata: önce o kod çalışmalı)
    with open(EN_IYI_PARAMETRELER, encoding="utf-8") as f:
        return json.load(f)


def veriyi_oku():
    oz = pd.read_csv(os.path.join(VERI, "pencereler_ozellik.csv"), sep=";", low_memory=False)
    with open(os.path.join(VERI, "bilgi.json"), encoding="utf-8") as f:
        bilgi = json.load(f)
    return oz, bilgi


def bolmeler(oz):
    # Sürüş bazında bölmeler. Aynı sürüşün sağlam ve arızalı bütün pencereleri hep aynı tarafta kalır.
    #   5 çapraz doğrulama katı: eğitim = diğer 4 kat, doğrulama = o kat (son test sürüşleri hiç girmez)
    #   son değerlendirme:      eğitim = 28 eğitim sürüşü, test = 7 test sürüşü (kat = −1)
    katlar = sorted(k for k in oz["kat"].unique() if k >= 0)
    cv = [((oz["kat"] >= 0) & (oz["kat"] != k), oz["kat"] == k) for k in katlar]
    son = ((oz["kat"] >= 0), (oz["kat"] == -1))
    return cv, son


def kat_sabitleri(oz, egitim, bilgi):
    # Fizik kalıntılarının sabitleri — YALNIZCA bu katın eğitim pencerelerindeki SAĞLAM pencerelerden
    s = oz[egitim & (oz["ariza"] == 0)]
    P, dT = s["isitici_guc_pozitif__ort"], s["su_isinmasi__ort"]
    # K = ṁ·c_p [Tez-O Eq 4.26]: ısıtıcı açıkken P/ΔT (koşullar estimate_flow.py ile aynı)
    m = (P > 1000) & (dT > 3)
    K = float(np.median(P[m] / dT[m]))
    # Q [Tez-B Eq 3.26]: ∫I dt / ΔSoC (pencereler üst üste binse de oran değişmez)
    Q = float(s["akim_Ah"].sum() / (s["soc_degisim"].sum() / 100))
    # beklenen V–I eğimi = a·sıcaklık + b [Rint; sıcaklık ilişkisi check_battery: r = −0.96]
    e = s[["vi_egimi", "Battery Temperature [°C]__ort"]].dropna()
    R_fit = np.polyfit(e["Battery Temperature [°C]__ort"], e["vi_egimi"], 1)
    return {"K": K, "Q": Q, "R_fit": R_fit}


def fizik_ekle(oz, sabitler):
    # Sabite bağlı kalıntılar (bu katın sabitleriyle) — tanımlar veri_hazirla.py'de
    o = oz.copy()
    o["su_isinmasi_farki__ort"] = o["su_isinmasi__ort"] - o["isitici_guc_pozitif__ort"] / sabitler["K"]
    o["soc_farki"] = o["soc_degisim"] - 100 * o["akim_Ah"] / sabitler["Q"]
    o["direnc_orani"] = o["vi_egimi"] / np.polyval(sabitler["R_fit"], o["Battery Temperature [°C]__ort"])
    if "soc_degisim_5dk" in o:                             # uzun geçmiş özelliği (veri_hazirla.py'de UZUN)
        o["soc_farki_5dk"] = o["soc_degisim_5dk"] - 100 * o["akim_Ah_5dk"] / sabitler["Q"]
    return o


# Fizik kalıntısı → hesaplanırken kullandığı sensörler (tanımlar veri_hazirla.py'de).
# Bir sensör çıkarılınca/karıştırılınca onu kullanan kalıntılar da çıkar/karışır (o sensör yoksa hesaplanamazlar).
V_BAT, I_BAT, T_BAT, SOC = "Battery Voltage [V]", "Battery Current [A]", "Battery Temperature [°C]", "SoC [%]"
ISI_P, ISTENEN = "Heating Power LIN [W]", "Requested Heating Power [W]"
ISI_V, ISI_I = "Heater Voltage [V]", "Heater Current [A]"
GIRIS, CIKIS = "Temperature Coolant Heater Inlet [°C]", "Temperature Coolant Heater Outlet [°C]"
KALINTI_SENSORLERI = {
    "bat_guc": [V_BAT, I_BAT],
    "gerilim_farki": [ISI_V, V_BAT],
    "isitici_VI_farki": [ISI_V, ISI_I, ISI_P],
    "isitici_guc_farki": [ISTENEN, ISI_P],
    "su_isinmasi": [CIKIS, GIRIS],
    "soc_farki": [SOC, I_BAT],
    "soc_farki_5dk": [SOC, I_BAT],
    "su_isinmasi_farki": [CIKIS, GIRIS, ISI_P],
    "direnc_orani": [V_BAT, I_BAT, T_BAT],
}


def sensoru_kullanir(kolon, sensorler):
    # Bu özellik verilen sensörlerden birinin istatistiği mi, ya da onlardan birini kullanan bir kalıntı mı?
    kok = kolon.split("__")[0]
    return kok in sensorler or any(s in sensorler for s in KALINTI_SENSORLERI.get(kok, []))


def ozellik_listesi(bilgi, gruplar, cikarilan=()):
    # gruplar: ["ham_istatistik"], ["ham_istatistik", "fizik_dusuk"], ya da üçü birden (+ "fizik_riskli")
    # cikarilan: modelden çıkarılacak sensörler (kendi istatistikleri + onları kullanan kalıntılar çıkar)
    return [c for g in gruplar for c in bilgi["ozellik_gruplari"][g] if not sensoru_kullanir(c, cikarilan)]
