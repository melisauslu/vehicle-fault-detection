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
from ortak import (ANA_KLASOR, SEED, EN_IYI_PARAMETRELER, veriyi_oku, bolmeler, kat_sabitleri, fizik_ekle,
                   ozellik_listesi, model_kur, denetimli_egit_tahmin, if_egit_skor, if_kalibrasyon_bolmesi)

RAPOR = os.path.join(ANA_KLASOR, "reports", "model")
DENEMELER = os.path.join(RAPOR, "hiperparametre_denemeleri.csv")

# EŞİT BÜTÇELİ HİPERPARAMETRE AYARI
# KAYNAKTAN [Bergstra & Bengio 2012, JMLR 13, 281–305]: rastgele arama, aynı deneme sayısında ızgara aramadan
#   daha iyi ayar bulur (önemli olan birkaç parametrenin daha çok farklı değeri denenir).
# BİZİM KARARIMIZ:
#   - Her modele AYNI deneme sayısı (eşit bütçe) — süre değil deneme sayısı eşitlenir.
#   - Deneme 0 = varsayılan ayarlar (ml_modeller.py'deki), kalan denemeler arama uzayından rastgele.
#     Böylece ayarlı model çapraz doğrulamada varsayılandan kötü olamaz.
#   - Ölçüt: 5 katın ortalama macro-F1'i (denetimli), ortalama AUC (Isolation Forest — arıza tipi söyleyemez).
#   - Aynı 5 kat, aynı kat sabitleri, aynı özellik seti (ana set) — modeller arasında tek fark model ayarı.
# DİKKAT (tezde yazılmalı): ayar aynı 5 katla seçilip aynı katlarla raporlandığı için ayarlı çapraz doğrulama
#   skoru biraz iyimserdir [Cawley & Talbot 2010, JMLR 11, 2079–2107]. Tarafsız sonuç: 7 test sürüşündeki son test.
# Test sürüşlerine dokunulmaz. Kesilirse kaldığı yerden devam eder (biten denemeler CSV'den okunur).

# ============================================================================
# AYARLAR
# ============================================================================
OZELLIK_GRUPLARI = ["ham_istatistik", "fizik_dusuk", "uzun_gecmis"]   # ana set: ham+fizik+gecmis
# Modeller ayrı ayrı da çalıştırılabilir (ör. ["RandomForest"]); sonuçlar aynı CSV'de birikir
MODELLER = ["SVM", "IsolationForest"]
DENEME_SAYISI = 20          # model başına (deneme 0 = varsayılan dahil)

# Arama uzayları — her modelin en çok ayarlanan parametreleri, yaygın aralıklar (BİZİM KARARIMIZ).
# Adlar ortak.py/model_kur'daki boru hattı adımlarına göre (set_params biçimi).
ARAMA_UZAYI = {
    "RandomForest": {
        "randomforestclassifier__n_estimators": [100, 300, 500],
        "randomforestclassifier__max_depth": [None, 10, 20, 30],          # None = sınırsız
        "randomforestclassifier__min_samples_leaf": [1, 2, 5, 10],
        "randomforestclassifier__max_features": ["sqrt", "log2", 0.3],    # her bölmede bakılan özellik oranı
        "randomforestclassifier__class_weight": ["balanced", "balanced_subsample"],
    },
    "XGBoost": {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [3, 4, 6, 8, 10],
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": [0.6, 0.8, 1.0],              # her ağaçta kullanılan pencere oranı
        "colsample_bytree": [0.5, 0.7, 1.0],       # her ağaçta kullanılan özellik oranı
        "min_child_weight": [1, 3, 5, 10],
        "reg_lambda": loguniform(0.1, 10),         # yaprak değerlerinin L2 cezası (λ)
    },
    "SVM": {
        "svc__C": loguniform(0.1, 1000),           # marj genişliği ↔ eğitim hatası dengesi
        "svc__gamma": loguniform(1e-4, 1e-1),      # RBF çekirdeğinin genişliği (varsayılan "scale" ≈ 1/özellik sayısı)
    },
    "IsolationForest": {
        "isolationforest__n_estimators": [100, 300, 500],
        "isolationforest__max_samples": [256, 512, 1024, 4096],   # her ağacın gördüğü pencere sayısı
        "isolationforest__max_features": [0.5, 0.75, 1.0],
    },
}


def denemeler(ad):
    # Deneme listesi her çalıştırmada AYNI üretilir (aynı tohum) → kaldığı yerden devam edilebilir
    rastgele = ParameterSampler(ARAMA_UZAYI[ad], n_iter=DENEME_SAYISI - 1, random_state=SEED)
    return [{}] + [{k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()} for p in rastgele]


# ============================================================================
# VERİ: her kat için özellikler, o katın sabitleriyle BİR KEZ hazırlanır (bütün denemeler aynısını kullanır)
# ============================================================================
oz, bilgi = veriyi_oku()
cv, _ = bolmeler(oz)
kolonlar = ozellik_listesi(bilgi, OZELLIK_GRUPLARI)
siniflar = sorted(oz["ariza_tipi"].unique())
adim_saat = bilgi["adim_satir"] * 0.1 / 3600
katlar = []
for egitim, dogrulama in cv:
    o = fizik_ekle(oz, kat_sabitleri(oz, egitim, bilgi))
    X = o[kolonlar].to_numpy(np.float32)
    y = o["ariza_tipi"].to_numpy()
    ogren, kalib = if_kalibrasyon_bolmesi(o, egitim)
    katlar.append({"X_tr": X[egitim], "y_tr": y[egitim], "X_va": X[dogrulama], "y_va": y[dogrulama],
                   "X_ogren": X[ogren], "X_kalib": X[kalib]})
print(f"Özellik seti: {'+'.join(OZELLIK_GRUPLARI)} ({len(kolonlar)} özellik), {len(katlar)} kat, "
      f"model başına {DENEME_SAYISI} deneme\n")


def dene(ad, parametreler):
    # Bir ayarı 5 katta dener → katların skorları + bilgi amaçlı tespit ölçütleri
    skor, tespit_f1, alarm, sure = [], [], [], 0.0
    for k in katlar:
        model = model_kur(ad, parametreler)
        if ad == "IsolationForest":
            esik, s, egitim_s, _ = if_egit_skor(model, k["X_ogren"], k["X_kalib"], k["X_va"])
            skor.append(roc_auc_score(k["y_va"] != "saglam", s))
            tahmin = np.where(s > esik, "ariza", "saglam")
        else:
            tahmin, egitim_s, _ = denetimli_egit_tahmin(ad, model, k["X_tr"], k["y_tr"], k["X_va"])
            skor.append(f1_score(k["y_va"], tahmin, labels=siniflar, average="macro", zero_division=0))
        g, t = k["y_va"] != "saglam", tahmin != "saglam"
        tespit_f1.append(f1_score(g, t, zero_division=0))
        alarm.append(np.sum(~g & t) / (np.sum(~g) * adim_saat))
        sure += egitim_s
    return {"skor_ort": np.mean(skor), "skor_std": np.std(skor), **{f"skor_kat{i}": v for i, v in enumerate(skor)},
            "tespit_f1_ort": np.mean(tespit_f1), "yanlis_alarm_saatte_ort": np.mean(alarm), "egitim_s_toplam": sure}


# ============================================================================
# ARAMA (kaldığı yerden devam eder)
# ============================================================================
os.makedirs(RAPOR, exist_ok=True)
biten = pd.read_csv(DENEMELER, sep=";") if os.path.exists(DENEMELER) else pd.DataFrame(columns=["model", "deneme"])
for ad in MODELLER:
    olcut = "AUC" if ad == "IsolationForest" else "macro-F1"
    print(f"=== {ad} ({olcut}) ===")
    t_basla = time.time()
    for i, p in enumerate(denemeler(ad)):
        if ((biten["model"] == ad) & (biten["deneme"] == i)).any():
            continue
        t0 = time.time()
        r = {"model": ad, "deneme": i, "parametreler": json.dumps(p), **dene(ad, p)}
        pd.DataFrame([r]).to_csv(DENEMELER, sep=";", index=False, mode="a", header=not os.path.exists(DENEMELER))
        biten = pd.concat([biten, pd.DataFrame([r])], ignore_index=True)
        print(f"  deneme {i:2d}/{DENEME_SAYISI - 1}  {olcut} {r['skor_ort']:.3f} ± {r['skor_std']:.3f}  "
              f"tespit-F1 {r['tespit_f1_ort']:.3f}  yanlış alarm/saat {r['yanlis_alarm_saatte_ort']:6.1f}  "
              f"({time.time() - t0:.0f} s)  {'varsayılan' if not p else json.dumps(p)}")
    print(f"  bitti ({(time.time() - t_basla) / 60:.1f} dk)\n")

# ============================================================================
# EN İYİ AYARLAR → hiperparametre_en_iyi.json (ml_modeller.py AYARLI = True ile kullanır)
# Yalnızca bu çalıştırmadakiler değil, CSV'de denemesi olan BÜTÜN modeller yazılır — modeller ayrı ayrı
# (ör. bugün RF, yarın XGBoost) çalıştırılınca öncekinin ayarı kaybolmasın diye.
# ============================================================================
d = biten.copy()
d["deneme"] = d["deneme"].astype(int)
en_iyi, ozet = {}, []
for ad in [m for m in ARAMA_UZAYI if m in set(d["model"])]:
    m = d[d["model"] == ad]
    if len(m) < DENEME_SAYISI:
        print(f"UYARI: {ad} için {len(m)}/{DENEME_SAYISI} deneme var — eşit bütçe için tamamlanmalı")
    b = m.loc[m["skor_ort"].idxmax()]
    v = m[m["deneme"] == 0].iloc[0]
    en_iyi[ad] = json.loads(b["parametreler"])
    ozet.append({"model": ad, "olcut": "AUC" if ad == "IsolationForest" else "macro-F1",
                 "varsayilan": v["skor_ort"], "ayarli": b["skor_ort"], "fark": b["skor_ort"] - v["skor_ort"],
                 "ayarli_std": b["skor_std"], "en_iyi_deneme": int(b["deneme"]),
                 "varsayilan_alarm_saatte": v["yanlis_alarm_saatte_ort"], "ayarli_alarm_saatte": b["yanlis_alarm_saatte_ort"]})
with open(EN_IYI_PARAMETRELER, "w", encoding="utf-8") as f:
    json.dump(en_iyi, f, ensure_ascii=False, indent=1)

pd.set_option("display.width", 250)
print("=== ÖZET: varsayılan → ayarlı (5 katın ortalaması; ayarlı skor biraz iyimser, bkz. yukarıdaki not) ===")
print(pd.DataFrame(ozet).round(3).to_string(index=False))
print("\nSeçilen ayarlar:")
for ad, p in en_iyi.items():
    print(f"  {ad}: {'varsayılan (hiçbir deneme daha iyi değil)' if not p else json.dumps(p)}")
print(f"\nKaydedildi: reports/model/hiperparametre_denemeleri.csv, hiperparametre_en_iyi.json")
