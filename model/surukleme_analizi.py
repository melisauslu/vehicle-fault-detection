import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import ANA_KLASOR

# SÜRÜKLENME (DRIFT) ARIZASININ ŞİDDETE GÖRE ANALİZİ — model eğitmez, mevcut tahminleri okur.
# Soru: sürüklenme geç/az yakalanıyorsa bunun sebebi model mi, yoksa arızanın fiziksel olarak henüz görünmemesi mi?
# Sürüklenme [Wang §2.3–2.4]: sapma(t) = L · aralık · t / 5000 s  → tam sapma C = L · aralık, 5000 s'de ulaşılır.
# Alarm anındaki sapma = L · aralık · gecikme / 5000  (sinyalin kendi biriminde) ve C'nin yüzdesi = gecikme / 50.
# Alarm kuralı: alarm_kurali.py'nin çapraz doğrulamada seçtiği N (değiştirilmez). Test sürüşleri yalnızca okunur.

# ============================================================================
# AYARLAR
# ============================================================================
MODEL = "XGBoost"
SET = "ham+fizik+gecmis"
SURUKLENME_S = 5000

TAHMIN = os.path.join(ANA_KLASOR, "data", "model")
RAPOR = os.path.join(ANA_KLASOR, "reports", "model")
with open(os.path.join(RAPOR, "alarm_kurali_secim_ayarli.json"), encoding="utf-8") as f:
    N = json.load(f)["N"][MODEL]
liste = pd.read_csv(os.path.join(ANA_KLASOR, "data", "faulty", "ariza_listesi.csv"), sep=";")
liste = liste[liste["ariza_tipi"] == "surukleme"][["dosya", "ariza_sinyali", "ariza_siddeti", "aralik", "kullanilabilir"]]
print(f"Model: {MODEL}  Set: {SET}  Alarm kuralı N = {N} (çapraz doğrulamada seçilen)\n")
print(f"Üretilen sürüklenme dosyası: {len(liste)}, görünmez sayılıp çıkarılan: {(~liste['kullanilabilir']).sum()} "
      f"(sinyale göre: {liste.loc[~liste['kullanilabilir'], 'ariza_sinyali'].value_counts().to_dict()})\n")


def ardisik_say(a):
    c = np.zeros(len(a), dtype=int)
    for i, v in enumerate(a):
        c[i] = (c[i - 1] + 1 if i else 1) if v else 0
    return c


def dosya_sonuclari(yol, kume):
    t = pd.read_csv(yol, sep=";", low_memory=False)
    t = t[(t["model"] == MODEL) & (t["ozellik_seti"] == SET) & (t["ariza_tipi"] == "surukleme")]
    satirlar = []
    for dosya, d in t.sort_values("pencere_bitis_s").groupby("dosya"):
        tahmin = d["tahmin"].to_numpy()
        c = ardisik_say(tahmin != "saglam")
        i = np.flatnonzero(c >= N)
        satirlar.append({
            "kume": kume, "dosya": dosya,
            "pencere_ariza_orani": float(np.mean(tahmin != "saglam")),        # herhangi bir arıza dedi mi
            "pencere_dogru_tip_orani": float(np.mean(tahmin == "surukleme")),  # tam olarak sürüklenme dedi mi
            "tespit": len(i) > 0,
            "gecikme_s": float(d["ariza_suresi_s"].to_numpy()[i[0]]) if len(i) else np.nan,
            "arizali_sure_s": float(d["ariza_suresi_s"].max()),
        })
    return pd.DataFrame(satirlar)


parcalar = [dosya_sonuclari(os.path.join(TAHMIN, "ml_tahminler_ayarli.csv"), "çapraz doğrulama")]
yol_test = os.path.join(TAHMIN, "ml_tahminler_ayarli_son_test.csv")
if os.path.exists(yol_test):
    parcalar.append(dosya_sonuclari(yol_test, "son test"))
s = pd.concat(parcalar, ignore_index=True).merge(liste, on="dosya", how="left")
s["sinyal"] = s["ariza_sinyali"].str.replace("Battery ", "").str.replace(r" \[.*\]", "", regex=True)
# alarm anındaki sapma (sinyal biriminde) ve tam sapmanın (C) yüzdesi
s["sapma_alarmda"] = s["ariza_siddeti"] * s["aralik"] * s["gecikme_s"] / SURUKLENME_S
s["tam_sapmanin_yuzdesi"] = 100 * s["gecikme_s"] / SURUKLENME_S
# sürüş bitene kadar ulaşılabilen en büyük sapma (yakalanamayanlar için: arıza ne kadar büyüyebildi?)
s["sapma_surus_sonunda"] = s["ariza_siddeti"] * s["aralik"] * s["arizali_sure_s"] / SURUKLENME_S


def ozet(g):
    return pd.Series({
        "dosya": len(g),
        "yakalama_%": 100 * g["tespit"].mean(),
        "gecikme_ortanca_s": g["gecikme_s"].median(),
        "alarmda_sapma_ortanca": g["sapma_alarmda"].median(),
        "alarmda_C_yuzdesi_ortanca": g["tam_sapmanin_yuzdesi"].median(),
        "pencere_ariza_%": 100 * g["pencere_ariza_orani"].mean(),
        "pencere_dogru_tip_%": 100 * g["pencere_dogru_tip_orani"].mean(),
    })


pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("=== Şiddete göre (çapraz doğrulama + son test birlikte) ===")
print(s.groupby("ariza_siddeti").apply(ozet, include_groups=False).drop(columns="alarmda_sapma_ortanca")
      .round(2).to_string(), "\n")                     # sapma birimi sinyale göre farklı → sinyal tablosunda
print("=== Şiddet × sinyal ===")
print(s.groupby(["ariza_siddeti", "sinyal"]).apply(ozet, include_groups=False).round(2).to_string(), "\n")
print("=== Küme × şiddet ===")
print(s.groupby(["kume", "ariza_siddeti"]).apply(ozet, include_groups=False).round(2).to_string(), "\n")

print("=== Yakalanamayan sürüklenme dosyaları: sürüş bitene kadar sapma ne kadar büyüyebildi? ===")
k = s[~s["tespit"]][["kume", "dosya", "sinyal", "ariza_siddeti", "aralik", "arizali_sure_s", "sapma_surus_sonunda"]]
print(k.round(3).to_string(index=False) if len(k) else "Yok", "\n")

print("Okuma: 'alarmda_sapma' sinyalin kendi biriminde (V, A, °C); 'C_yuzdesi' tam sapmanın yüzde kaçına "
      "ulaşıldığında alarm verildiği.\nHafif sürüklenmede alarm anındaki sapma küçükse model küçük sapmaları "
      "yakalıyor demektir; yakalanamayanlarda sürüş sonundaki sapma çok küçükse arıza fiziksel olarak hiç "
      "görünür hale gelmemiştir.")

s.to_csv(os.path.join(RAPOR, "surukleme_analizi.csv"), sep=";", index=False)
print("\nKaydedildi: reports/model/surukleme_analizi.csv")
