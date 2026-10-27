import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import ANA_KLASOR

# Zor arızaları İYİLEŞTİRMEDEN ÖNCE TEŞHİS: model nerede hata yapıyor?
# Yalnızca çapraz doğrulama tahminlerine bakar (test sürüşlerine dokunmaz).
MODEL = "RandomForest"
SET = "ham+fizik+gecmis"    # ana sonuç seti (uzun geçmişten önceyle karşılaştırmak için: "ham+fizik")
ZOR = ["surukleme", "sabit_kayma", "takilma", "kazanc"]

t = pd.read_csv(os.path.join(ANA_KLASOR, "data", "model", "ml_tahminler.csv"), sep=";", low_memory=False)
t = t[(t["model"] == MODEL) & (t["ozellik_seti"] == SET)].copy()
liste = pd.read_csv(os.path.join(ANA_KLASOR, "data", "faulty", "ariza_listesi.csv"), sep=";")
t = t.merge(liste[["dosya", "ariza_sinyali"]], on="dosya", how="left")
t["dogru_tip"] = t["tahmin"] == t["ariza_tipi"]          # arıza tipini doğru bildi mi
t["alarm"] = t["tahmin"] != "saglam"                      # en azından "arıza var" dedi mi
pd.set_option("display.width", 250)

z = t[t["ariza_tipi"].isin(ZOR)]

print(f"=== {MODEL}, {SET} — zor arızalar, çapraz doğrulama tahminleri ===\n")
print("1) Şiddete (L) göre: alarm oranı / doğru tip oranı")
print("   (L = 0.05'te düşükse sorun modelde değil, arızanın fiziksel olarak çok küçük olmasında)")
print(z.groupby(["ariza_tipi", "ariza_siddeti"])[["alarm", "dogru_tip"]].mean().round(2).unstack().to_string(), "\n")

print("2) Sensöre göre: doğru tip oranı")
print(z.groupby(["ariza_tipi", "ariza_sinyali"])["dogru_tip"].mean().round(2).unstack().to_string(), "\n")

print("3) Arıza başladıktan sonra geçen süreye göre: doğru tip oranı")
print("   (süre arttıkça yükseliyorsa model arızayı ancak büyüyünce görebiliyor — sürüklenmede beklenen)")
z = z.assign(sure=pd.cut(z["ariza_suresi_s"], [0, 60, 300, 900, np.inf],
                         labels=["0-1 dk", "1-5 dk", "5-15 dk", ">15 dk"], include_lowest=True))
print(z.groupby(["ariza_tipi", "sure"], observed=True)["dogru_tip"].mean().round(2).unstack().to_string(), "\n")

print("4) Yanlış tahminler en çok neye gidiyor? (satır = gerçek, oran)")
y = z[~z["dogru_tip"]]
print(pd.crosstab(y["ariza_tipi"], y["tahmin"], normalize="index").round(2).to_string(), "\n")

print("5) Sağlam pencereler yanlışlıkla hangi arıza sanılıyor? (yanlış alarmların kaynağı)")
s = t[(t["ariza_tipi"] == "saglam") & t["alarm"]]
print((s["tahmin"].value_counts(normalize=True).round(2)).to_string())
