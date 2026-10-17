import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ISTENEN = "Requested Heating Power [W]"
GERCEK = "Heating Power LIN [W]"

parcalar = []
for f in sorted(glob.glob("data/clean_B/*.csv")):
    d = pd.read_csv(f, sep=";", encoding="utf-8", usecols=["Time [s]", ISTENEN, GERCEK])
    d["dosya"] = os.path.basename(f)
    parcalar.append(d)
d = pd.concat(parcalar, ignore_index=True)
fark = d[GERCEK] - d[ISTENEN]

print("=== 1) Her an: gerçekleşen − istenen güç (W) ===")
print(fark.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(1).to_string())

calisirken = d[ISTENEN] > 500                      # ısıtıcıdan anlamlı güç isteniyorken
oran = (fark[calisirken] / d.loc[calisirken, ISTENEN]).abs() * 100
print("\n=== 2) Isıtıcı çalışırken bağıl fark (%) ===")
print(f"Satır sayısı: {calisirken.sum()}")
print(oran.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(2).to_string())

print("\n=== 3) Isıtıcı istenen değere ne kadar geç yetişiyor? ===")
hatalar = {}
for k in range(0, 31):                             # 0 ile 3 saniye arası gecikmeleri dene
    kaymis = d.groupby("dosya")[ISTENEN].shift(k)  # istenen gücü k satır kaydır (dosya içinde)
    hatalar[k] = (d[GERCEK] - kaymis).abs().mean()
en_iyi = min(hatalar, key=hatalar.get)
print(f"En iyi uyum: {en_iyi} satır = {en_iyi * 0.1:.1f} saniye gecikme")
print(f"Gecikmesiz ortalama mutlak fark: {hatalar[0]:.1f} W")
print(f"En iyi gecikmeyle ortalama mutlak fark: {hatalar[en_iyi]:.1f} W")

print("\n=== 4) İstenen güç 0 iken gerçekleşen güç ===")
kapali = d[ISTENEN] == 0
print(f"Satır sayısı: {kapali.sum()}")
if kapali.any():
    print(f"Gerçekleşen gücün en büyüğü: {d.loc[kapali, GERCEK].max():.1f} W")

# grafikler: B17'de iki gücü üst üste ve tüm verideki farkın dağılımı
b17 = d[d["dosya"] == "TripB17.csv"]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(b17["Time [s]"] / 60, b17[ISTENEN], label="İstenen", linewidth=2)
ax[0].plot(b17["Time [s]"] / 60, b17[GERCEK], label="Gerçekleşen", linewidth=1)
ax[0].set_xlabel("dakika")
ax[0].set_ylabel("Isıtıcı gücü [W]")
ax[0].set_title("TripB17")
ax[0].legend()
ax[1].hist(fark.clip(-500, 500), bins=100)
ax[1].set_xlabel("Gerçekleşen − istenen [W]")
ax[1].set_ylabel("Satır sayısı")
ax[1].set_yscale("log")
plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/isitici_istenen_gercek.png", dpi=120)
plt.show()