import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

files = sorted(glob.glob("data/clean_B/*.csv"))
print(len(files), "temiz dosya")


def load(path):
    return pd.read_csv(path, sep=";", encoding="utf-8")


all_df = pd.concat([load(f) for f in files], ignore_index=True)
ozet = all_df.describe().T[["min", "mean", "max", "std"]].round(2)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)
print(ozet.to_string())


hv, hi = "Heater Voltage [V]", "Heater Current [A]"
hp = "Heating Power LIN [W]"
aktif = all_df[all_df[hp] > 500]
oran = (aktif[hv] * aktif[hi]) / aktif[hp]
print()
print("Isıtıcı V*I / LIN gücü oranı (1.0'a yakın olmalı):")
print(oran.describe().round(3).to_string())


debi = "Coolant Volume Flow +500 [l/h]"
print()
print("Debi kolonu:")
print(all_df[debi].describe().round(2).to_string())
print("Farklı değer sayısı:", all_df[debi].nunique())


d = load("data/clean_B/TripB17.csv")
t = d["Time [s]"] / 60
fig, ax = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
ax[0].plot(t, d["Velocity [km/h]"]); ax[0].set_ylabel("Hız km/h")
ax[1].plot(t, d["Battery Voltage [V]"]); ax[1].set_ylabel("Batarya V")
ax[2].plot(t, d["Battery Current [A]"]); ax[2].set_ylabel("Batarya A")
ax[3].plot(t, d[hp]); ax[3].set_ylabel("Isıtıcı gücü W")
ax[4].plot(t, d[debi]); ax[4].set_ylabel("Debi (ham)")
ax[4].set_xlabel("dakika")
fig.suptitle("TripB17")
plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/TripB17.png", dpi=120)
plt.show()