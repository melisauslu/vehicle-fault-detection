import os
import pandas as pd
import matplotlib.pyplot as plt


d = pd.read_csv("data/B_winter/TripB02.csv", sep=";", encoding="cp1252")
d.columns = d.columns.str.replace(r"\s+", " ", regex=True).str.strip()


d = d[d["Time [s]"] >= 30].reset_index(drop=True)

def kolon(bas):
    # Derece işareti bozuk olsa bile kolonu adının başından bulur
    return [c for c in d.columns if c.startswith(bas)][0]

cols = [
    (kolon("Battery Voltage"), "Batarya V"),
    (kolon("Heater Voltage"), "Isıtıcı V"),
    (kolon("Heating Power LIN"), "Isıtıcı gücü W"),
    (kolon("Coolant Volume Flow"), "Debi (ham)"),
    (kolon("Temperature Coolant Heater Outlet"), "Isıtıcı çıkış °C"),
    (kolon("Cabin Temperature Sensor"), "Kabin °C"),
]

t = d["Time [s]"] / 60
fig, ax = plt.subplots(len(cols), 1, figsize=(11, 12), sharex=True)
for a, (c, etiket) in zip(ax, cols):
    a.plot(t, d[c])
    a.set_ylabel(etiket)
ax[-1].set_xlabel("dakika")
fig.suptitle("TripB02 (temizlik öncesi)")
plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/TripB02_once.png", dpi=110)
plt.show()