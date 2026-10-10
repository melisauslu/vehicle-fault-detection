import os
import pandas as pd
import matplotlib.pyplot as plt

debi = "Coolant Volume Flow +500 [l/h]"
cols = [
    ("Battery Voltage [V]", "Batarya V"),
    ("Heater Voltage [V]", "Isıtıcı V"),
    ("Heating Power LIN [W]", "Isıtıcı gücü W"),
    (debi, "Debi (ham)"),
    ("Temperature Coolant Heater Outlet [°C]", "Isıtıcı çıkış °C"),
    ("Cabin Temperature Sensor [°C]", "Kabin °C"),
]

def load(name):
    return pd.read_csv(f"data/clean_B/{name}.csv", sep=";")

def plot_trip(name, title):
    d = load(name)
    t = d["Time [s]"] / 60
    fig, ax = plt.subplots(len(cols), 1, figsize=(11, 12), sharex=True)
    for a, (c, label) in zip(ax, cols):
        a.plot(t, d[c])
        a.set_ylabel(label)
    ax[-1].set_xlabel("dakika")
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig(f"reports/{name}_olay.png", dpi=110)
    plt.show()

plot_trip("TripB04", "TripB04")
plot_trip("TripB09", "TripB09")
plot_trip("TripB02", "TripB02")

# B02: ısıtıcı gücü 10 kW'ın üstündeyken V*I / güç oranı
d = load("TripB02")
m = d["Heating Power LIN [W]"] > 10000
oran = d.loc[m, "Heater Voltage [V]"] * d.loc[m, "Heater Current [A]"] / d.loc[m, "Heating Power LIN [W]"]
print("B02, güç>10kW satırlarında V*I/P:")
print(oran.describe().round(3).to_string())