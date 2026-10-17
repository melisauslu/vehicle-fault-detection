import pandas as pd

a = pd.read_csv("data/clean_B/TripB03.csv", sep=";")
b = pd.read_csv("data/clean_B/TripB07.csv", sep=";")

print("Satır sayısı  B03:", len(a), "  B07:", len(b))
print("Süre (dk)     B03:", round(a["Time [s]"].iloc[-1] / 60, 1),
      "  B07:", round(b["Time [s]"].iloc[-1] / 60, 1))

for kolon in ["Velocity [km/h]", "Battery Voltage [V]",
              "Battery Current [A]", "Ambient Temperature [°C]"]:
    print(f"{kolon:28s} ort. B03: {a[kolon].mean():8.2f}   "
          f"B07: {b[kolon].mean():8.2f}")

n = min(len(a), len(b))
ayni = (a["Battery Voltage [V]"].values[:n] == b["Battery Voltage [V]"].values[:n]).mean()
print(f"\nİlk {n} satırda batarya gerilimi aynı olan satır oranı: %{ayni*100:.1f}")