import glob
import pandas as pd

files = sorted(glob.glob("data/clean_B/*.csv"))
d = pd.concat([pd.read_csv(f, sep=";", encoding="utf-8") for f in files], ignore_index=True)

guc = "Heating Power LIN [W]"
debi = "Coolant Volume Flow +500 [l/h]"
giris = "Temperature Coolant Heater Inlet [°C]"
cikis = "Temperature Coolant Heater Outlet [°C]"

dT = d[cikis] - d[giris]
secim = (d[guc] > 1000) & (dT > 3)
tahmin = d.loc[secim, guc] / dT[secim]

print("Toplam satır:", len(d))
print("Kullanılan satır sayısı:", secim.sum())
print("Fizikten hesaplanan debi (l/h), ortanca:", round(tahmin.median()))
print("Tablodaki ham debi, ortanca:            ", round(d.loc[secim, debi].median(), 2))
print("Ham + 500, ortanca:                     ", round(d.loc[secim, debi].median() + 500))

print()
print("Hesaplanan debinin dağılımı (l/h):")
print(tahmin.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_string())