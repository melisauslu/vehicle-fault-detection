import glob
import pandas as pd

files = sorted(glob.glob("data/clean_B/*.csv"))
d = pd.concat([pd.read_csv(f, sep=";", encoding="utf-8") for f in files], ignore_index=True)

guc = "Heating Power LIN [W]"
debi = "Coolant Volume Flow +500 [l/h]"
giris = "Temperature Coolant Heater Inlet [°C]"
cikis = "Temperature Coolant Heater Outlet [°C]"

# Suyun özellikleri — Çengel & Boles, Termodinamik, sıvı su özellikleri tablosu (tablo numarasını kitaptan kontrol et)
RHO = 1.0       # yoğunluk [kg/l]
CP = 4180       # özgül ısı [J/(kg·°C)]
# NOT: soğutma sıvısı büyük ihtimalle su-glikol karışımı; karışımın özellikleri bilinmediği için su kullanıldı

dT = d[cikis] - d[giris]
secim = (d[guc] > 1000) & (dT > 3)

# Q = ṁ·c_p·ΔT  →  P / ΔT = ṁ·c_p  [W/°C]   (bu sayı debi DEĞİL, debi × ısı kapasitesi)
mcp = d.loc[secim, guc] / dT[secim]
# ṁ = P / (c_p·ΔT) [kg/s]  →  hacimsel debi = ṁ / ρ [l/s]  →  × 3600 = [l/h]
tahmin = mcp / (RHO * CP) * 3600

print("Toplam satır:", len(d))
print("Kullanılan satır sayısı:", secim.sum())
print("ṁ·c_p = P / ΔT, ortanca [W/°C]:         ", round(mcp.median(), 1))
print("Fizikten hesaplanan debi (l/h), ortanca:", round(tahmin.median()))
print("Tablodaki ham debi, ortanca:            ", round(d.loc[secim, debi].median(), 2))
print("Ham + 500, ortanca:                     ", round(d.loc[secim, debi].median() + 500))

print()
print("Hesaplanan debinin dağılımı (l/h):")
print(tahmin.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_string())