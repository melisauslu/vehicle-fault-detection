import glob
import os
import pandas as pd

files = sorted(glob.glob("data/clean_B/*.csv"))
HIZ = "Velocity [km/h]"

# Adım testi (WMO step test):0,1 sn'de izin verilen en büyük değişim
ADIM_SINIRI = {
    "Velocity [km/h]": 5.0,                         # en sert fren ~3,3 km/h / 0,1 sn
    "SoC [%]": 1.0,                                 # 400 A'de bile ~%0,02 / 0,1 sn
    "Battery Temperature [°C]": 3.0,
    "Temperature Coolant Heater Inlet [°C]": 3.0,
    "Temperature Coolant Heater Outlet [°C]": 3.0,
    "Cabin Temperature Sensor [°C]": 3.0,
}
# sınır koymadan sadece dağılımına bakılacak sinyaller
SADECE_DAGILIM = ["Battery Voltage [V]", "Battery Current [A]", "Heating Power LIN [W]"]

# Süreklilik testi (WMO persistence test): araç giderken bu süre boyunca hiç değişmemeli
SURE_SN = {"Battery Voltage [V]": 60, "Battery Current [A]": 60}

adimlar = {k: [] for k in list(ADIM_SINIRI) + SADECE_DAGILIM}
siçramalar, takilmalar = [], []

for f in files:
    d = pd.read_csv(f, sep=";", encoding="utf-8")
    ad = os.path.basename(f)
    dk = d["Time [s]"] / 60

    # adım testi
    for k in adimlar:
        fark = d[k].diff().abs()
        adimlar[k].append(fark.dropna())
        if k in ADIM_SINIRI:
            asan = fark > ADIM_SINIRI[k]
            if asan.any():
                siçramalar.append(f"{ad}  {k:42s} {asan.sum():4d} kez  en büyük {fark.max():.2f}  "
                                  f"dk {dk[asan].min():.1f}-{dk[asan].max():.1f}")

    # süreklilik testi
    hareket = d[HIZ] > 5
    for k, sn in SURE_SN.items():
        ayni = (d[k].diff() == 0) & hareket
        grup = (ayni != ayni.shift()).cumsum()
        for _, g in ayni.groupby(grup):
            if g.iloc[0] and len(g) >= sn * 10:
                takilmalar.append(f"{ad}  {k:25s} dk {dk[g.index[0]]:.1f}-{dk[g.index[-1]]:.1f}")

print("=== 1) Adım testi: sınırı aşan ani sıçramalar ===")
print("\n".join(siçramalar) if siçramalar else "Yok")

print("\n=== 2) Süreklilik testi: araç giderken takılı kalan sensör ===")
print("\n".join(takilmalar) if takilmalar else "Yok")

print("\n=== 3) 0,1 sn'lik değişimlerin dağılımı (sınırların kontrolü için) ===")
ozet = {}
for k, v in adimlar.items():
    s = pd.concat(v)
    ozet[k] = [s.median(), s.quantile(0.99), s.quantile(0.9999), s.max()]
ozet = pd.DataFrame(ozet, index=["ortanca", "%99", "%99,99", "en büyük"]).T.round(3)
pd.set_option("display.width", 200)
print(ozet.to_string())