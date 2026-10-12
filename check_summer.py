import glob
import os
import pandas as pd

def oku(path):
    df = pd.read_csv(path, sep=";", encoding="cp1252")
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    for bozuk in ["ï¿½", "\ufffd", "Â°"]:
        df.columns = df.columns.str.replace(bozuk, "°", regex=False)
    return df


SINYALLER = {
    "hız": "Velocity",
    "bat.V": "Battery Voltage",
    "bat.A": "Battery Current",
    "SoC": "SoC [%]",
    "bat.°C": "Battery Temperature",
    "ısıt.W": "Heating Power LIN",
    "ısıt.V": "Heater Voltage",
    "ısıt.A": "Heater Current",
    "debi": "Coolant Volume Flow",
    "giriş°C": "Temperature Coolant Heater Inlet",
    "çıkış°C": "Temperature Coolant Heater Outlet",
    "kabin°C": "Cabin Temperature Sensor",
    "klima": "AirCon Power",
}

satirlar = []
for f in sorted(glob.glob("data/A_summer/*.csv")):
    d = oku(f)
    satir = {"dosya": os.path.basename(f), "dakika": round(len(d) * 0.1 / 60, 1)}
    for kisa, ifade in SINYALLER.items():
        kolon = next((c for c in d.columns if ifade in c and not c.startswith(("min.", "max.", "displayed"))), None)
        if kolon is None:
            satir[kisa] = "YOK"
        else:
            satir[kisa] = f"{d[kolon].notna().mean() * 100:.0f}"
    satirlar.append(satir)

r = pd.DataFrame(satirlar)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(r.to_string(index=False))
print()
print("Yaz dosyası sayısı:", len(r))
print("Toplam süre (saat):", round(r["dakika"].sum() / 60, 1))