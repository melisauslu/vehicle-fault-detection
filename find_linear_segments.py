import glob
import os
import pandas as pd

KOLONLAR = ["Battery Voltage [V]", "Battery Current [A]",
            "Heater Voltage [V]", "Heating Power LIN [W]"]
MIN_SURE_SN = 10

def duz_bolumler(s):
    fark = s.diff()
    cozunurluk = fark.abs()[fark.abs() > 0].min()
    duz = (fark.diff().abs() <= cozunurluk * 1.01) & (fark.abs() > 0)
    grup = (duz != duz.shift()).cumsum()
    for _, g in duz.groupby(grup):
        if g.iloc[0] and len(g) >= MIN_SURE_SN * 10:
            yield g.index[0], g.index[-1]

for f in sorted(glob.glob("data/clean_B/*.csv")):
    d = pd.read_csv(f, sep=";", encoding="utf-8")
    t = d["Time [s]"] / 60
    for k in KOLONLAR:
        for a, b in duz_bolumler(d[k]):
            print(f"{os.path.basename(f)}  {k:25s}  dk {t[a]:.1f} - {t[b]:.1f}")