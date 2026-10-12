import glob
import os
import pandas as pd

for f in sorted(glob.glob("data/clean_B/*.csv")):
    d = pd.read_csv(f, sep=";", encoding="utf-8")
    m = d["Coolant Volume Flow +500 [l/h]"] < -100
    if m.any():
        t = d.loc[m, "Time [s]"] / 60
        print(f"{os.path.basename(f)}  {m.sum()} satır  dk {t.min():.1f} - {t.max():.1f}")