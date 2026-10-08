import glob
import os
import pandas as pd

debi = "Coolant Volume Flow +500 [l/h]"

def runs(mask):
    """Arka arkaya True olan bölümleri (başlangıç, bitiş) satır numaralarıyla döndürür."""
    out = []
    start = None
    for i, v in enumerate(mask.values):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out

for f in sorted(glob.glob("data/clean_B/*.csv")):
    d = pd.read_csv(f, sep=";")
    ad = os.path.basename(f)
    t = d["Time [s]"] / 60

    uzun = []
    for ad_kolon, maske in [
        ("ısıtıcı V > batarya V + 20", d["Heater Voltage [V]"] > d["Battery Voltage [V]"] + 20),
        ("debi < -100", d[debi] < -100),
    ]:
        for s, e in runs(maske):
            if e - s + 1 >= 100:            # en az 10 saniye süren bölümler
                uzun.append(f"{ad_kolon}: dk {t[s]:.1f} - {t[e]:.1f} ({e - s + 1} satır)")
    if uzun:
        print(ad, f"(toplam {t.iloc[-1]:.1f} dk)")
        for u in uzun:
            print("   ", u)