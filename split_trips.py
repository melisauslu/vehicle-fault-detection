import glob
import os
import numpy as np
import pandas as pd

files = sorted(glob.glob("data/clean_B/*.csv"))
adlar = [os.path.basename(f) for f in files]

rng = np.random.default_rng(seed=42)
test = set(rng.permutation(adlar)[:7])

satirlar = []
for f, ad in zip(files, adlar):
    n = len(pd.read_csv(f, sep=";", usecols=["Time [s]"]))
    satirlar.append({"dosya": ad, "grup": "test" if ad in test else "egitim",
                     "dakika": round(n * 0.1 / 60, 1)})

s = pd.DataFrame(satirlar)
s.to_csv("split.csv", sep=";", index=False)

print("Grup özeti:")
ozet = s.groupby("grup")["dakika"].agg(sürüş="count", dakika="sum")
ozet["saat"] = (ozet["dakika"] / 60).round(1)
print(ozet.to_string())
print("\nTest sürüşleri:")
print(s[s["grup"] == "test"].to_string(index=False))