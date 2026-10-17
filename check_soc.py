import glob
import os
import numpy as np
import pandas as pd

ADIM = 10  # 10 satır = 1 saniye
SOC = "SoC [%]"

dosyalar = sorted(glob.glob("data/clean_B/*.csv"))

tum_dsoc = []
for f in dosyalar:
    d = pd.read_csv(f, sep=";")
    dsoc = (d[SOC].shift(-ADIM) - d[SOC]).dropna().abs()
    tum_dsoc.append(dsoc)
    print(f"{os.path.basename(f)}: başta %{d[SOC].iloc[0]:.1f}, "
          f"sonda %{d[SOC].iloc[-1]:.1f}, "
          f"1 sn'de en fazla değişim %{dsoc.max():.3f}")

hepsi = pd.concat(tum_dsoc)
print("\n--- 1 saniyede SoC değişimi (tüm sürüşler) ---")
print(f"Ortanca : %{hepsi.median():.4f}")
print(f"%99     : %{np.percentile(hepsi, 99):.4f}")
print(f"En fazla: %{hepsi.max():.4f}")