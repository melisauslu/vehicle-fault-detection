import glob
import os
import numpy as np
import pandas as pd

# Sinyallerin gerçek sensör çözünürlüğünü ve güncellenme hızını anlamak için TANI kodu.
# Hiçbir dosyayı değiştirmez, yalnızca ekrana yazar.

ANA_KLASOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMIZ = os.path.join(ANA_KLASOR, "data", "clean_B")
SINYALLER = ["Battery Temperature [°C]", "Battery Voltage [V]", "Battery Current [A]", "SoC [%]",
             "Heating Power LIN [W]", "Heater Voltage [V]", "Heater Current [A]",
             "Temperature Coolant Heater Outlet [°C]"]

veri = [pd.read_csv(f, sep=";", usecols=SINYALLER) for f in sorted(glob.glob(os.path.join(TEMIZ, "*.csv")))]

for c in SINYALLER:
    x = np.concatenate([d[c].to_numpy() for d in veri])
    fark = np.concatenate([np.round(np.abs(np.diff(d[c].to_numpy())), 6) for d in veri])   # sürüş içinde
    degisen = fark[fark > 0]
    adimlar = pd.Series(degisen).value_counts().head(5)

    # Değerin aynı kaldığı ardışık satır sayısı (plato uzunluğu)
    uzunluklar, platolar = [], []
    for d in veri:
        v = d[c].to_numpy()
        sinir = np.flatnonzero(np.diff(v) != 0) + 1
        bas = np.r_[0, sinir]
        uz = np.diff(np.r_[bas, len(v)])
        uzunluklar.append(uz)
        platolar.append(pd.DataFrame({"deger": v[bas], "uzunluk": uz}))
    uz = np.concatenate(uzunluklar)
    p = pd.concat(platolar)
    # en az 1 s sabit kalan değerler arasındaki adımlar (sürüş içinde)
    pa = np.concatenate([np.round(np.abs(np.diff(q[q["uzunluk"] >= 10]["deger"].to_numpy())), 6) for q in platolar])
    plato_adim = pd.Series(pa[pa > 0]).value_counts().head(3)

    print(f"=== {c} ===")
    print(f"  Satırların yüzde kaçında değer bir öncekinden farklı: %{100 * len(degisen) / len(fark):.1f}")
    print(f"  En sık 5 adım (adım: kaç kez): {({float(k): int(v) for k, v in adimlar.items()})}")
    print(f"  Değerin sabit kaldığı süre: ortanca {np.median(uz) * 0.1:.1f} s, "
          f"tek satırlık değer oranı %{100 * np.mean(uz == 1):.0f}")
    print(f"  En az 1 s sabit kalan değerler arasındaki en sık adımlar: "
          f"{({float(k): int(v) for k, v in plato_adim.items()})}")
    print(f"  Örnek değerler: {np.unique(np.round(x, 4))[:8]}")
    print()

# Bir değişimin yakından görünüşü: batarya sıcaklığı ilk değiştiği yerin çevresi
d = veri[0]
T = d["Battery Temperature [°C]"].to_numpy()
i = int(np.flatnonzero(np.diff(T) != 0)[0])
print("Batarya sıcaklığının ilk değiştiği yerin çevresi (ardışık 0.1 s'lik satırlar):")
print(np.round(T[max(0, i - 5):i + 15], 3))
