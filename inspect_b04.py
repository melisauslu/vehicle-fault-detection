import os
import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv("data/B_winter/TripB04.csv", sep=";", encoding="cp1252")
d.columns = d.columns.str.replace(r"\s+", " ", regex=True).str.strip()
dk = d["Time [s]"] / 60

def kolon(*kelimeler):                      # adında bu kelimeler geçen ilk kolon
    for c in d.columns:
        if all(k.lower() in c.lower() for k in kelimeler):
            return c

hiz = kolon("velocity")
soc = kolon("soc")
gerilim = "Battery Voltage [V]"
akim = "Battery Current [A]"
isitici = "Heating Power LIN [W]"
debi = "Coolant Volume Flow +500 [l/h]"
print("Kullanılan kolonlar:", hiz, "|", soc)

# 1) kayıtta zaman atlaması var mı (normalde her satır 0,1 sn)
adim = d["Time [s]"].diff()
atlama = adim[adim > 0.15]
print()
print("Zaman atlaması sayısı:", len(atlama))
for i, a in atlama.head(10).items():
    print(f"   dk {dk[i]:.1f}: {a:.1f} sn atlama")

# 2) üç bölümün karşılaştırması
bolumler = {"0-15 dk (temiz)": (0, 15), "15-42 dk (şüpheli)": (15, 42), "42 dk-son": (42, 999)}
satirlar = []
for ad, (a, b) in bolumler.items():
    p = d[(dk >= a) & (dk < b)]
    satirlar.append({
        "bölüm": ad,
        "ort. hız": p[hiz].mean(),
        "hareket %": (p[hiz] > 1).mean() * 100,
        "SoC baş": p[soc].iloc[0],
        "SoC son": p[soc].iloc[-1],
        "ort. akım A": p[akim].mean(),
        "gerilim std": p[gerilim].std(),
        "ort. ısıtıcı W": p[isitici].mean(),
        "debi ortanca": p[debi].median(),
        "eksik %": p.isna().any(axis=1).mean() * 100,
    })
print()
pd.set_option("display.width", 200)
print(pd.DataFrame(satirlar).round(2).to_string(index=False))

# 3) grafik
fig, ax = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
for i, (k, etiket) in enumerate([(hiz, "Hız km/h"), (soc, "SoC"), (gerilim, "Batarya V"),
                                 (akim, "Batarya A"), (isitici, "Isıtıcı W"), (debi, "Debi (ham)")]):
    ax[i].plot(dk, d[k])
    ax[i].set_ylabel(etiket)
    ax[i].axvspan(15, 42, color="red", alpha=0.1)   # şüpheli bölüm kırmızı
ax[-1].set_xlabel("dakika")
fig.suptitle("TripB04 (ham) - kırmızı: şüpheli bölüm")
plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/TripB04_inceleme.png", dpi=120)
plt.show()