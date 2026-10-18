import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

V = "Battery Voltage [V]"
I = "Battery Current [A]"
T = "Battery Temperature [°C]"
SOC = "SoC [%]"
ADIM = 10        # 10 satır = 1 saniye arayla bak
MIN_DI = 30      # akım 1 sn'de en az 30 A değişmiş olsun

def egim(dI, dV):
    # ΔV = R · ΔI doğrusuna en iyi uyan R (en küçük kareler)
    return np.sum(dV * dI) / np.sum(dI * dI)

satirlar, tum_dI, tum_dV = [], [], []
for f in sorted(glob.glob("data/clean_B/*.csv")):
    d = pd.read_csv(f, sep=";", encoding="utf-8")
    dV = d[V].diff(ADIM)                 # 1 sn'deki gerilim değişimi
    dI = d[I].diff(ADIM)                 # 1 sn'deki akım değişimi
    m = dI.abs() > MIN_DI
    dV, dI = dV[m], dI[m]
    if len(dI) < 50:
        continue
    satirlar.append({
        "dosya": os.path.basename(f),
        "nokta": len(dI),
        "R_ohm": egim(dI, dV),
        "bat_sıcaklık": d[T].median(),
        "SoC": d[SOC].median(),
    })
    tum_dI.append(dI)
    tum_dV.append(dV)

r = pd.DataFrame(satirlar).sort_values("bat_sıcaklık")
dI, dV = pd.concat(tum_dI), pd.concat(tum_dV)
R = egim(dI, dV)
R2 = 1 - np.sum((dV - R * dI) ** 2) / np.sum((dV - dV.mean()) ** 2)

print("=== 1) Gerilim akımla doğrusal mı değişiyor? ===")
print(f"Kullanılan nokta sayısı: {len(dI)}")
print(f"Tüm veriden iç direnç R: {R:.4f} ohm")
print(f"Uyum (R²): {R2:.3f}   (1'e yakınsa ilişki güçlü)")

print("\n=== 2) Sürüş başına iç direnç (sıcaklığa göre sıralı) ===")
pd.set_option("display.width", 200)
print(r.round({"R_ohm": 4, "bat_sıcaklık": 1, "SoC": 1}).to_string(index=False))

print("\n=== 3) Soğukta iç direnç değişiyor mu? ===")
print(f"Sıcaklık ile direnç arasındaki korelasyon: {r['bat_sıcaklık'].corr(r['R_ohm']):.2f}")
print("(eksi ise: sıcaklık düştükçe direnç artıyor)")

# grafikler
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ornek = np.random.default_rng(0).choice(len(dI), size=min(20000, len(dI)), replace=False)
ax[0].scatter(dI.iloc[ornek], dV.iloc[ornek], s=2, alpha=0.3)
x = np.linspace(dI.min(), dI.max(), 2)
ax[0].plot(x, R * x, color="red", label=f"R = {R:.3f} ohm")
ax[0].set_xlabel("1 sn'de akım değişimi ΔI [A]")
ax[0].set_ylabel("1 sn'de gerilim değişimi ΔV [V]")
ax[0].legend()
ax[1].scatter(r["bat_sıcaklık"], r["R_ohm"])
ax[1].set_xlabel("Sürüşün ortanca batarya sıcaklığı [°C]")
ax[1].set_ylabel("İç direnç R [ohm]")
plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/batarya_direnc.png", dpi=120)
plt.show()

from scipy import stats

print("\n=== 4) Korelasyonun güvenilirliği ===")
x, y = r["bat_sıcaklık"], r["R_ohm"]
n = len(r)

pr, pp = stats.pearsonr(x, y)
sr, sp = stats.spearmanr(x, y)

# %95 güven aralığı (Fisher z dönüşümü)
z = np.arctanh(pr)
se = 1 / np.sqrt(n - 3)
alt, ust = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

print(f"Sürüş sayısı      : {n}")
print(f"Pearson r         : {pr:.3f}   p = {pp:.1e}")
print(f"%95 güven aralığı : [{alt:.3f}, {ust:.3f}]")
print(f"Spearman ρ        : {sr:.3f}   p = {sp:.1e}")