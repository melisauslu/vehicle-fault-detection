import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import ANA_KLASOR, veriyi_oku, bolmeler, kat_sabitleri, fizik_ekle

# YANLIŞ ALARMLARIN KAYNAĞI: sağlam pencereler neden arıza sanılıyor?
# Sınanan varsayım: sağlam batarya sıcaklığı dakikalarca değişmediği için (1 °C çözünürlük) model bu pencereleri
#   "sıcaklık sensörü takılması" sanıyor. Doğruysa: "takılma" sanılan sağlam pencerelerde sıcaklığın değişmeden
#   geçen süresi, doğru bilinen sağlam pencerelerden belirgin şekilde uzun olmalı.
# Yalnızca çapraz doğrulama tahminlerine bakar (test sürüşlerine dokunmaz). Hiçbir dosyaya yazmaz.
MODEL = "XGBoost"
SET = "ham+fizik+gecmis"

T, V, I = "Battery Temperature [°C]", "Battery Voltage [V]", "Battery Current [A]"
INCELE = {                                             # kısa ad → özellik
    "T_degismeden_s": f"{T}__degismeden_s",
    "V_degismeden_s": f"{V}__degismeden_s",
    "I_degismeden_s": f"{I}__degismeden_s",
    "soc_farki_5dk": "soc_farki_5dk",
    "gerilim_egim_5dk": "gerilim_farki__egim_5dk",
    "T_ort": f"{T}__ort",
    "V_fark_ort": "gerilim_farki__ort",            # V_ısıtıcı − V_bat (sağlamda ~2.5 V)
}

# Özellikler, her katta o katın sabitleriyle (ml_modeller.py ile aynı) → yalnızca doğrulama pencereleri
oz, bilgi = veriyi_oku()
cv, _ = bolmeler(oz)
parcalar = []
for k, (egitim, dogrulama) in enumerate(cv):
    o = fizik_ekle(oz, kat_sabitleri(oz, egitim, bilgi))
    parcalar.append(o.loc[dogrulama, ["dosya", "surus", "kat", "pencere_bitis_s", "ariza_tipi"] + list(INCELE.values())])
oz_va = pd.concat(parcalar).rename(columns={v: k for k, v in INCELE.items()})

t = pd.read_csv(os.path.join(ANA_KLASOR, "data", "model", "ml_tahminler.csv"), sep=";", low_memory=False)
t = t[(t["model"] == MODEL) & (t["ozellik_seti"] == SET) & (t["ariza_tipi"] == "saglam")]
s = t[["dosya", "pencere_bitis_s", "tahmin"]].merge(oz_va, on=["dosya", "pencere_bitis_s"], how="left")
assert s["T_degismeden_s"].notna().all(), "tahmin dosyası ile özellik dosyası eşleşmedi — veri_hazirla.py → ml_modeller.py sırayla çalıştı mı?"
s["yanlis_alarm"] = s["tahmin"] != "saglam"
adim_s = bilgi["adim_satir"] * 0.1
sinir = bilgi["gecmis_satir"] * 0.1 - 1                # "5 dk boyunca hiç değişmedi" (299 s ve üstü)
pd.set_option("display.width", 250)

print(f"=== {MODEL}, {SET} — sağlam doğrulama pencereleri: {len(s)}, yanlış alarm: {s['yanlis_alarm'].sum()} "
      f"(%{100 * s['yanlis_alarm'].mean():.1f}) ===\n")

print("1) Sağlam pencereler, modelin tahminine göre: özelliklerin ortancası")
print("   (satır 'saglam' = doğru bilinenler; diğer satırlar = o arıza sanılan sağlam pencereler)")
g = s.groupby("tahmin")
tablo = g[list(INCELE)].median()
tablo.insert(0, "pencere", g.size())
tablo["T_5dk_sabit_%"] = 100 * g["T_degismeden_s"].apply(lambda v: (v >= sinir).mean())
print(tablo.sort_values("pencere", ascending=False).round(4).to_string(), "\n")

print("2) VARSAYIM TESTİ: sıcaklık 5 dk boyunca hiç değişmemiş sağlam pencerelerde yanlış alarm oranı")
s["sicaklik_5dk_sabit"] = np.where(s["T_degismeden_s"] >= sinir, "evet (≥5 dk sabit)", "hayır")
v = s.groupby("sicaklik_5dk_sabit").agg(pencere=("tahmin", "size"),
                                        yanlis_alarm_orani=("yanlis_alarm", "mean"),
                                        takilma_sanilma_orani=("tahmin", lambda x: (x == "takilma").mean()))
print(v.round(3).to_string())
print("   → 'evet' satırında takılma oranı 'hayır'dan belirgin şekilde yüksekse varsayım doğrulanır\n")

print("3) Yanlış alarmlar birkaç sürüşte mi toplanıyor? (sürüş başına saatteki yanlış alarm, en yüksek 10)")
srs = s.groupby(["surus", "kat"]).agg(saglam_saat=("tahmin", lambda x: len(x) * adim_s / 3600),
                                      yanlis_alarm=("yanlis_alarm", "sum"),
                                      takilma=("tahmin", lambda x: (x == "takilma").sum()),
                                      sabit_kayma=("tahmin", lambda x: (x == "sabit_kayma").sum()),
                                      T_ort=("T_ort", "median"),
                                      V_fark_ort=("V_fark_ort", "median"),
                                      T_5dk_sabit_orani=("sicaklik_5dk_sabit", lambda x: (x != "hayır").mean()))
srs["alarm_saatte"] = srs["yanlis_alarm"] / srs["saglam_saat"]
print(srs.sort_values("alarm_saatte", ascending=False).head(10).round(2).to_string())
print(f"   (V_fark_ort karşılaştırması: bütün sağlam pencerelerde ortanca {s['V_fark_ort'].median():.2f} V, "
      f"%5–%95 aralığı {s['V_fark_ort'].quantile(0.05):.2f}–{s['V_fark_ort'].quantile(0.95):.2f} V)")

print("\n4) İKİNCİ VARSAYIM: 'sabit kayma' yanlış alarmları, gerilim farkı alışılmadık sürüşlerden mi geliyor?")
print("   Sağlam pencereler, V_fark_ort'a göre 5 eşit gruba bölündü (1 = en küçük fark, 5 = en büyük)")
s["V_fark_grubu"] = pd.qcut(s["V_fark_ort"], 5, labels=[1, 2, 3, 4, 5])
w = s.groupby("V_fark_grubu", observed=True).agg(
    V_fark_aralik=("V_fark_ort", lambda x: f"{x.min():.2f}–{x.max():.2f}"),
    pencere=("tahmin", "size"),
    yanlis_alarm_orani=("yanlis_alarm", "mean"),
    sabit_kayma_sanilma_orani=("tahmin", lambda x: (x == "sabit_kayma").mean()))
print(w.round(3).to_string())
print("   → uç gruplarda (1 ya da 5) sabit kayma oranı ortadakilerden belirgin şekilde yüksekse varsayım doğrulanır")

print("\n   Katlara göre saatteki yanlış alarm:")
print(srs.groupby(level="kat")[["yanlis_alarm", "saglam_saat"]].sum()
      .assign(alarm_saatte=lambda d: d["yanlis_alarm"] / d["saglam_saat"]).round(1).to_string())
