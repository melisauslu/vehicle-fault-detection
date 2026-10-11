import glob
import os
import shutil
import pandas as pd

BASLANGIC_KES_SN = 30
BAS_KES_DK = {
    "TripB02.csv": 3.5,
}
SON_KES_DK = {
    "TripB04.csv": 15.0,
    "TripB09.csv": 45.0,
    "TripB35.csv": 22.5,
}
MIN_ROWS = 3000

def load_trip(path):
    df = pd.read_csv(path, sep=";", encoding="cp1252")
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    df.columns = df.columns.str.replace("Velocity [km/h]]]", "Velocity [km/h]", regex=False)
    for bozuk in ["ï¿½", "\ufffd", "Â°"]:
        df.columns = df.columns.str.replace(bozuk, "°", regex=False)
    cols = list(df.columns)
    seen = {}
    for i, c in enumerate(cols):
        if cols.count(c) > 1:
            seen[c] = seen.get(c, 0) + 1
            cols[i] = f"{c} #{seen[c]}"
    df.columns = cols
    return df

def clean_trip(d, ad):
    ok = ~d.isna().any(axis=1)
    if not ok.any():
        return None
    first = ok.idxmax()
    last = ok[::-1].idxmax()
    d = d.loc[first:last].copy()
    d = d.interpolate(limit=50)

    bas = max(BASLANGIC_KES_SN, BAS_KES_DK.get(ad, 0) * 60)
    d = d[d["Time [s]"] >= bas]
    if ad in SON_KES_DK:
        d = d[d["Time [s]"] <= SON_KES_DK[ad] * 60]
    d = d.reset_index(drop=True)
    if len(d) > 0:
        d["Time [s]"] = (d["Time [s]"] - d["Time [s]"].iloc[0]).round(1)
    return d

shutil.rmtree("data/clean_B", ignore_errors=True)
os.makedirs("data/clean_B")

rapor = []
for f in sorted(glob.glob("data/B_winter/*.csv")):
    ad = os.path.basename(f)
    d = load_trip(f)
    c = clean_trip(d, ad)
    if c is None or len(c) == 0:
        durum, yeni = "ELENDİ (kullanılabilir satır yok)", 0
    elif len(c) < MIN_ROWS:
        durum, yeni = "ELENDİ (5 dk'dan kısa)", len(c)
    elif c.isna().any().any():
        durum, yeni = "ELENDİ (doldurulamayan eksik kaldı)", len(c)
    else:
        c.to_csv(f"data/clean_B/{ad}", sep=";", index=False)
        durum, yeni = "tamam", len(c)
    rapor.append({"dosya": ad, "ilk_satır": len(d), "temiz_satır": yeni, "durum": durum})

r = pd.DataFrame(rapor)
pd.set_option("display.width", 200)
print(r.to_string(index=False))
print()
kalan = r[r["durum"] == "tamam"]
print("Kullanılacak dosya:", len(kalan))
print("Toplam süre (saat):", round(kalan["temiz_satır"].sum() * 0.1 / 3600, 1))