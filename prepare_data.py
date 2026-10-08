import glob
import os
import pandas as pd

def load_trip(path):
    df = pd.read_csv(path, sep=";", encoding="cp1252")
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    cols = list(df.columns)
    seen = {}
    for i, c in enumerate(cols):
        if cols.count(c) > 1:
            seen[c] = seen.get(c, 0) + 1
            cols[i] = f"{c} #{seen[c]}"
    df.columns = cols
    return df

def clean_trip(d):
    ok = ~d.isna().any(axis=1)
    if not ok.any():
        return None
    first = ok.idxmax()
    last = ok[::-1].idxmax()
    d = d.loc[first:last].copy()
    d = d.interpolate(limit=50)
    return d.reset_index(drop=True)

MIN_ROWS = 3000                           

os.makedirs("data/clean_B", exist_ok=True)
rapor = []
for f in sorted(glob.glob("data/B_winter/*.csv")):
    ad = os.path.basename(f)
    d = load_trip(f)
    c = clean_trip(d)
    if c is None:
        durum = "ELENDİ (kullanılabilir satır yok)"
        yeni = 0
    elif len(c) < MIN_ROWS:
        durum = "ELENDİ (5 dk'dan kısa)"
        yeni = len(c)
    elif c.isna().any().any():
        durum = "ELENDİ (doldurulamayan eksik kaldı)"
        yeni = len(c)
    else:
        c.to_csv(f"data/clean_B/{ad}", sep=";", index=False)
        durum = "tamam"
        yeni = len(c)
    rapor.append({"dosya": ad, "ilk_satır": len(d), "temiz_satır": yeni, "durum": durum})

r = pd.DataFrame(rapor)
pd.set_option("display.width", 200)
print(r.to_string(index=False))
print()
kalan = r[r["durum"] == "tamam"]
print("Kullanılacak dosya:", len(kalan))
print("Toplam süre (saat):", round(kalan["temiz_satır"].sum() * 0.1 / 3600, 1))