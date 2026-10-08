import glob
import pandas as pd

def load_trip(path):
    df = pd.read_csv(path, sep=";", encoding="cp1252")
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    # Aynı isimli kolonları ayır (Temperature Vent right iki kez geçiyor)
    cols = list(df.columns)
    seen = {}
    for i, c in enumerate(cols):
        if cols.count(c) > 1:
            seen[c] = seen.get(c, 0) + 1
            cols[i] = f"{c} #{seen[c]}"
    df.columns = cols
    return df

files = sorted(glob.glob("data/B_winter/*.csv"))

print("=== 1) Örnekleme aralığı 0.1 sn'den farklı olan dosyalar ===")
for f in files:
    d = load_trip(f)
    dt = d["Time [s]"].diff().median()
    if abs(dt - 0.1) > 0.001:
        print(f.split("\\")[-1], "aralık:", dt)
print("(Hiçbir şey yazmadıysa hepsi 0.1 sn)")

print()
print("=== 2) Eksik veri: hangi kolonda, satırların neresinde ===")
for f in files:
    d = load_trip(f)
    name = f.split("\\")[-1]
    na = d.isna()
    rows_na = na.any(axis=1)
    if rows_na.any():
        idx = rows_na[rows_na].index
        per_col = na.sum()
        per_col = per_col[per_col > 0].sort_values(ascending=False)
        print(f"{name}: {len(idx)} satırda eksik (satır {idx.min()} - {idx.max()} arası)")
        print("   en çok eksik kolonlar:", {k: int(v) for k, v in per_col.head(3).items()})

print()
print("=== 3) Temiz dosyalarda hiç değişmeyen (sabit) kolonlar ===")
temiz = ["TripB01", "TripB02", "TripB03", "TripB15",
         "TripB16", "TripB17", "TripB27", "TripB33"]
for t in temiz:
    d = load_trip(f"data/B_winter/{t}.csv")
    sabit = d.columns[d.nunique() <= 1].tolist()
    print(t, "sabit kolonlar:", sabit)