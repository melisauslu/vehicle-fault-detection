import pandas as pd
import glob

def load_trip(path):
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df.columns = (df.columns
                  .str.replace("\ufffd", "°", regex=False)
                  .str.replace(r"\s+", " ", regex=True)
                  .str.strip())
    return df

files = sorted(glob.glob("data/B_winter/*.csv"))
print(len(files), "dosya bulundu")

df = load_trip(files[0])
print(files[0])
print(df.shape)
print(df.columns.tolist())

for f in files:
    try:
        d = load_trip(f)
        print(f.split("\\")[-1], d.shape, "eksik:", int(d.isna().sum().sum()))
    except Exception as e:
        print(f, "HATA:", e)

import pandas as pd

rows = []
for f in files:
    d = load_trip(f)
    name = f.split("\\")[-1]
    na = d.isna().sum()
    full_missing = na[na == len(d)].index.tolist()       # tamamen boş kolonlar
    part_missing = na[(na > 0) & (na < len(d))]          # kısmen boş kolonlar
    rows.append({
        "dosya": name,
        "satır": len(d),
        "süre_dk": round(d["Time [s]"].max() / 60, 1),
        "tam_boş_kolon_sayısı": len(full_missing),
        "kısmen_boş_kolon_sayısı": len(part_missing),
        "en_çok_eksik_oran_%": round(na.max() / len(d) * 100, 1),
        "boş_kolonlar": "; ".join(full_missing[:4]),
    })

summary = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
print(summary.to_string(index=False))

# Örnekleme aralığı kontrolü (ilk dosya)
d0 = load_trip(files[0])
print("Örnekleme aralığı (sn):", d0["Time [s]"].diff().median())