import glob
import os
import pandas as pd

files = sorted(glob.glob("data/clean_B/*.csv"))
pd.set_option("display.width", 220)

print("=== 1) TripB01'den farklı kolon adı olan dosyalar ===")
ref = list(pd.read_csv(files[0], sep=";", nrows=1).columns)
for f in files[1:]:
    cols = list(pd.read_csv(f, sep=";", nrows=1).columns)
    fazla = [c for c in cols if c not in ref]
    eksik = [c for c in ref if c not in cols]
    if fazla or eksik:
        print(os.path.basename(f), "| fazla:", fazla, "| eksik:", eksik)
print("(Hiçbir şey yazmadıysa hepsi aynı)")

print()
print("=== 2) Uç değer sayıları (sadece sorun olan dosyalar) ===")
debi = "Coolant Volume Flow +500 [l/h]"
satirlar = []
dusuk_debi_guc = []
diger_debi_guc = []
for f in files:
    d = pd.read_csv(f, sep=";")
    ad = os.path.basename(f)
    maske = d[debi] < -100
    satirlar.append({
        "dosya": ad,
        "satır": len(d),
        "ısıtıcı V>450": int((d["Heater Voltage [V]"] > 450).sum()),
        "ısıtıcı P>10kW": int((d["Heating Power LIN [W]"] > 10000).sum()),
        "gaz>100": int((d["Throttle [%]"] > 100).sum()),
        "debi<-100": int(maske.sum()),
        "ilk debi<-100 satırı": int(maske.idxmax()) if maske.any() else -1,
    })
    dusuk_debi_guc.append(d.loc[maske, "Heating Power LIN [W]"])
    diger_debi_guc.append(d.loc[~maske, "Heating Power LIN [W]"])

r = pd.DataFrame(satirlar)
sorunlu = r[(r.drop(columns=["dosya", "satır", "ilk debi<-100 satırı"]) > 0).any(axis=1)]
print(sorunlu.to_string(index=False))

print()
print("=== 3) Debi çok düşükken ısıtıcı gücü ne kadar? ===")
print("debi < -100 iken ort. ısıtıcı gücü:", round(pd.concat(dusuk_debi_guc).mean(), 1))
print("diğer zamanlarda ort. ısıtıcı gücü:", round(pd.concat(diger_debi_guc).mean(), 1))