import pandas as pd

liste = pd.read_csv("data/faulty/ariza_listesi.csv", sep=";")
print("Toplam arızalı dosya:", len(liste))

print("\nEşik değiştikçe kaç dosya 'görünmez' sayılırdı?")
for esik in [0.01, 0.02, 0.05, 0.10, 0.20]:
    elenen = (liste["degisen_oran"] < esik).sum()
    print(f"  eşik %{esik*100:4.0f}  →  {elenen:3d} dosya elenirdi")

print("\n%5 eşiğinde elenen dosyalar hangi arıza ve sinyalde?")
x = liste[liste["degisen_oran"] < 0.05]
print(x.groupby(["ariza_tipi", "ariza_sinyali"]).size().to_string())