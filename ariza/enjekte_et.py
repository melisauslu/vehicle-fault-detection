import glob
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arizalar import (SENSOR_ARIZALARI, SADECE_ORAN_OLCEKLI, aralik, sensor_arizasi, olcum_ozellikleri, olcum_zinciri,
                      ic_direnc, ocv_egimi, kapasite, isitici_olcegi, isitici_zaman_sabiti, isitici_guc_kaybi,
                      pompa_debi_kaybi, batarya_direnc_artisi)

ANA_KLASOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMIZ = os.path.join(ANA_KLASOR, "data", "clean_B")
CIKTI = os.path.join(ANA_KLASOR, "data", "faulty")
RAPOR = os.path.join(ANA_KLASOR, "reports", "ariza")
SPLIT = os.path.join(ANA_KLASOR, "split.csv")

V_BAT = "Battery Voltage [V]"
I_BAT = "Battery Current [A]"
T_BAT = "Battery Temperature [°C]"
SOC = "SoC [%]"
ISI_P = "Heating Power LIN [W]"
ISI_V = "Heater Voltage [V]"
ISI_I = "Heater Current [A]"
GIRIS = "Temperature Coolant Heater Inlet [°C]"
CIKIS = "Temperature Coolant Heater Outlet [°C]"
# ısıtıcı arızasında birlikte değişen sinyaller (arizalar.py'deki kısa adlarla)
ISI_SUTUN = {"P": ISI_P, "I_bat": I_BAT, "V_bat": V_BAT, "SoC": SOC, "V_isi": ISI_V, "I_isi": ISI_I, "T_cikis": CIKIS}
DEBI = "Coolant Volume Flow +500 [l/h]"
# pompa arızasında kullanılan sinyaller; yalnızca T_cikis (ve debi_olculen ise debi) değişir
POMPA_SUTUN = {"P": ISI_P, "T_giris": GIRIS, "T_cikis": CIKIS, "debi": DEBI}
POMPA_DEGISEN = ["T_cikis", "debi"]
# BİZİM KARARIMIZ: debi sütunu ayar değeri kabul edilir → pompa arızasında değişmez (gerekçe arizalar.py'de)
# Veri seti açıklamasında sütunun "ölçülen debi" olduğu yazıyorsa True yap
DEBI_OLCULEN = False

# batarya iç direnç arızasında kullanılan sinyaller; ısıtıcı gücü yalnızca ısıtıcı akımını hesaplamak için
BAT_SUTUN = {"V_bat": V_BAT, "I_bat": I_BAT, "SoC": SOC, "V_isi": ISI_V, "I_isi": ISI_I, "P_isi": ISI_P}
# Direnç artışı k — KAYNAKTAN [Grandjean]: k = 2 ömür sonu; k = 1.33 BİZİM KARARIMIZ (Yue η = 0.75 → R/η)
K_DEGERLERI = [4 / 3, 2.0]

ARIZA_TIPLERI = SENSOR_ARIZALARI + ["isitici_guc_kaybi", "pompa_debi_kaybi", "batarya_direnc_artisi"]

# BİZİM KARARIMIZ (fiziksel gerekçe arizalar.py'de): kazanç ve tam arıza yalnızca gerilim ve akıma
TUM_SENSORLER = [V_BAT, I_BAT, T_BAT]
ORAN_SENSORLERI = [V_BAT, I_BAT]

# Sensör arızası şiddeti L — KAYNAKTAN [Wang]: L 0.05 ile 0.5 arasında denenmiş, ana sonuçlar L = 0.2
# BİZİM KARARIMIZ: bu üç değerin seçilmesi (uçlar + ana değer). Wang'ın denediği değerleri PDF'ten kontrol et.
L_DEGERLERI = [0.05, 0.2, 0.5]
# Isıtıcı arızası şiddeti η — KAYNAKTAN [Yue]: η = 0.75 (%25 güç kaybı) ve 0.5 (%50 güç kaybı)
ETA_DEGERLERI = [0.75, 0.5]
# Arızanın başlama anı — KAYNAKTAN [Safavi]: sinyalin %25'inden sonra
# BİZİM KARARIMIZ: en geç %75'te başlar → her dosyada en az %25 arızalı veri olur
BAS_ORAN, SON_ORAN = 0.25, 0.75
ORNEK_SURUS = "TripB17.csv"      # grafiği çizilecek sürüş
# BİZİM KARARIMIZ: arızalı satırların %5'inden azında sinyal değiştiyse arıza sağlam veriden ayırt edilemez
# (ör. yavaş değişen sıcaklıkta Wang'ın aralığa bağlı ölçeği 1 °C çözünürlüğün altında kalıyor).
# Bu dosyalar silinmez; ariza_listesi.csv'de kullanilabilir = False olur, model aşamasında dışarıda bırakılır.
GORUNURLUK_ESIGI = 0.05

# KAYNAKTAN [Safavi]: arıza parametreleri sözde rastgele seçilir; seed 42 ile her çalıştırmada aynı sonuç
rng = np.random.default_rng(seed=42)

dosyalar = sorted(glob.glob(os.path.join(TEMIZ, "*.csv")))
if not dosyalar:
    raise SystemExit(f"Temiz veri bulunamadı: {TEMIZ}  (önce prepare_data.py çalıştırılmalı)")

grup = {}
if os.path.exists(SPLIT):
    s = pd.read_csv(SPLIT, sep=";")
    grup = dict(zip(s["dosya"], s["grup"]))
else:
    print("UYARI: split.csv yok, grup sütunu '?' olacak (önce split_trips.py çalıştırılmalı)")

veri = {os.path.basename(f): pd.read_csv(f, sep=";") for f in dosyalar}

# ============================================================================
# Fiziksel sabitler: hepsi verinin kendisinden tahmin edilir (formüller arizalar.py'de)
# ============================================================================
R_surus = {ad: ic_direnc(d[V_BAT].to_numpy(), d[I_BAT].to_numpy()) for ad, d in veri.items()}
R_gecerli = [r for r in R_surus.values() if r is not None]
if not R_gecerli:
    raise SystemExit("Hiçbir sürüşte iç direnç hesaplanamadı (1 s'de > 30 A akım değişimi yok)")
R_ortanca = float(np.median(R_gecerli))
if R_ortanca <= 0:
    raise SystemExit("İç direnç negatif çıktı → akım işareti beklenenin tersi; ısıtıcı formülü kontrol edilmeli")
for ad in R_surus:                     # yeterli akım değişimi olmayan sürüşte ortanca R kullanılır
    R_surus[ad] = R_surus[ad] if R_surus[ad] is not None else R_ortanca

Q_list = [q for q in (kapasite(d[I_BAT].to_numpy(), d[SOC].to_numpy()) for d in veri.values()) if q is not None]
b_list = [ocv_egimi(d[V_BAT].to_numpy(), d[SOC].to_numpy(), d[I_BAT].to_numpy())
          for d in veri.values() if d[SOC].max() - d[SOC].min() >= 5]
if not Q_list or not b_list:
    raise SystemExit("SoC hiçbir sürüşte %5'ten fazla değişmemiş; Q ve b veriden hesaplanamıyor")
Q = float(np.median(Q_list))
b = float(np.median(b_list))

P_list = [d[ISI_P].to_numpy() for d in veri.values()]
dT_list = [(d[CIKIS] - d[GIRIS]).to_numpy() for d in veri.values()]
K = isitici_olcegi(np.concatenate(P_list), np.concatenate(dT_list))
tau, tau_adaylar = isitici_zaman_sabiti(P_list, dT_list, K)

# Isıtıcı batarya ile aynı yüksek gerilim hattında mı? (gerilimleri %2'den yakınsa evet)
oranlar = np.concatenate([(d[ISI_V] / d[V_BAT])[d[ISI_V] > 1].to_numpy() for d in veri.values()])
hv_ortak = bool(len(oranlar) and abs(np.median(oranlar) - 1) < 0.02)

# Arıza eklenen her sinyalin ölçüm zinciri: çözünürlük g ve güncelleme aralığı u (ayrıntı arizalar.py'de)
OLCUM, REF = {}, {}
for c in sorted(set(TUM_SENSORLER) | set(ISI_SUTUN.values()) | {DEBI}):
    OLCUM[c] = olcum_ozellikleri([d[c].to_numpy() for d in veri.values()])
    REF[c] = pd.concat([d[c] for d in veri.values()]).mode().iloc[0]   # ızgaranın hizalanacağı sık görülen değer

print("=== Ölçüm zinciri (veriden): gerçek çözünürlük ve sensörün güncellenme aralığı ===")
for c, (g, u) in OLCUM.items():
    print(f"  {c:42s} çözünürlük {'belirlenemedi (zincir uygulanmaz)' if g is None else round(g, 4)}"
          f"{'' if g is None else f'   güncelleme ~{u * 0.1:.1f} s'}")
print()

print("=== Veriden tahmin edilen fiziksel sabitler ===")
print(f"İç direnç R (ortanca, sürüş başına ayrı kullanılır): {R_ortanca:.4f} ohm  "
      f"[{min(R_gecerli):.3f} – {max(R_gecerli):.3f}]   (check_battery: 0.1235)")
print(f"Kapasite Q (BMS SoC'si ile tutarlı): {Q:.1f} Ah   ({len(Q_list)} sürüşten; veri seti: 60 Ah)")
print(f"V₀ eğimi b = dV₀/dSoC: {b:.3f} V/%   ({len(b_list)} sürüşten)")
print(f"Isıtıcı ölçeği K = ṁ·c_p: {K:.1f} W/°C   (estimate_flow.py ile aynı koşullar)")
print(f"Isıl gecikme τ: {tau} s" + ("   UYARI: aday sınırında, aralık genişletilmeli" if tau == tau_adaylar[-1] else ""))
print(f"Isıtıcı batarya ile aynı hatta mı: {'evet' if hv_ortak else 'hayır'}   "
      f"(V_ısıtıcı / V_batarya ortanca = {np.median(oranlar):.3f})\n")

shutil.rmtree(CIKTI, ignore_errors=True)
os.makedirs(CIKTI)
os.makedirs(RAPOR, exist_ok=True)

liste, ornekler, isitici_ornek, pompa_ornek, batarya_ornek = [], [], None, None, None
guc_sinirli = 0          # yaşlı bataryanın istenen gücü veremediği satır sayısı
for ad, d in veri.items():
    N = len(d)

    # BİZİM KARARIMIZ: her sürüşten her arıza tipi için bir kopya (6 sensör + 1 ısıtıcı + 1 pompa = 8)
    for tip in ARIZA_TIPLERI:
        # KAYNAKTAN [Meckel]: başlangıç anı sürüş durumundan bağımsız, rastgele
        n0 = int(rng.integers(int(N * BAS_ORAN), int(N * SON_ORAN)))
        a = d.copy()

        if tip == "isitici_guc_kaybi":
            eta = float(rng.choice(ETA_DEGERLERI))
            p = {"K": K, "tau": tau, "R": R_surus[ad], "b": b, "Q": Q, "hv_ortak": hv_ortak}
            yeni = isitici_guc_kaybi({k: d[c].to_numpy() for k, c in ISI_SUTUN.items()}, n0, eta, p)
            for k, c in ISI_SUTUN.items():
                if not np.array_equal(yeni[k], d[c].to_numpy()):     # değişmeyen sütuna dokunulmaz
                    a[c] = olcum_zinciri(yeni[k], n0, *OLCUM[c], REF[c])
            sinyal, siddet, ar = ISI_P, 1 - eta, np.nan
            degisen = ISI_P
            if ad == ORNEK_SURUS:
                isitici_ornek = (eta, n0, d, a)
        elif tip == "pompa_debi_kaybi":
            eta = float(rng.choice(ETA_DEGERLERI))
            p = {"K": K, "tau": tau, "debi_olculen": DEBI_OLCULEN}
            yeni = pompa_debi_kaybi({k: d[c].to_numpy() for k, c in POMPA_SUTUN.items()}, n0, eta, p)
            for k in POMPA_DEGISEN:
                c = POMPA_SUTUN[k]
                if not np.array_equal(yeni[k], d[c].to_numpy()):     # değişmeyen sütuna dokunulmaz
                    a[c] = olcum_zinciri(yeni[k], n0, *OLCUM[c], REF[c])
            # arızanın konumu pompa (debi); görünürlük ise asıl etkinin olduğu çıkış suyunda ölçülür
            sinyal, siddet, ar = DEBI, 1 - eta, np.nan
            degisen = CIKIS
            if ad == ORNEK_SURUS:
                pompa_ornek = (eta, n0, d, a)
        elif tip == "batarya_direnc_artisi":
            kat = float(rng.choice(K_DEGERLERI))
            p = {"R": R_surus[ad], "b": b, "Q": Q, "hv_ortak": hv_ortak}
            yeni, sinirli = batarya_direnc_artisi({k: d[c].to_numpy() for k, c in BAT_SUTUN.items()}, n0, kat, p)
            guc_sinirli += sinirli
            for k, c in BAT_SUTUN.items():
                if not np.array_equal(yeni[k], d[c].to_numpy()):     # değişmeyen sütuna dokunulmaz
                    a[c] = olcum_zinciri(yeni[k], n0, *OLCUM[c], REF[c])
            # şiddet = direncin bağıl artışı (k − 1): 0.33 ya da 1.0
            sinyal, siddet, ar = V_BAT, kat - 1, np.nan
            degisen = V_BAT
            if ad == ORNEK_SURUS:
                batarya_ornek = (kat, n0, d, a)
        else:
            # BİZİM KARARIMIZ: arızalı sensör izin verilen sensörlerden rastgele seçilir
            sinyal = str(rng.choice(ORAN_SENSORLERI if tip in SADECE_ORAN_OLCEKLI else TUM_SENSORLER))
            # takılma ve tam arızada L kullanılmaz (Wang'da da bu ikisinin şiddeti yok)
            L = float(rng.choice(L_DEGERLERI)) if tip not in ("takilma", "tam_ariza") else np.nan
            x = d[sinyal].to_numpy()
            y = sensor_arizasi(x, n0, tip, L, rng)
            if tip not in ("takilma", "tam_ariza"):     # bu ikisi zaten sabit, geçerli bir değerde
                y = olcum_zinciri(y, n0, *OLCUM[sinyal], REF[sinyal])
            a[sinyal] = y
            siddet, ar = L, aralik(x, n0)
            degisen = sinyal

        # Etiketler: tip, konum (sinyal) ve zaman birlikte tutulur [Abboush §3.2]
        # Etiket bileşenin durumunu gösterir: ısıtıcı kapalıyken de arızalı bileşen "arızalı" sayılır
        arizali = np.arange(N) >= n0
        a["ariza"] = arizali.astype(int)
        a["ariza_tipi"] = np.where(arizali, tip, "saglam")
        a["ariza_sinyali"] = np.where(arizali, sinyal, "")
        a["ariza_siddeti"] = np.where(arizali, siddet, 0.0)

        # Kontrol: arıza öncesi satırlar sağlam veriyle birebir aynı olmalı
        ayni = np.array_equal(a.loc[:n0 - 1, d.columns].to_numpy(float), d.loc[:n0 - 1].to_numpy(float))

        # Arızalı satırların yüzde kaçında sinyal gerçekten değişti? (çözünürlükten küçük arızalar görünmez)
        degisen_oran = float(np.mean(np.abs(a[degisen].to_numpy()[n0:] - d[degisen].to_numpy()[n0:]) > 1e-9))

        cikti_ad = f"{ad[:-4]}_{tip}.csv"
        a.to_csv(os.path.join(CIKTI, cikti_ad), sep=";", index=False)
        liste.append({"dosya": cikti_ad, "kaynak": ad, "grup": grup.get(ad, "?"),
                      "ariza_tipi": tip, "ariza_sinyali": sinyal, "ariza_siddeti": siddet,
                      "baslangic_dk": round(n0 * 0.1 / 60, 1),
                      "arizali_oran": round(1 - n0 / N, 2), "aralik": ar,
                      "degisen_oran": round(degisen_oran, 3),
                      "kullanilabilir": degisen_oran >= GORUNURLUK_ESIGI,
                      "R_ohm": round(R_surus[ad], 4), "oncesi_ayni": ayni})
        if ad == ORNEK_SURUS:
            ornekler.append((tip, sinyal, siddet, n0, d[degisen].to_numpy(), a[degisen].to_numpy()))

l = pd.DataFrame(liste)
l.to_csv(os.path.join(CIKTI, "ariza_listesi.csv"), sep=";", index=False)

pd.set_option("display.width", 220)
print(f"Üretilen arızalı dosya: {len(l)}  ({len(veri)} sürüş × {len(ARIZA_TIPLERI)} arıza tipi)\n")
print("Arıza tipi × grup:")
print(pd.crosstab(l["ariza_tipi"], l["grup"]).to_string())
print("\nArıza tipi × sensör:")
print(pd.crosstab(l["ariza_tipi"], l["ariza_sinyali"]).to_string())
print("\nArızalı satır oranı (ortalama):", round(l["arizali_oran"].mean(), 2))
print(f"Batarya direnç arızasında istenen gücün verilemediği satır: {guc_sinirli} "
      f"(bu satırlarda batarya en yüksek gücünde tutuldu; BMS güç sınırlaması modellenmedi)")
print("\nArızalı satırlarda sinyalin gerçekten değiştiği oran (arıza tipi × sensör, ortalama):")
print(l.pivot_table(index="ariza_tipi", columns="ariza_sinyali", values="degisen_oran", aggfunc="mean").round(2).to_string())
gorunmez = l[~l["kullanilabilir"]]
print(f"\nGörünmeyen arıza (değişen oran < %{GORUNURLUK_ESIGI * 100:.0f}) → kullanilabilir = False: {len(gorunmez)} dosya")
if len(gorunmez):
    print(gorunmez[["dosya", "ariza_sinyali", "ariza_siddeti", "aralik", "degisen_oran"]].to_string(index=False))
k = l[l["kullanilabilir"]]
print(f"\nModelde kullanılacak arızalı dosya: {len(k)} / {len(l)}")
print("Kullanılabilir dosyalar, arıza tipi × grup:")
print(pd.crosstab(k["ariza_tipi"], k["grup"], margins=True, margins_name="toplam").to_string())
print("Arıza öncesi satırlar her dosyada aynı mı?", "EVET" if l["oncesi_ayni"].all() else "HAYIR – kontrol et!")

def ciz(ax, t, eski, yeni, n0, baslik):
    ax.plot(t, eski, color="gray", linewidth=1, label="sağlam")
    ax.plot(t, yeni, color="red", linewidth=0.8, label="arızalı")
    ax.axvline(n0 * 0.1 / 60, color="black", linestyle="--")
    ax.set_title(baslik, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

if ornekler:
    fig, ax = plt.subplots(len(ornekler), 1, figsize=(12, 2.4 * len(ornekler)), sharex=True)
    t = np.arange(len(ornekler[0][4])) * 0.1 / 60
    for k, (tip, sinyal, siddet, n0, eski, yeni) in enumerate(ornekler):
        ciz(ax[k], t, eski, yeni, n0, f"{tip} – {sinyal} – şiddet {'-' if np.isnan(siddet) else round(siddet, 2)}")
    ax[-1].set_xlabel("Dakika (kesikli çizgi = arızanın başladığı an)")
    plt.tight_layout()
    plt.savefig(os.path.join(RAPOR, f"{ORNEK_SURUS[:-4]}_arizalar.png"), dpi=120)

    # Isıtıcı arızasında birlikte değişen bütün sinyaller
    eta, n0, d, a = isitici_ornek
    fig, ax = plt.subplots(len(ISI_SUTUN), 1, figsize=(12, 2.2 * len(ISI_SUTUN)), sharex=True)
    t = np.arange(len(d)) * 0.1 / 60
    for k, c in enumerate(ISI_SUTUN.values()):
        ciz(ax[k], t, d[c], a[c], n0, f"{c}   (ısıtıcı güç kaybı, η = {eta})")
    ax[-1].set_xlabel("Dakika (kesikli çizgi = arızanın başladığı an)")
    plt.tight_layout()
    plt.savefig(os.path.join(RAPOR, f"{ORNEK_SURUS[:-4]}_isitici_ayrinti.png"), dpi=120)

    # Pompa arızasında ilgili sinyaller (ısıtıcı gücü ve giriş suyu değişmez, çıkış suyu ısınır)
    eta, n0, d, a = pompa_ornek
    fig, ax = plt.subplots(len(POMPA_SUTUN), 1, figsize=(12, 2.2 * len(POMPA_SUTUN)), sharex=True)
    t = np.arange(len(d)) * 0.1 / 60
    for k, c in enumerate(POMPA_SUTUN.values()):
        ciz(ax[k], t, d[c], a[c], n0, f"{c}   (pompa debi kaybı, η = {eta})")
    ax[-1].set_xlabel("Dakika (kesikli çizgi = arızanın başladığı an)")
    plt.tight_layout()
    plt.savefig(os.path.join(RAPOR, f"{ORNEK_SURUS[:-4]}_pompa_ayrinti.png"), dpi=120)

    # Batarya iç direnç arızasında birlikte değişen sinyaller (ısıtıcı gücü değişmez)
    kat, n0, d, a = batarya_ornek
    fig, ax = plt.subplots(len(BAT_SUTUN), 1, figsize=(12, 2.2 * len(BAT_SUTUN)), sharex=True)
    t = np.arange(len(d)) * 0.1 / 60
    for k, c in enumerate(BAT_SUTUN.values()):
        ciz(ax[k], t, d[c], a[c], n0, f"{c}   (batarya iç direnç ×{kat:.2f})")
    ax[-1].set_xlabel("Dakika (kesikli çizgi = arızanın başladığı an)")
    plt.tight_layout()
    plt.savefig(os.path.join(RAPOR, f"{ORNEK_SURUS[:-4]}_batarya_ayrinti.png"), dpi=120)
    plt.show()
else:
    print(f"\n{ORNEK_SURUS} bulunamadı, grafik çizilmedi")
