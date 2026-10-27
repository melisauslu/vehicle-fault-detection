import json
import os
import numpy as np
import pandas as pd

# Bütün model kodlarının ortak kullandığı fonksiyonlar.
# VERİ SIZINTISI KURALI: veriden öğrenilen her şey (sabitler, normalizasyon, model) yalnızca o katın
# EĞİTİM pencerelerinden öğrenilir; doğrulama/test pencerelerine yalnızca uygulanır.

ANA_KLASOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERI = os.path.join(ANA_KLASOR, "data", "model")
DT = 0.1


def veriyi_oku():
    oz = pd.read_csv(os.path.join(VERI, "pencereler_ozellik.csv"), sep=";", low_memory=False)
    with open(os.path.join(VERI, "bilgi.json"), encoding="utf-8") as f:
        bilgi = json.load(f)
    return oz, bilgi


def bolmeler(oz):
    # Sürüş bazında bölmeler. Aynı sürüşün sağlam ve arızalı bütün pencereleri hep aynı tarafta kalır.
    #   5 çapraz doğrulama katı: eğitim = diğer 4 kat, doğrulama = o kat (son test sürüşleri hiç girmez)
    #   son değerlendirme:      eğitim = 28 eğitim sürüşü, test = 7 test sürüşü (kat = −1)
    katlar = sorted(k for k in oz["kat"].unique() if k >= 0)
    cv = [((oz["kat"] >= 0) & (oz["kat"] != k), oz["kat"] == k) for k in katlar]
    son = ((oz["kat"] >= 0), (oz["kat"] == -1))
    return cv, son


def kat_sabitleri(oz, egitim, bilgi):
    # Fizik kalıntılarının sabitleri — YALNIZCA bu katın eğitim pencerelerindeki SAĞLAM pencerelerden
    s = oz[egitim & (oz["ariza"] == 0)]
    P, dT = s["isitici_guc_pozitif__ort"], s["su_isinmasi__ort"]
    # K = ṁ·c_p [Tez-O Eq 4.26]: ısıtıcı açıkken P/ΔT (koşullar estimate_flow.py ile aynı)
    m = (P > 1000) & (dT > 3)
    K = float(np.median(P[m] / dT[m]))
    # Q [Tez-B Eq 3.26]: ∫I dt / ΔSoC (pencereler üst üste binse de oran değişmez)
    Q = float(s["akim_Ah"].sum() / (s["soc_degisim"].sum() / 100))
    # beklenen V–I eğimi = a·sıcaklık + b [Rint; sıcaklık ilişkisi check_battery: r = −0.96]
    e = s[["vi_egimi", "Battery Temperature [°C]__ort"]].dropna()
    R_fit = np.polyfit(e["Battery Temperature [°C]__ort"], e["vi_egimi"], 1)
    return {"K": K, "Q": Q, "R_fit": R_fit}


def fizik_ekle(oz, sabitler):
    # Sabite bağlı kalıntılar (bu katın sabitleriyle) — tanımlar veri_hazirla.py'de
    o = oz.copy()
    o["su_isinmasi_farki__ort"] = o["su_isinmasi__ort"] - o["isitici_guc_pozitif__ort"] / sabitler["K"]
    o["soc_farki"] = o["soc_degisim"] - 100 * o["akim_Ah"] / sabitler["Q"]
    o["direnc_orani"] = o["vi_egimi"] / np.polyval(sabitler["R_fit"], o["Battery Temperature [°C]__ort"])
    if "soc_degisim_5dk" in o:                             # uzun geçmiş özelliği (veri_hazirla.py'de UZUN)
        o["soc_farki_5dk"] = o["soc_degisim_5dk"] - 100 * o["akim_Ah_5dk"] / sabitler["Q"]
    return o


# Fizik kalıntısı → hesaplanırken kullandığı sensörler (tanımlar veri_hazirla.py'de).
# Bir sensör çıkarılınca/karıştırılınca onu kullanan kalıntılar da çıkar/karışır (o sensör yoksa hesaplanamazlar).
V_BAT, I_BAT, T_BAT, SOC = "Battery Voltage [V]", "Battery Current [A]", "Battery Temperature [°C]", "SoC [%]"
ISI_P, ISTENEN = "Heating Power LIN [W]", "Requested Heating Power [W]"
ISI_V, ISI_I = "Heater Voltage [V]", "Heater Current [A]"
GIRIS, CIKIS = "Temperature Coolant Heater Inlet [°C]", "Temperature Coolant Heater Outlet [°C]"
KALINTI_SENSORLERI = {
    "bat_guc": [V_BAT, I_BAT],
    "gerilim_farki": [ISI_V, V_BAT],
    "isitici_VI_farki": [ISI_V, ISI_I, ISI_P],
    "isitici_guc_farki": [ISTENEN, ISI_P],
    "su_isinmasi": [CIKIS, GIRIS],
    "soc_farki": [SOC, I_BAT],
    "soc_farki_5dk": [SOC, I_BAT],
    "su_isinmasi_farki": [CIKIS, GIRIS, ISI_P],
    "direnc_orani": [V_BAT, I_BAT, T_BAT],
}


def sensoru_kullanir(kolon, sensorler):
    # Bu özellik verilen sensörlerden birinin istatistiği mi, ya da onlardan birini kullanan bir kalıntı mı?
    kok = kolon.split("__")[0]
    return kok in sensorler or any(s in sensorler for s in KALINTI_SENSORLERI.get(kok, []))


def ozellik_listesi(bilgi, gruplar, cikarilan=()):
    # gruplar: ["ham_istatistik"], ["ham_istatistik", "fizik_dusuk"], ya da üçü birden (+ "fizik_riskli")
    # cikarilan: modelden çıkarılacak sensörler (kendi istatistikleri + onları kullanan kalıntılar çıkar)
    return [c for g in gruplar for c in bilgi["ozellik_gruplari"][g] if not sensoru_kullanir(c, cikarilan)]
