import numpy as np
from scipy.signal import lfilter

# ============================================================================
# KAYNAKLAR (tezde kaynakçaya girecek olanlar)
# [Wang]   Wang vd. 2025, Processes 13, 929 – sensör arıza formülleri ve şiddet ölçeği L
# [Yue]    Yue vd. 2019 – eyleyici (aktüatör) çarpımsal arızası y = η·u (Eq 14), şiddetler η = 0.75 ve 0.5
#          (ısıtıcı ve pompa ikisi de eyleyicidir → aynı çerçeve)
# [Tez-B]  Türkçe Li-iyon batarya tezi – P = V·I (Eq 3.8), P = I²R (Eq 3.10),
#          toplu ısıl model dT/dt = (1/C)[ısı girişi − kayıp] (Eq 3.25), SOC = SOC₀ + ∫i dt / Q (Eq 3.26)
# [Tez-O]  Otel ısı pompası tezi – ısı dengesi Q = ṁ·c_p·ΔT (Eq 4.26)
# [Rint]   Steger 2022 (Eq 9), Ng 2009 – batarya modeli V = V₀(SoC) + R·I  (+ akım = şarj)
# [Meckel] Meckel vd. – arıza başladıktan sonra kalıcıdır (§II.B)
# [Grandjean] Grandjean vd. 2018, Batteries 4(4) 49, §1 – "EOL is commonly defined as being attained when the
#          battery internal resistance has doubled" (hibrit araçlar için; kaynağı [Farmann] Farmann vd. 2015,
#          J. Power Sources 281, 114–130). Aynı yerde: direnç artışı gücü sınırlar ve ek ısı üretir [Vetter 2005]
#
# Her hesabın yanında üç etiketten biri var:
#   KAYNAKTAN       : formül kaynakta aynen var
#   TÜRETME         : kaynaktaki formülden adım adım çıkarıldı (adımlar yazılı)
#   BİZİM KARARIMIZ : hiçbir kaynakta yok, tezde "bu çalışmada seçilmiştir" diye yazılacak
# Sabitler (R, K, τ, Q, b) kitaptan alınmaz, hepsi bu araçların kendi verisinden tahmin edilir.
# Bölüm/denklem numaralarını PDF'ten kontrol et.
# ============================================================================

DT = 0.1         # örnekleme aralığı [s] (check_data.py ile doğrulandı: 10 Hz)
PENCERE = 5000   # KAYNAKTAN [Wang]: aralık son 500 s'den hesaplanır → 500 s / 0.1 s = 5000 satır

SENSOR_ARIZALARI = ["surukleme", "sabit_kayma", "kazanc", "dogruluk_kaybi", "takilma", "tam_ariza"]

# BİZİM KARARIMIZ (fiziksel gerekçeli): kazanç (×G) ve tam arıza (=0) yalnızca GERÇEK SIFIRI olan
# büyüklüklere uygulanır: gerilim (0 V = gerilim yok) ve akım (0 A = akım yok).
# °C'nin sıfırı keyfidir (suyun donma noktası): 20 °C × 0.3 = 6 °C ama 293 K × 0.3 = 88 K = −185 °C.
# Aynı arıza birime göre farklı sonuç verdiği için sıcaklığa çarpma ve "0" uygulamak fiziksel olarak anlamsızdır.
SADECE_ORAN_OLCEKLI = {"kazanc", "tam_ariza"}


# ---------------------------------------------------------------------------
# 1) SENSÖR ARIZALARI
# ---------------------------------------------------------------------------

def aralik(x, n0):
    # Sensör arızasının büyüklüğünü sinyalin kendi ölçeğine göre ayarlamak için.
    # KAYNAKTAN [Wang]: aralık = max − min, [t0 − 500 s, t0] penceresinde
    # BİZİM KARARIMIZ: arızadan önce 500 s veri yoksa eldeki kadarı kullanılır
    # Örnek: son 500 s'de gerilim 350–360 V arasında → aralık = 10 V
    gecmis = x[max(0, n0 - PENCERE):n0 + 1]
    return gecmis.max() - gecmis.min()


def sensor_arizasi(x, n0, tip, L, rng):
    # SENSÖR ARIZASI: batarya sağlam, ölçen sensör yanlış gösteriyor.
    # Fiziksel büyüklük değişmediği için başka hiçbir sinyal değiştirilmez
    # (bozuk bir termometre bataryayı ısıtmaz).
    # SINIRLILIK: gerçek araçta kontrol sistemleri yanlış ölçüme tepki verebilir (ör. BMS'in SoC tahmini
    # akım ölçümüyle hesaplanır). BMS algoritması bilinmediği için bu tepki modellenmez.
    # KAYNAKTAN [Meckel]: arıza n0'dan dosya sonuna kadar sürer (kalıcı)
    # BİZİM KARARIMIZ: Wang formüllerindeki δ (ölçüm gürültüsü) eklenmez; gerçek veride gürültü zaten var
    y = x.astype(float).copy()
    a = aralik(x, n0)
    n = np.arange(len(x) - n0)          # arıza başladığından beri geçen satır sayısı: 0, 1, 2, ...

    if tip == "surukleme":
        # KAYNAKTAN [Wang]: x̂ = x + k·t,  k = L·aralık/5000
        # BİZİM KARARIMIZ: t yerine "arıza başladığından beri geçen satır" → t0'da sıçrama olmaz
        # SINIRLILIK: gerçek sensör sürüklenmesi aylar sürer; burada model fark edebilsin diye
        # 500 s'de L·aralık kadar hızlandırılmıştır (Wang'ın ölçeği)
        # Örnek: aralık 10 V, L 0.2 → k = 0.0004 V/satır → 500 s (5000 satır) sonra +2 V
        y[n0:] += (L * a / PENCERE) * n
    elif tip == "sabit_kayma":
        # KAYNAKTAN [Wang]: x̂ = x + C,  C = L·aralık
        # Örnek: aralık 10 V, L 0.2 → C = 2 V → 355 V olan gerilim 357 V okunur
        y[n0:] += L * a
    elif tip == "kazanc":
        # KAYNAKTAN [Wang]: x̂ = G·x,  G = −(2/π)·arctan(10L) + 1   (yalnızca gerilim/akım, yukarıya bak)
        # Örnek: L 0.2 → arctan(2) = 1.107 → G = 1 − 0.705 = 0.295 → −50 A, −14.8 A okunur
        y[n0:] *= -(2 / np.pi) * np.arctan(10 * L) + 1
    elif tip == "dogruluk_kaybi":
        # KAYNAKTAN [Wang]: x̂ = x + σ(t),  σ ~ N(0, R),  R = (L·aralık)²/9
        # R varyanstır; numpy standart sapma ister → √R = L·aralık/3
        # Örnek: aralık 10 V, L 0.2 → standart sapma 0.67 V'luk rastgele titreşim
        y[n0:] += rng.normal(0, L * a / 3, size=len(n))
    elif tip == "takilma":
        # KAYNAKTAN [Wang]: x̂ = x(t0) → sensör arızanın başladığı andaki değerde donar
        y[n0:] = x[n0]
    elif tip == "tam_ariza":
        # KAYNAKTAN [Wang]: x̂ = 0 → sensör tamamen ölür   (yalnızca gerilim/akım, yukarıya bak)
        y[n0:] = 0.0
    else:
        raise ValueError(tip)
    return y


# ---------------------------------------------------------------------------
# ÖLÇÜM ZİNCİRİ: sensör → kendi çözünürlüğü ve güncelleme hızı → veri setinde 10 Hz'e doğrusal ara doldurma
# cozunurluk_kontrol.py ile görüldü: batarya sıcaklığı 7 → 7.239 → 7.553 → 7.866 → 8
#   → sensör 1 °C adımlarla ölçüyor, aradaki değerler veri setinin doğrusal ara doldurmasından geliyor.
# BİZİM KARARIMIZ (fiziksel gerekçeli): arızalı sensör de aynı zincirden geçer. Geçirilmezse model
# "ızgara dışı değer" ya da "sensörün güncellenemeyeceği hızda değişim" gibi kısayolları öğrenebilir.
# ---------------------------------------------------------------------------

def olcum_ozellikleri(x_list, min_plato=10, min_sayi=50):
    # g: gerçek çözünürlük = en az 1 s (10 satır) sabit kalan değerler arasındaki en sık adım
    #    (0.09 / 0.08 gibi 2 haneye yuvarlama ikizleri ağırlıklı ortalamayla birleştirilir → 0.0873)
    # u: güncelleme aralığı [satır] = g / (g'den küçük 10 Hz adımlarının ortancası)
    #    ör. sıcaklık: 1 °C / ~0.11 °C ≈ 9 satır ≈ 1 s
    # Sinyal neredeyse hiç sabit kalmıyorsa (ör. batarya akımı) g belirlenemez → zincir uygulanmaz
    plato_adim, adim = [], []
    for x in x_list:
        fark = np.round(np.abs(np.diff(x)), 6)
        adim.append(fark[fark > 0])
        sinir = np.r_[0, np.flatnonzero(np.diff(x) != 0) + 1]
        uz = np.diff(np.r_[sinir, len(x)])
        p = np.round(np.abs(np.diff(x[sinir][uz >= min_plato])), 6)
        plato_adim.append(p[p > 0])
    plato_adim, adim = np.concatenate(plato_adim), np.concatenate(adim)
    if len(plato_adim) < min_sayi:
        return None, 1
    deger, sayi = np.unique(plato_adim, return_counts=True)
    sira = np.argsort(sayi)[::-1]
    g = deger[sira[0]]
    if len(sira) > 1 and abs(deger[sira[1]] - g) <= 0.015:      # yuvarlama ikizi
        g = np.average(deger[sira[:2]], weights=sayi[sira[:2]])
    rampa = adim[adim < 0.95 * g]
    u = max(1, int(round(g / np.median(rampa)))) if len(rampa) else 1
    return float(g), u


def olcum_zinciri(y, n0, g, u, ref):
    # Arızalı sinyali sensörün ölçüm zincirinden geçirir (yalnızca n0'dan sonrası):
    #   1) sensör her u satırda bir ölçer      2) değeri g çözünürlüğüne yuvarlar (ızgara ref'e hizalı)
    #   3) veri seti aradaki satırları doğrusal doldurur
    # Örnek (sıcaklık, g = 1, u = 9): 13.37 → 13, sonraki ölçüme kadar doğrusal geçiş
    if g is None:
        return y
    y = y.copy()
    idx = np.arange(n0, len(y), u)
    if idx[-1] != len(y) - 1:
        idx = np.r_[idx, len(y) - 1]
    yq = ref + np.round((y[idx] - ref) / g) * g
    y[n0:] = np.interp(np.arange(n0, len(y)), idx, yq)
    return y


# ---------------------------------------------------------------------------
# 2) VERİDEN TAHMİN EDİLEN FİZİKSEL SABİTLER (uydurma sabit kullanmamak için)
# ---------------------------------------------------------------------------

def ic_direnc(V, I, adim=10, min_dI=30):
    # Batarya iç direnci R [ohm] — check_battery.py ile AYNI yöntem (R = 0.1235 ohm, R² = 0.953 bulunmuştu)
    # KAYNAKTAN [Rint]: V = V₀ + R·I. 1 s içinde SoC neredeyse değişmez (check_soc: ≤ %0.2) → ΔV₀ ≈ 0
    # TÜRETME: ΔV = R·ΔI → en küçük kareler: R = Σ(ΔV·ΔI) / Σ(ΔI²)
    # R sıcaklıkla güçlü değiştiği için (r = −0.96) her sürüş için ayrı hesaplanır
    dV = V[adim:] - V[:-adim]
    dI = I[adim:] - I[:-adim]
    m = np.abs(dI) > min_dI
    if m.sum() < 50:
        return None
    return np.sum(dV[m] * dI[m]) / np.sum(dI[m] ** 2)


def ocv_egimi(V, SoC, I):
    # b = dV₀/dSoC [V/%]: SoC değişince boştaki gerilimin (V₀) ne kadar değiştiği
    # KAYNAKTAN [Rint]: V = V₀(SoC) + R·I → V₀ SoC'ye bağlıdır
    # BİZİM KARARIMIZ: bir sürüşün SoC aralığında V₀(SoC) doğrusal kabul edilir: V ≈ c + b·SoC + r·I
    # b en küçük kareler ile bulunur (burada r yalnızca b'yi doğru bulmak için var, kullanılmaz)
    A = np.column_stack([np.ones_like(SoC), SoC, I])
    katsayi, *_ = np.linalg.lstsq(A, V, rcond=None)
    return katsayi[1]


def kapasite(I, SoC):
    # Q [Ah]: SoC'yi %100 değiştirmek için gereken yük
    # KAYNAKTAN [Tez-B Eq 3.26]: SOC = SOC₀ + ∫i dt / Q
    # TÜRETME: Q = ∫i dt / ΔSOC   (∫i dt Ah cinsinden: Σ I·0.1 s / 3600)
    # Veri seti 60 Ah diyor; burada BMS'in SoC'si ile tutarlı olan değer veriden bulunur
    dSoC = (SoC[-1] - SoC[0]) / 100
    if abs(dSoC) < 0.05:          # SoC %5'ten az değiştiyse oran güvenilmez
        return None
    return (np.sum(I) * DT / 3600) / dSoC


def isitici_olcegi(P, dT, min_guc=1000, min_dT=3):
    # K = ṁ·c_p [W/°C] (ṁ: soğutma sıvısı debisi, c_p: ısı kapasitesi)
    # TÜRETME [Tez-O Eq 4.26]: Q = ṁ·c_p·ΔT → ṁ·c_p = Q / ΔT,  Q = ölçülen ısıtıcı gücü P
    # VARSAYIM: ısıtıcının harcadığı elektrik gücünün tamamı suya geçer (enerjinin korunumu)
    # Koşullar (P > 1000 W, ΔT > 3 °C) estimate_flow.py ile aynı: ısıtıcı kapalıyken bölme anlamsız olur
    # Örnek: P = 3000 W, ΔT = 18.2 °C → K = 165 W/°C
    m = (P > min_guc) & (dT > min_dT)
    return np.median(P[m] / dT[m])


def gecikme(x, tau, y0=0.0):
    # Birinci dereceden ısıl gecikme: τ·dy/dt = x − y
    # TÜRETME [Tez-B Eq 3.25 + Tez-O Eq 4.26]: toplu ısıl model C·dΔT/dt = Q − ṁ·c_p·ΔT
    #   → (C/K)·dΔT/dt = Q/K − ΔT  → τ = C/K. Ayrık hali (Euler): y[n] = y[n−1] + (Δt/τ)·(x[n] − y[n−1])
    a = min(DT / tau, 1.0)
    return lfilter([a], [1, -(1 - a)], x, zi=[(1 - a) * y0])[0]


def isitici_zaman_sabiti(P_list, dT_list, K, adaylar=(0.1, 1, 2, 5, 10, 20, 30, 60, 90, 120, 180, 300)):
    # τ [s]: suyun ısıtıcı gücündeki değişime ne kadar gecikmeyle tepki verdiği.
    # Kitaptan değer almamak için veriden bulunur: her aday τ için ölçülen P'den beklenen ΔT hesaplanır,
    # ölçülen ΔT'ye en yakın sonucu veren τ seçilir (en küçük kareler).
    hatalar = []
    for tau in adaylar:
        h = sum(np.sum((gecikme(P / K, tau, dT[0]) - dT) ** 2) for P, dT in zip(P_list, dT_list))
        hatalar.append(h)
    return adaylar[int(np.argmin(hatalar))], adaylar


# ---------------------------------------------------------------------------
# 3) ISITICI GÜÇ KAYBI
# ---------------------------------------------------------------------------

def isitici_guc_kaybi(s, n0, eta, p):
    # ISITICI ARIZASI: sensör değil cihaz bozuk → ısıtıcı normalde vereceği gücün yalnızca η katını verir.
    # Cihaz gerçekten daha az enerji kullandığı için fiziksel olarak bağlı TÜM sinyaller birlikte değişir.
    # s: sinyaller (sözlük, numpy dizileri), p: veriden tahmin edilen sabitler (K, tau, R, b, Q, hv_ortak)
    # İstenen güç (Requested Heating Power) DEĞİŞMEZ: kontrolcü hâlâ aynı gücü istiyor.
    # SINIRLILIK: kabin sıcaklığı ve ısıtıcı giriş suyu (devreden dönen su) yavaş yavaş düşerdi; bunun için
    # kabin/ısı değiştirici modeli gerekir, verimizde yok → değiştirilmez.
    y = {k: v.astype(float).copy() for k, v in s.items()}
    P = np.clip(s["P"], 0, None)        # güç negatif olamaz (negatif değerler ölçüm gürültüsü)
    eksik = (1 - eta) * P[n0:]          # ısıtıcının artık çekmediği güç [W]

    # 1) Isıtıcı gücü — KAYNAKTAN [Yue Eq 14]: P' = η·P
    #    Örnek: η = 0.75, P = 3000 W → P' = 2250 W, eksik = 750 W
    y["P"][n0:] = eta * s["P"][n0:]

    # 2) Batarya akımı — TÜRETME [Tez-B Eq 3.8]: ısıtıcı yüksek gerilim bataryasından beslenir;
    #    bataryadan çekilen güç 'eksik' kadar azalır → ΔI = eksik / V_bat
    #    İşaret: veride + akım = şarj (R > 0 çıktığı için doğrulandı) → deşarj azalınca akım + yöne kayar
    #    Örnek: 750 W / 360 V = +2.08 A  (−50 A → −47.9 A)
    dI = eksik / s["V_bat"][n0:]
    y["I_bat"][n0:] = s["I_bat"][n0:] + dI

    # 3) SoC — TÜRETME [Tez-B Eq 3.26]: ΔSOC = ∫ΔI dt / Q   (yüzde için ×100)
    #    Örnek: +2.08 A, 10 dk, Q = 60 Ah → 2.08 × 600 / 3600 / 60 × 100 = %0.58 daha yüksek SoC
    dSoC = 100 * np.cumsum(dI) * DT / 3600 / p["Q"]
    y["SoC"][n0:] = s["SoC"][n0:] + dSoC

    # 4) Batarya gerilimi — TÜRETME [Rint]: V = V₀(SoC) + R·I → ΔV = R·ΔI + b·ΔSoC
    #    Örnek: R = 0.12 ohm → 0.12 × 2.08 = +0.25 V (anlık) + SoC farkından gelen yavaş artış
    dV = p["R"] * dI + p["b"] * dSoC
    y["V_bat"][n0:] = s["V_bat"][n0:] + dV

    # 5) Isıtıcı gerilimi ve akımı — TÜRETME [Tez-B Eq 3.8]: I = P / V
    #    Isıtıcı batarya ile aynı hatta ise (veride kontrol edilir) onun gerilimi de ΔV kadar değişir.
    #    I' = P' / V'   (V' ≈ V olduğundan I' ≈ η·I;  örnek: 2250 W / 360 V = 6.25 A)
    if p["hv_ortak"]:
        y["V_isi"][n0:] = s["V_isi"][n0:] + dV
    V2 = y["V_isi"][n0:]
    y["I_isi"][n0:] = np.where(V2 > 1, y["P"][n0:] / np.where(V2 > 1, V2, 1), eta * s["I_isi"][n0:])

    # 6) Çıkış suyu sıcaklığı — TÜRETME [Tez-O Eq 4.26 + Tez-B Eq 3.25]:
    #    kararlı durumda su eksik / K kadar daha az ısınır; bu fark τ gecikmesiyle yerleşir
    #    T_çıkış' = T_çıkış − gecikme(eksik / K, τ)
    #    Örnek: 750 W / 165 W/°C = 4.5 °C daha soğuk (τ sonra farkın %63'ü, 3τ sonra %95'i oluşur)
    y["T_cikis"][n0:] = s["T_cikis"][n0:] - gecikme(eksik / p["K"], p["tau"])

    # İhmal edilen: batarya sıcaklığı. Akım azalınca batarya iç ısınması [Tez-B Eq 3.10, P = I²R] azalır:
    # 0.12 × (50² − 47.9²) ≈ 25 W. Bataryanın ısı kapasitesi (C) verimizde yok → değiştirilmez (sınırlılık).
    return y


# ---------------------------------------------------------------------------
# 4) SOĞUTMA SIVISI POMPASI DEBİ KAYBI
# ---------------------------------------------------------------------------

def pompa_debi_kaybi(s, n0, eta, p):
    # POMPA ARIZASI: pompa normalde basacağı debinin yalnızca η katını basar (aşınma, tıkanma, hava gibi).
    # KAYNAKTAN [Yue Eq 14]: eyleyici çarpımsal arızası y = η·u → ṁ' = η·ṁ,  η = 0.75 / 0.5 [Yue]
    # s: sinyaller (P: ısıtıcı gücü, T_giris / T_cikis: ısıtıcı giriş/çıkış suyu, debi: debi sütunu)
    # p: veriden tahmin edilen sabitler (K = ṁ·c_p, tau) ve debi sütununun anlamı (debi_olculen)
    #
    # NE DEĞİŞİR, NE DEĞİŞMEZ:
    #   Isıtıcı gücü P DEĞİŞMEZ: ısıtıcı aynı elektrik gücünü harcar, aynı ısıyı suya verir (enerjinin korunumu)
    #   → batarya akımı, gerilimi, SoC, ısıtıcı akımı/gerilimi de DEĞİŞMEZ
    #   Çıkış suyu sıcaklığı ARTAR: aynı ısı daha az suya verilir (aşağıda formül)
    #   Debi sütunu: anlamı belirsiz → p["debi_olculen"]'e göre (aşağıda)
    # SINIRLILIK (kapalı çevrim, modellenmez):
    #   - Isıtıcı kontrolcüsü fazla ısınan suya tepki verip gücü kısabilir (algoritması bilinmiyor)
    #   - Giriş suyu: devre kapalı olduğu için dönen su da zamanla değişir (ısı değiştirici modeli yok)
    #   - Kabin ısınması ve pompanın kendi elektrik tüketimi (veride yok)
    y = {k: v.astype(float).copy() for k, v in s.items()}
    P = np.clip(s["P"][n0:], 0, None)   # güç negatif olamaz (negatif değerler ölçüm gürültüsü)
    dT0 = s["T_cikis"][n0] - s["T_giris"][n0]   # arıza başladığı andaki ısınma (iki model de buradan başlar)

    # 1) Çıkış suyu sıcaklığı
    #    TÜRETME [Tez-O Eq 4.26]: Q = ṁ·c_p·ΔT → kararlı durumda ΔT = P / (ṁ·c_p) = P / K
    #      Debi η katına inince K' = η·K → ΔT' = P / (η·K) = ΔT / η
    #      Örnek: P = 1300 W, K = 591 W/°C → sağlam ΔT = 2.2 °C; η = 0.5 → ΔT' = 4.4 °C (2.2 °C daha sıcak)
    #    TÜRETME [Tez-B Eq 3.25 + Tez-O Eq 4.26]: gecikme τ = C / (ṁ·c_p). Sıvının ısı kapasitesi C aynı,
    #      ṁ·c_p η katına iniyor → τ' = τ / η  (debi yarıya inince su ısıtıcıda iki kat uzun kalır)
    #      Örnek: τ = 5 s, η = 0.5 → τ' = 10 s
    #    Sağlam ve arızalı ısınma, aynı modelle aynı başlangıç değerinden (dT0) hesaplanır;
    #    aradaki fark ölçülen çıkış sıcaklığına eklenir (ölçülen verinin doğal dalgalanması korunur):
    #      T_çıkış' = T_çıkış + [gecikme(P/(η·K), τ/η) − gecikme(P/K, τ)]
    saglam = gecikme(P / p["K"], p["tau"], dT0)
    arizali = gecikme(P / (eta * p["K"]), p["tau"] / eta, dT0)
    y["T_cikis"][n0:] = s["T_cikis"][n0:] + (arizali - saglam)

    # 2) Debi sütunu ("Coolant Volume Flow +500 [l/h]")
    #    Veri bulgusu (estimate_flow.py): gerçek debi = sütun + 500 l/h; enerji dengesiyle uyumlu (~508 l/h)
    #    BELİRSİZLİK: sütun ölçülen debi mi, kontrolcünün istediği (ayar) debi mi bilinmiyor.
    #      Ölçülen debiyse: gerçek debiyi gösterir → (sütun + 500)·η − 500
    #      Ayar değeriyse: komut değişmediği için aynı kalır
    #    BİZİM KARARIMIZ: varsayılan "değişmez" (debi_olculen = False). Gerekçe:
    #      - değer tüm sürüşlerde neredeyse sabit (500 ± 0.5 l/h) → ayar değeri olma ihtimali yüksek
    #      - yanlışlıkla değiştirirsek model "debi sütunu düştü" kısayolunu öğrenir (gerçekte olmayabilir);
    #        değiştirmezsek en kötü ihtimalle arıza gerçekte olduğundan ZOR bulunur (temkinli taraf)
    if p["debi_olculen"]:
        y["debi"][n0:] = (s["debi"][n0:] + 500) * eta - 500
    return y


# ---------------------------------------------------------------------------
# 5) BATARYA İÇ DİRENÇ ARTIŞI (bataryanın kendisinin yaşlanması)
# ---------------------------------------------------------------------------

def batarya_direnc_artisi(s, n0, k, p):
    # BATARYA ARIZASI: sensör değil bataryanın kendisi yaşlanmış → iç direnci k katına çıkmış: R' = k·R
    # Şiddet k = 1.33 ve 2:
    #   k = 2    KAYNAKTAN [Grandjean §1, Farmann'a atfen]: direnç iki katına çıkınca ömür sonu (hibrit araç ölçütü;
    #            i3 tam elektrikli olduğu için tezde "ağır yaşlanma senaryosu" olarak sunulacak)
    #   k = 1.33 BİZİM KARARIMIZ: Yue'nin η = 0.75 şiddeti dirence R' = R/η olarak uygulanmış hâli
    #            (η = 0.5 → R/0.5 = 2R, yani Grandjean'ın ömür sonu ile aynı nokta)
    # s: sinyaller (V_bat, I_bat, SoC, V_isi, I_isi, P_isi), p: veriden sabitler (R: o sürüşün direnci, b, Q, hv_ortak)
    #
    # NE DEĞİŞİR: batarya gerilimi, batarya akımı, SoC, (aynı hattaysa) ısıtıcı gerilimi ve akımı
    # NE DEĞİŞMEZ: ısıtıcı gücü (ısıtıcı istenen gücü vermeye devam eder), su sıcaklıkları, hız
    # SINIRLILIK:
    #   - Batarya sıcaklığı: ek ısı ΔR·I² [Tez-B Eq 3.10, Grandjean §1] gerçekte bataryayı ısıtır; bataryanın ısı
    #     kapasitesi C verimizde olmadığı için değiştirilmez
    #   - BMS'in güç sınırlaması (derating) modellenmez; istenen güç verilemiyorsa batarya en yüksek gücünde tutulur
    #   - Rint modeli şarj ve deşarjda aynı direnci kullanır (gerçekte şarjda %5–20 farklı olabilir [Grandjean §3.3.3])
    y = {kk: v.astype(float).copy() for kk, v in s.items()}
    R, R2 = p["R"], k * p["R"]
    V, I = s["V_bat"][n0:], s["I_bat"][n0:]

    # 1) Boştaki gerilim — KAYNAKTAN [Rint]: V = V₀ + R·I  →  V₀ = V − R·I  (R bu sürüşün kendi verisinden)
    V0 = V - R * I
    # 2) Bataryadan istenen güç DEĞİŞMEZ: motor ve ısıtıcı aynı gücü ister — TÜRETME [Tez-B Eq 3.8]: P = V·I
    P = V * I            # + şarj, − deşarj (veride + akım = şarj; R > 0 çıktığı için doğrulandı)

    def akim(V0):
        # TÜRETME [Rint + Tez-B Eq 3.8]: V' = V₀ + R'·I'  ve  V'·I' = P  →  R'·I'² + V₀·I' − P = 0
        #   I' = (−V₀ + √(V₀² + 4·R'·P)) / (2·R')   (P = 0 iken I' = 0 veren fiziksel kök)
        # Kök içi negatifse batarya o gücü veremez (yaşlı batarya gücü sınırlar [Grandjean §1]);
        # bu durumda verebileceği en yüksek güçteki akım alınır: I' = −V₀ / (2·R')
        ic = V0 ** 2 + 4 * R2 * P
        return (-V0 + np.sqrt(np.clip(ic, 0, None))) / (2 * R2), int(np.sum(ic < 0))

    # 3) Yeni akım ve SoC — iki adımda: önce SoC değişimi ihmal edilerek akım bulunur, sonra
    #    SoC farkının boştaki gerilime etkisi (b·ΔSoC) eklenip akım yeniden hesaplanır
    #    TÜRETME [Tez-B Eq 3.26]: ΔSoC = ∫ΔI dt / Q ;  [Rint]: V₀ SoC'ye bağlı → ΔV₀ = b·ΔSoC
    I2, _ = akim(V0)
    dSoC = 100 * np.cumsum(I2 - I) * DT / 3600 / p["Q"]
    I2, sinirli = akim(V0 + p["b"] * dSoC)
    dSoC = 100 * np.cumsum(I2 - I) * DT / 3600 / p["Q"]
    V2 = V0 + p["b"] * dSoC + R2 * I2
    # Örnek: R = 0.12 → R' = 0.24 ohm, I = −100 A, V = 360 V → V₀ = 372 V, P = −36 kW
    #        I' = (−372 + √(372² − 4·0.24·36000)) / 0.48 ≈ −103.7 A,  V' = 372 − 0.24·103.7 ≈ 347.1 V
    #        → gerilim ~12.9 V daha düşük, akım ~%3.7 daha büyük (aynı 36 kW için)
    y["V_bat"][n0:] = V2
    y["I_bat"][n0:] = I2
    y["SoC"][n0:] = s["SoC"][n0:] + dSoC

    # 4) Isıtıcı — aynı yüksek gerilim hattındaysa (veride kontrol edilir) gerilimi bataryayla birlikte değişir;
    #    ısıtıcı aynı gücü verdiği için akımı TÜRETME [Tez-B Eq 3.8]: I_ısıtıcı' = P_ısıtıcı / V_ısıtıcı'
    if p["hv_ortak"]:
        Vi = s["V_isi"][n0:] + (V2 - V)
        y["V_isi"][n0:] = Vi
        y["I_isi"][n0:] = np.where(Vi > 1, s["P_isi"][n0:] / np.where(Vi > 1, Vi, 1), s["I_isi"][n0:])
    return y, sinirli
