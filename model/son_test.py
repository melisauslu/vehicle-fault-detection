import json
import os
import sys
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import ANA_KLASOR, SEED, veriyi_oku

RAPOR = os.path.join(ANA_KLASOR, "reports", "model")
TAHMIN = os.path.join(ANA_KLASOR, "data", "model")

# SON TEST DEĞERLENDİRMESİ — 7 test sürüşü, BİR KEZ.
# Önce: ml_modeller.py (PyCharm) ve dl_modeller.py (Colab) AYARLI = True, SON_TEST = True ile çalışmış olmalı.
# Hiçbir şey SEÇİLMEZ: modeller, ayarlar ve alarm kuralının N'i çapraz doğrulamada seçilmişti
#   (alarm_kurali_secim_ayarli.json) → burada AYNEN uygulanır. Bu dosyanın sonucuna bakıp geri dönüp bir şey
#   değiştirmek testi geçersiz kılar.
# Güven aralığı: bootstrap [Efron & Tibshirani 1993 — ÖNERİ, kontrol et]. BİZİM KARARIMIZ: örnekleme SÜRÜŞ
#   bazında yapılır (bir sürüşün sağlam ve arızalı bütün kopyaları birlikte seçilir), çünkü aynı sürüşün
#   pencereleri/dosyaları birbirinden bağımsız değil. Yalnızca 7 sürüş olduğu için aralıklar GENİŞ çıkar —
#   bu bir hata değil, test setinin küçüklüğünü dürüstçe gösterir.

# ============================================================================
# AYARLAR
# ============================================================================
AYARLI = True
DL_SURUM = "_merkezli"
SET = "ham+fizik+gecmis"
TEKRAR = 2000                      # bootstrap tekrar sayısı
DENETIMSIZ = {"IsolationForest", "Autoencoder"}

# ============================================================================
oz, bilgi = veriyi_oku()
adim_s = bilgi["adim_satir"] * 0.1
ek = "_ayarli" if AYARLI else ""
parcalar = []
for yol in (os.path.join(TAHMIN, f"ml_tahminler{ek}_son_test.csv"),
            os.path.join(TAHMIN, f"dl_tahminler{DL_SURUM}{ek}_son_test.csv")):
    if os.path.exists(yol):
        parcalar.append(pd.read_csv(yol, sep=";", low_memory=False))
        print(f"Okundu: {os.path.basename(yol)}")
    else:
        print(f"!!! {os.path.basename(yol)} bulunamadı → bu modeller yok")
if not parcalar:
    sys.exit("Son test tahmini yok — önce ml_modeller.py / dl_modeller.py SON_TEST = True ile çalışmalı.")
t = pd.concat(parcalar, ignore_index=True)
t = t[t["ozellik_seti"] == SET].sort_values(["model", "dosya", "pencere_bitis_s"])
assert set(t["bolme"]) == {"son_test"}, "Bu dosyada son test dışı tahminler var — yanlış dosya okunuyor"
t = t.merge(oz[["dosya", "surus"]].drop_duplicates(), on="dosya", how="left")
surusler = sorted(t["surus"].unique())

with open(os.path.join(RAPOR, f"alarm_kurali_secim{ek}.json"), encoding="utf-8") as f:
    N_secim = json.load(f)["N"]                        # çapraz doğrulamada seçilen N'ler (değiştirilmez)
ml_dl_sirasi = ["XGBoost", "RandomForest", "SVM", "IsolationForest", "CNN", "GRU", "LSTM", "Autoencoder"]
modeller = [m for m in ml_dl_sirasi if m in set(t["model"])] + sorted(set(t["model"]) - set(ml_dl_sirasi))
siniflar = sorted(oz["ariza_tipi"].unique())
print(f"Test sürüşleri ({len(surusler)}): {', '.join(surusler)}")
print(f"Alarm kuralı N (çapraz doğrulamadan): {json.dumps(N_secim)}\n")


def ardisik_say(a):
    # alarm_kurali.py ile aynı: her pencerede o ana kadar kesintisiz kaç pencere "arıza" dendi
    c = np.zeros(len(a), dtype=int)
    for i, v in enumerate(a):
        c[i] = (c[i - 1] + 1 if i else 1) if v else 0
    return c


def macro_f1(cm):
    # karışıklık matrisinden macro-F1 (sklearn f1_score(average="macro", zero_division=0) ile aynı)
    tp = np.diag(cm).astype(float)
    pay, payda = 2 * tp, cm.sum(0) + cm.sum(1)
    return float(np.mean(np.divide(pay, payda, out=np.zeros_like(pay), where=payda > 0)))


def tespit_f1(cm, s):
    # "arıza var mı" düzeyinde F1; s = "saglam" sınıfının sırası
    tp = cm.sum() - cm[s, :].sum() - cm[:, s].sum() + cm[s, s]   # gerçek arıza, tahmin arıza
    fp, fn = cm[s, :].sum() - cm[s, s], cm[:, s].sum() - cm[s, s]
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


# ---- her model için sürüş bazında parçalar (bootstrap bunları yeniden örnekler) ----
etiket = siniflar + ["ariza"]                          # Autoencoder/IF yalnızca "ariza"/"saglam" der
s_idx = etiket.index("saglam")
veri = {}
for model in modeller:
    m = t[t["model"] == model]
    N = N_secim.get(model)
    parca = {}
    for surus in surusler:
        x = m[m["surus"] == surus]
        g = pd.Categorical(x["ariza_tipi"], categories=etiket).codes
        p = pd.Categorical(x["tahmin"], categories=etiket).codes
        cm = np.bincount(g * len(etiket) + p, minlength=len(etiket) ** 2).reshape(len(etiket), -1)
        dosyalar = []
        for dosya, d in x.groupby("dosya", sort=False):
            tahmin, tip = d["tahmin"].to_numpy(), d["ariza_tipi"].iloc[0]
            c = ardisik_say(tahmin != "saglam")
            r = {"saglam": tip == "saglam", "tip": tip, "saat": len(c) * adim_s / 3600,
                 "olay_N1": int(np.sum(c == 1))}
            if N:
                i = np.flatnonzero(c >= N)
                r.update(olay=int(np.sum(c == N)), tespit=len(i) > 0,
                         gecikme=d["ariza_suresi_s"].to_numpy()[i[0]] if len(i) else np.nan,
                         teshis=(Counter(tahmin[i[0] - N + 1:i[0] + 1]).most_common(1)[0][0] == tip) if len(i) else False)
            dosyalar.append(r)
        parca[surus] = {"cm": cm, "dosyalar": dosyalar}
    veri[model] = parca


def olcutler(model, secilen):
    # secilen: sürüş listesi (bootstrap'te tekrarlı olabilir)
    N = N_secim.get(model)
    cm = sum(veri[model][s]["cm"] for s in secilen)
    f = [r for s in secilen for r in veri[model][s]["dosyalar"]]
    sag, ari = [r for r in f if r["saglam"]], [r for r in f if not r["saglam"]]
    saat = sum(r["saat"] for r in sag)
    o = {"macro_f1": np.nan if model in DENETIMSIZ else macro_f1(cm[:len(siniflar), :len(siniflar)]),
         "tespit_f1_pencere": tespit_f1(cm, s_idx),
         "yanlis_alarm_olay_saatte_N1": sum(r["olay_N1"] for r in sag) / saat if saat else np.nan}
    if N:
        tespit = [r for r in ari if r["tespit"]]
        o.update({"yanlis_alarm_olay_saatte": sum(r["olay"] for r in sag) / saat if saat else np.nan,
                  "tespit_edilen_dosya_orani": len(tespit) / len(ari) if ari else np.nan,
                  "tespit_gecikmesi_ortanca_s": float(np.median([r["gecikme"] for r in tespit])) if tespit else np.nan,
                  "alarm_aninda_dogru_teshis": np.nan if model in DENETIMSIZ else
                  (np.mean([r["teshis"] for r in tespit]) if tespit else np.nan)})
    return o


rng = np.random.default_rng(SEED)
orneklemler = [rng.choice(surusler, size=len(surusler), replace=True) for _ in range(TEKRAR)]
satirlar = []
for model in modeller:
    nokta = olcutler(model, surusler)
    boot = pd.DataFrame([olcutler(model, s) for s in orneklemler])
    r = {"model": model, "N": N_secim.get(model)}
    for k, v in nokta.items():
        r[k] = v
        r[k + "_alt"], r[k + "_ust"] = (np.nanpercentile(boot[k], [2.5, 97.5]) if boot[k].notna().any()
                                        else (np.nan, np.nan))
    satirlar.append(r)
s = pd.DataFrame(satirlar)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def aralik(r, k, yuzde=False, ondalik=3):
    if pd.isna(r.get(k)):
        return "–"
    c = 100 if yuzde else 1
    o = 0 if yuzde else ondalik
    return f"{r[k] * c:.{o}f} [{r[k + '_alt'] * c:.{o}f}–{r[k + '_ust'] * c:.{o}f}]"


print(f"=== SON TEST ({len(surusler)} test sürüşü, {SET}) — değer [%95 güven aralığı, sürüş bazında bootstrap] ===")
tablo = pd.DataFrame([{
    "model": r["model"], "N": "–" if pd.isna(r["N"]) else int(r["N"]),
    "macro-F1": aralik(r, "macro_f1"),
    "tespit-F1 (pencere)": aralik(r, "tespit_f1_pencere"),
    "yanlış alarm/saat (N=1)": aralik(r, "yanlis_alarm_olay_saatte_N1", ondalik=1),
    "yanlış alarm/saat (N)": aralik(r, "yanlis_alarm_olay_saatte", ondalik=2),
    "yakalanan dosya %": aralik(r, "tespit_edilen_dosya_orani", yuzde=True),
    "gecikme s": aralik(r, "tespit_gecikmesi_ortanca_s", ondalik=0),
    "doğru teşhis %": aralik(r, "alarm_aninda_dogru_teshis", yuzde=True),
} for r in satirlar])
print(tablo.to_string(index=False))

print("\n=== Seçilen N'de arıza tipine göre (yalnızca nokta değer; tip başına 7 dosya → aralık anlamsız) ===")
for model in modeller:
    if not N_secim.get(model):
        continue
    f = pd.DataFrame([r for sv in veri[model].values() for r in sv["dosyalar"] if not r["saglam"]])
    g = f.groupby("tip").agg(dosya=("tespit", "size"), tespit=("tespit", "mean"),
                             gecikme_s=("gecikme", "median"), dogru_teshis=("teshis", "mean"))
    if model in DENETIMSIZ:
        g = g.drop(columns="dogru_teshis")
    print(f"--- {model} (N = {N_secim[model]}) ---")
    print(g.round(2).to_string(), "\n")

s.to_csv(os.path.join(RAPOR, f"son_test_sonuclar{ek}.csv"), sep=";", index=False)
tablo.to_csv(os.path.join(RAPOR, f"son_test_tablo{ek}.csv"), sep=";", index=False)
print(f"Kaydedildi: reports/model/son_test_sonuclar{ek}.csv, son_test_tablo{ek}.csv")
