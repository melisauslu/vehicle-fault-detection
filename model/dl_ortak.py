import copy
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight

from ortak import ANA_KLASOR, VERI, SEED, kat_sabitleri, fizik_ekle, if_kalibrasyon_bolmesi, IF_YANLIS_ALARM

# Derin öğrenme kodlarının (dl_modeller.py, dl_hiperparametre_ayari.py) ortak parçaları.
# ML ile AYNI kurallar: aynı katlar, kat sabitleri yalnızca eğitimden, normalizasyon yalnızca eğitimden,
# balanced sınıf ağırlıkları, test sürüşlerine dokunulmaz.

DL_EN_IYI_PARAMETRELER = os.path.join(ANA_KLASOR, "reports", "model", "dl_hiperparametre_en_iyi.json")
CIHAZ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BİZİM KARARIMIZ (eğitim düzeni, bütün DL modellerinde aynı):
#   - Erken durdurma: her katın EĞİTİM sürüşlerinin ~%15'i (sürüş bazında) iç doğrulamaya ayrılır; katın asıl
#     doğrulama sürüşleri eğitim sırasında HİÇ kullanılmaz (yalnızca sonda ölçmek için) → sızıntı yok.
#   - En fazla EPOCH tur; iç doğrulama kaybı SABIR tur boyunca iyileşmezse durur, en iyi turun ağırlıkları alınır.
IC_DOGRULAMA_ORANI = 0.15
EPOCH = 40
SABIR = 5
TAHMIN_PAKETI = 4096

# Varsayılan ayarlar (deneme 0) — yaygın başlangıç değerleri (BİZİM KARARIMIZ)
VARSAYILAN = {
    "CNN": {"n_katman": 3, "filtre": 64, "cekirdek": 5, "dropout": 0.2, "lr": 1e-3, "paket": 128},
    "GRU": {"gizli": 64, "katman": 2, "cift_yonlu": False, "dropout": 0.2, "lr": 1e-3, "paket": 128},
    "LSTM": {"gizli": 64, "katman": 2, "cift_yonlu": False, "dropout": 0.2, "lr": 1e-3, "paket": 128},
    "Autoencoder": {"filtre": 32, "gizil": 8, "dropout": 0.0, "lr": 1e-3, "paket": 128},
}


def tohum(s):
    np.random.seed(s)
    torch.manual_seed(s)


def ham_dizileri_oku():
    # veri_hazirla.py'nin ürettiği diziler; satır sırası pencereler_ozellik.csv ile aynı
    d = np.load(os.path.join(VERI, "pencereler_ham.npz"))
    return d["X_ham"], d["X_fizik"]


def en_iyi_parametreleri_oku():
    with open(DL_EN_IYI_PARAMETRELER, encoding="utf-8") as f:
        return json.load(f)


def atomik_csv(df, yol):
    # Önce geçici dosyaya yaz, sonra adını değiştir → yazarken kesilirse yarım dosya kalmaz (kaldığı yerden devam)
    gecici = yol + ".tmp"
    df.to_csv(gecici, sep=";", index=False)
    os.replace(gecici, yol)


# ============================================================================
# KAT VERİSİ
# ============================================================================
def kat_verisi(oz, X_ham, X_fizik, bilgi, egitim, dogrulama, set_adi, siniflar, tohum_degeri):
    # set "ham": yalnızca 13 ham sinyal dizisi
    # set "ham+fizik+gecmis": 13 ham + 5 fizik kanalı dizisi + 5 uzun geçmiş özelliği (ayrı girdi; soc_farki_5dk
    #   o katın Q'suyla — ML'deki ana setin DL karşılığı)
    egitim, dogrulama = np.asarray(egitim), np.asarray(dogrulama)
    o = fizik_ekle(oz, kat_sabitleri(oz, pd.Series(egitim, index=oz.index), bilgi))
    dizi = X_ham if set_adi == "ham" else np.concatenate([X_ham, X_fizik], axis=2)
    skaler = None if set_adi == "ham" else o[bilgi["ozellik_gruplari"]["uzun_gecmis"]].to_numpy(np.float32)

    # iç doğrulama: eğitim sürüşlerinin ~%15'i (sürüş bazında, tohuma bağlı)
    surus = o["surus"].to_numpy()
    egitim_surusleri = np.array(sorted(set(surus[egitim])))
    rng = np.random.default_rng(tohum_degeri)
    ic = rng.choice(egitim_surusleri, size=max(1, round(len(egitim_surusleri) * IC_DOGRULAMA_ORANI)), replace=False)
    ic_dog = egitim & np.isin(surus, ic)
    ogr = egitim & ~ic_dog

    # normalizasyon: kanal başına ortalama/std YALNIZCA modelin eğitildiği pencerelerden (sızıntı kuralı 2)
    ort = dizi[ogr].reshape(-1, dizi.shape[2]).mean(0)
    std = dizi[ogr].reshape(-1, dizi.shape[2]).std(0) + 1e-6
    dizi_n = np.nan_to_num((dizi - ort) / std).astype(np.float32)
    if skaler is not None:
        s_ort, s_std = skaler[ogr].mean(0), skaler[ogr].std(0) + 1e-6
        skaler = np.nan_to_num((skaler - s_ort) / s_std).astype(np.float32)

    y = np.searchsorted(siniflar, o["ariza_tipi"].to_numpy())
    saglam = o["ariza"].to_numpy() == 0
    # Autoencoder (Isolation Forest ile aynı düzen): sağlam eğitim pencereleri → öğrenme + eşik kalibrasyonu
    if_ogren, if_kalib = (np.asarray(m) for m in if_kalibrasyon_bolmesi(o, pd.Series(egitim, index=oz.index)))
    return {"dizi": dizi_n, "skaler": skaler, "y": y, "ogr": ogr, "ic_dog": ic_dog, "dog": dogrulama,
            "ae_ogr": if_ogren & ~ic_dog, "ae_ic_dog": if_ogren & ic_dog, "ae_kalib": if_kalib, "saglam": saglam}


# ============================================================================
# MODELLER
# ============================================================================
class CNN(nn.Module):
    # 1D evrişim: zaman ekseninde kayan filtreler yerel örüntüleri (sıçrama, düzleşme, gürültü) yakalar;
    # sonra zaman boyunca ortalama alınır (küresel ortalama havuzlama) ve sınıflandırılır
    def __init__(self, kanal, skaler, sinif, n_katman, filtre, cekirdek, dropout):
        super().__init__()
        k, c = [], kanal
        for _ in range(n_katman):
            k += [nn.Conv1d(c, filtre, cekirdek, padding=cekirdek // 2), nn.BatchNorm1d(filtre), nn.ReLU(),
                  nn.Dropout(dropout)]
            c = filtre
        self.govde = nn.Sequential(*k)
        self.kafa = nn.Sequential(nn.Linear(filtre + skaler, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, sinif))

    def forward(self, x, s):
        h = self.govde(x.transpose(1, 2)).mean(2)
        return self.kafa(h if s is None else torch.cat([h, s], 1))


class RNN(nn.Module):
    # GRU / LSTM: diziyi adım adım okur, önemli bilgiyi gizli durumda taşır; son durum sınıflandırılır.
    # Çift yönlü: pencere 30 s'nin tamamı karar anında elde olduğundan ileri+geri okumak geleceği kullanmak değildir.
    def __init__(self, tur, kanal, skaler, sinif, gizli, katman, cift_yonlu, dropout):
        super().__init__()
        r = nn.GRU if tur == "GRU" else nn.LSTM
        self.rnn = r(kanal, gizli, katman, batch_first=True, dropout=dropout if katman > 1 else 0.0,
                     bidirectional=cift_yonlu)
        self.gizli, self.cift = gizli, cift_yonlu
        d = gizli * (2 if cift_yonlu else 1)
        self.kafa = nn.Sequential(nn.Dropout(dropout), nn.Linear(d + skaler, 64), nn.ReLU(), nn.Linear(64, sinif))

    def forward(self, x, s):
        o, _ = self.rnn(x)
        h = torch.cat([o[:, -1, :self.gizli], o[:, 0, self.gizli:]], 1) if self.cift else o[:, -1]
        return self.kafa(h if s is None else torch.cat([h, s], 1))


class AE(nn.Module):
    # Evrişimli autoencoder: 60 adımı 15 adıma sıkıştırıp (gizil temsil) geri üretir. Yalnızca SAĞLAM veriyle
    # eğitilir; arızalı pencereler iyi üretilemez → yeniden üretme hatası (anormallik skoru) büyür.
    def __init__(self, kanal, filtre, gizil, dropout):
        super().__init__()
        self.kodlayici = nn.Sequential(nn.Conv1d(kanal, filtre, 5, stride=2, padding=2), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Conv1d(filtre, gizil, 5, stride=2, padding=2), nn.ReLU())        # 60→30→15
        self.cozucu = nn.Sequential(nn.ConvTranspose1d(gizil, filtre, 4, stride=2, padding=1), nn.ReLU(),
                                    nn.ConvTranspose1d(filtre, kanal, 4, stride=2, padding=1))           # 15→30→60

    def forward(self, x):
        return self.cozucu(self.kodlayici(x.transpose(1, 2))).transpose(1, 2)


def model_kur(ad, kanal, skaler, sinif, p):
    if ad == "CNN":
        return CNN(kanal, skaler, sinif, p["n_katman"], p["filtre"], p["cekirdek"], p["dropout"])
    if ad in ("GRU", "LSTM"):
        return RNN(ad, kanal, skaler, sinif, p["gizli"], p["katman"], p["cift_yonlu"], p["dropout"])
    if ad == "Autoencoder":
        return AE(kanal + skaler, p["filtre"], p["gizil"], p["dropout"])   # uzun geçmiş özellikleri de kanal olarak
    raise ValueError(ad)


# ============================================================================
# EĞİTİM
# ============================================================================
def _tensor(a):
    return None if a is None else torch.as_tensor(a, device=CIHAZ)


def _ae_girdi(k, maske):
    x = k["dizi"][maske]
    if k["skaler"] is not None:                 # uzun geçmiş özellikleri zaman boyunca tekrar eden kanallar olarak
        x = np.concatenate([x, np.repeat(k["skaler"][maske][:, None, :], x.shape[1], axis=1)], axis=2)
    return x


def _paketler(n, paket, karistir, rng):
    sira = rng.permutation(n) if karistir else np.arange(n)
    return [sira[i:i + paket] for i in range(0, n, paket)]


def egit_ve_tahmin_et(ad, k, p, siniflar, tohum_degeri):
    # Bir modeli bir katta eğitir (erken durdurma iç doğrulamayla), katın doğrulama pencerelerini değerlendirir.
    # Denetimli → (tahmin edilen sınıf adları, None, bilgi); Autoencoder → (0/1 alarm, anormallik skoru, bilgi)
    tohum(tohum_degeri)
    rng = np.random.default_rng(tohum_degeri)
    ae = ad == "Autoencoder"
    S = 0 if k["skaler"] is None else k["skaler"].shape[1]
    if ae:
        X_tr, X_ic = _tensor(_ae_girdi(k, k["ae_ogr"])), _tensor(_ae_girdi(k, k["ae_ic_dog"]))
        model = model_kur(ad, k["dizi"].shape[2], S, len(siniflar), p).to(CIHAZ)
        kayip_f = nn.MSELoss()
    else:
        X_tr, S_tr, y_tr = _tensor(k["dizi"][k["ogr"]]), _tensor(None if not S else k["skaler"][k["ogr"]]), k["y"][k["ogr"]]
        X_ic, S_ic = _tensor(k["dizi"][k["ic_dog"]]), _tensor(None if not S else k["skaler"][k["ic_dog"]])
        y_ic = torch.as_tensor(k["y"][k["ic_dog"]], device=CIHAZ)
        # balanced sınıf ağırlıkları (ML ile aynı formül); eğitimde olmayan sınıfın ağırlığı 0
        var = np.unique(y_tr)
        agirlik = np.zeros(len(siniflar), dtype=np.float32)
        agirlik[var] = compute_class_weight("balanced", classes=var, y=y_tr)
        kayip_f = nn.CrossEntropyLoss(weight=torch.as_tensor(agirlik, device=CIHAZ))
        y_tr = torch.as_tensor(y_tr, device=CIHAZ)
        model = model_kur(ad, k["dizi"].shape[2], S, len(siniflar), p).to(CIHAZ)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])

    def ic_kayip():
        model.eval()
        top, n = 0.0, 0
        with torch.no_grad():
            for i in _paketler(len(X_ic), TAHMIN_PAKETI, False, rng):
                if ae:
                    top += kayip_f(model(X_ic[i]), X_ic[i]).item() * len(i)
                else:
                    top += kayip_f(model(X_ic[i], None if S_ic is None else S_ic[i]), y_ic[i]).item() * len(i)
                n += len(i)
        return top / max(n, 1)

    t0 = time.time()
    en_iyi, en_iyi_durum, bekle, tur = np.inf, None, 0, 0
    for tur in range(1, EPOCH + 1):
        model.train()
        for i in _paketler(len(X_tr), p["paket"], True, rng):
            opt.zero_grad()
            if ae:
                kayip = kayip_f(model(X_tr[i]), X_tr[i])
            else:
                kayip = kayip_f(model(X_tr[i], None if S_tr is None else S_tr[i]), y_tr[i])
            kayip.backward()
            opt.step()
        v = ic_kayip()
        if v < en_iyi - 1e-6:
            en_iyi, en_iyi_durum, bekle = v, copy.deepcopy(model.state_dict()), 0
        else:
            bekle += 1
            if bekle >= SABIR:
                break
    model.load_state_dict(en_iyi_durum)
    egitim_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        if ae:
            def skorla(x):
                x = _tensor(x)
                return np.concatenate([((model(x[i]) - x[i]) ** 2).mean((1, 2)).cpu().numpy()
                                       for i in _paketler(len(x), TAHMIN_PAKETI, False, rng)])
            esik = np.quantile(skorla(_ae_girdi(k, k["ae_kalib"])), 1 - IF_YANLIS_ALARM)   # IF ile aynı eşik kuralı
            t0 = time.time()
            skor = skorla(_ae_girdi(k, k["dog"]))
            tahmin = np.where(skor > esik, "ariza", "saglam")
        else:
            X_va = _tensor(k["dizi"][k["dog"]])
            S_va = _tensor(None if not S else k["skaler"][k["dog"]])
            t0 = time.time()
            cikti = np.concatenate([model(X_va[i], None if S_va is None else S_va[i]).argmax(1).cpu().numpy()
                                    for i in _paketler(len(X_va), TAHMIN_PAKETI, False, rng)])
            tahmin, skor = np.asarray(siniflar)[cikti], None
    tahmin_ms = (time.time() - t0) / max(k["dog"].sum(), 1) * 1000
    return tahmin, skor, {"egitim_s": egitim_s, "tahmin_ms_pencere": tahmin_ms, "epoch": tur,
                          "ic_dogrulama_kaybi": en_iyi, "cihaz": str(CIHAZ)}
