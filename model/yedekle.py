import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import ANA_KLASOR

# ml_modeller.py'nin ürettiği sonuçları ayrı bir klasöre kopyalar (bir sonraki çalıştırma üzerine yazmasın diye).
# Kullanım: ETIKET'i değiştir, çalıştır. Asıl dosyalar silinmez, yalnızca kopyalanır.
ETIKET = "varsayilan"        # ör. "baslangic", "yeni_ozellikler", "ayarli"

hedef = os.path.join(ANA_KLASOR, "reports", "model", ETIKET)
if os.path.exists(hedef):
    sys.exit(f"'{hedef}' zaten var — üzerine yazmamak için durdum. ETIKET'i değiştir.")
os.makedirs(hedef)

dosyalar = (glob.glob(os.path.join(ANA_KLASOR, "reports", "model", "ml_*.csv"))
            + glob.glob(os.path.join(ANA_KLASOR, "reports", "model", "karisiklik_*.png"))
            + glob.glob(os.path.join(ANA_KLASOR, "data", "model", "ml_tahminler*.csv")))
for f in dosyalar:
    shutil.copy2(f, hedef)
    print("kopyalandı:", os.path.basename(f))
print(f"\n{len(dosyalar)} dosya → {hedef}")