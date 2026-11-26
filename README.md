Brackets: 95 % confidence intervals from a trip-level bootstrap (2000 resamples). Feature set: raw statistics + physics residuals + 5-minute history.


Longer windows help slow faults (drift), while 30 s gives the shortest alarm delay under the alarm rule.

├── prepare_data.py            # cleaning: trims corrupted segments, short gaps, first 30 s
├── split_trips.py             # trip-level train/test split (28 / 7)
├── check_*.py, find_*.py,     # data exploration and quality checks
│   inspect_*.py, plot_*.py    #   (WMO-style step/persistence tests, flow offset, battery resistance …)
├── esik_kontrol.py            # sensitivity of the fault visibility threshold
├── ariza/                     # fault injection
│   ├── arizalar.py            #   fault formulas with sources (sensor + component faults)
│   ├── enjekte_et.py          #   injects one fault of each type into every trip
│   └── cozunurluk_kontrol.py  #   sensor resolution / visibility check
├── model/
│   ├── veri_hazirla.py        # 30 s windows (5 s step), features, trip-grouped folds
│   ├── ortak.py               # shared model, physics-constant and metric functions
│   ├── ml_modeller.py         # ML models (RF, XGBoost, SVM, Isolation Forest)
│   ├── hiperparametre_ayari.py      # equal-budget random search (ML)
│   ├── dl_ortak.py, dl_modeller.py  # DL models (CNN, GRU, LSTM, Autoencoder), resumable
│   ├── dl_hiperparametre_ayari.py   # equal-budget random search (DL)
│   ├── dl_colab.ipynb         # notebook to run DL scripts on Google Colab (GPU)
│   ├── alarm_kurali.py        # N-consecutive-window alarm rule, N selection
│   ├── son_test.py            # final test with bootstrap CIs and paired comparison
│   ├── pencere_duyarlilik.py  # window length sensitivity analysis
│   └── sinyal_onemi.py, zor_analiz.py, yanlis_alarm_analizi.py   # analyses
└── reports/                   # figures and result tables



Python 3.10+ and:

numpy pandas scipy scikit-learn xgboost matplotlib
torch            # only for the deep learning models
Run from the repository root, in this order:

python prepare_data.py              # → data/clean_B/
python split_trips.py               # → split.csv
python ariza/enjekte_et.py          # → data/faulty/
python model/veri_hazirla.py        # → data/model/
python model/ml_modeller.py         # ML models, 5-fold CV
python model/hiperparametre_ayari.py
python model/alarm_kurali.py
# Deep learning: run model/dl_colab.ipynb on Google Colab (GPU)
python model/son_test.py            # final test (after setting SON_TEST = True in the model scripts)

