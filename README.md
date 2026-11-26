# Data-Driven Fault Detection and Diagnosis for an Electric Vehicle (BMW i3)

Machine learning and deep learning models that detect **whether** a fault is present and diagnose **which** fault it is, using real driving data from a BMW i3 electric vehicle. Faults are injected into healthy real-world winter driving data using formulas from the literature, and eight models are compared under the same, leakage-free evaluation protocol.

> Master's project (Project Design course). Code comments and variable names are in Turkish.

---

## Highlights

- **Real driving data:** 35 winter trips (~14.7 h, 10 Hz) of a BMW i3, 13 signals (battery, heating circuit, vehicle).
- **9 fault types** injected with literature-based formulas: 6 sensor faults (drift, bias, gain, precision degradation, stuck, complete failure) and 3 component faults (heater power loss, pump flow loss, battery internal resistance increase).
- **Physics-informed features:** residuals such as *requested − actual heater power* or *measured − current-integrated SoC change*, which are close to zero for a healthy vehicle.
- **8 models, equal tuning budget:** XGBoost, Random Forest, SVM, Isolation Forest, 1D-CNN, GRU, LSTM, convolutional Autoencoder (20 random-search trials each).
- **Leakage-free evaluation:** trip-grouped 5-fold cross-validation on 28 trips; 7 held-out test trips used **once** at the very end.
- **Alarm rule:** an alarm is raised only after *N* consecutive fault decisions; *N* is selected on cross-validation with a pre-defined criterion.

## Main result (7 held-out test trips)

| Model | Macro-F1 (which fault?) | Detection-F1 (fault or not?) | False alarms / h (with alarm rule) | Correct diagnosis at alarm |
|---|---|---|---|---|
| **XGBoost** | **0.891** [0.839–0.934] | 0.944 | **0.00** | **96 %** |
| Random Forest | 0.857 [0.772–0.922] | 0.956 | 0.75 | 94 % |
| SVM | 0.849 [0.788–0.871] | 0.932 | 0.37 | 96 % |
| 1D-CNN | 0.769 [0.716–0.814] | 0.895 | 4.10 | 83 % |
| LSTM | 0.748 [0.697–0.780] | 0.898 | 3.73 | 81 % |
| GRU | 0.725 [0.655–0.763] | 0.893 | 1.86 | 79 % |
| Autoencoder (unsupervised) | – (AUC 0.779) | 0.582 | – | – |
| Isolation Forest (unsupervised) | – (AUC 0.652) | 0.061 | – | – |

Brackets: 95 % confidence intervals from a trip-level bootstrap (2000 resamples). Feature set: raw statistics + physics residuals + 5-minute history.

**Findings**

- XGBoost is the best model; in a paired bootstrap comparison its macro-F1 is consistently higher than all other models.
- On this dataset, tree-based models outperform deep learning models in diagnosis and false alarms, while all supervised models detect a similar share of faulty trips (~90 %).
- Physics-informed features improve every model (XGBoost test macro-F1: 0.845 raw only → 0.891).
- Slowly developing faults (drift) are the hardest: they are physically invisible at onset.

## Window length sensitivity (cross-validation only)

| Window | 10 s | 20 s | **30 s (used)** | 45 s | 60 s |
|---|---|---|---|---|---|
| Macro-F1 | 0.823 | 0.839 | **0.854** | 0.874 | 0.876 |
| Drift F1 | 0.23 | 0.32 | 0.44 | 0.57 | 0.60 |
| Median alarm delay | 74 s | 80 s | **66 s** | 71 s | 77 s |

Longer windows help slow faults (drift), while 30 s gives the shortest alarm delay under the alarm rule.

---

## Repository structure

```
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
```

## Getting started

### 1. Data
The dataset is **not included** in this repository. Download it from IEEE DataPort (free IEEE account required):

> M. Steinstraeter, J. Buberger, D. Trifonov, *Battery and Heating Data in Real Driving Cycles*, IEEE DataPort. https://doi.org/10.21227/6jr9-5235

Place the winter trips (category B) under `data/B_winter/`.

### 2. Requirements
Python 3.10+ and:

```
numpy pandas scipy scikit-learn xgboost matplotlib
torch            # only for the deep learning models
```

### 3. Run the pipeline
Run from the repository root, in this order:

```bash
python prepare_data.py              # → data/clean_B/
python split_trips.py               # → split.csv
python ariza/enjekte_et.py          # → data/faulty/
python model/veri_hazirla.py        # → data/model/
python model/ml_modeller.py         # ML models, 5-fold CV
python model/hiperparametre_ayari.py
python model/alarm_kurali.py
# Deep learning: run model/dl_colab.ipynb on Google Colab (GPU)
python model/son_test.py            # final test (after setting SON_TEST = True in the model scripts)
```

Settings are at the top of each script (e.g. `AYARLI = True` for tuned models, `SON_TEST = True` for the final test). The final test is meant to be run **once**, after all decisions are fixed.

---

## Methodology notes

- **Fault injection:** faults start at a random time between 25 % and 75 % of the trip and stay until the end. Sensor fault severity uses three levels (L = 0.05, 0.2, 0.5). Physical constants (internal resistance, capacity, heating coefficient) are estimated from the vehicle's own data, not taken from textbooks.
- **Leakage prevention:** all trip windows stay on one side of every split; physics constants, normalization and thresholds are learned only from training trips of each fold.
- **Circularity:** two physics features use the same equations as the fault injection; they are excluded from the main feature set and reported only as an upper bound.
- **Unobservable faults:** faults that change less than 5 % of the faulty samples (all on the 1 °C-resolution battery temperature sensor) and stuck faults of that sensor are excluded.

## Limitations

- Faults are synthetic (literature-based), not recorded real faults.
- Small test set (7 trips); confidence intervals are wide.
- Only winter data; one vehicle.
- Window length (30 s) and several thresholds were chosen in this study; see the sensitivity analysis above.

## Main references

- Wang et al., *Processes* 13, 929 (2025) — sensor fault models
- Yue et al. (2019) — multiplicative actuator faults (heater, pump)
- Grandjean et al., *Batteries* 4(4), 49 (2018) — internal resistance and end of life
- Chen & Guestrin, *XGBoost*, KDD (2016)
- Breiman, *Random Forests*, Machine Learning 45 (2001)
- Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 13 (2012)
- Kim et al., *Reversible Instance Normalization (RevIN)*, ICLR (2022)
- Ismail Fawaz et al., *Deep learning for time series classification: a review*, DMKD (2019)
- Prechelt, *Early Stopping — But When?* (1998)
- Zahumenský, *Guidelines on Quality Control Procedures for Data from Automatic Weather Stations*, WMO (2004)

## Author

Melisa Uslu — Master's student, Computer Engineering
