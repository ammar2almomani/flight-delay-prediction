<div align="center">

# ✈️ Flight Delay Prediction

### Stacking Ensemble + Aircraft-Level Features under Strict Temporal Validation

*A dual-task ML framework that predicts **whether** a U.S. flight will arrive 15+ minutes late **and by how many minutes**, evaluated on two completely unseen months.*


| 🎯 AUC (Nov holdout) | 🎄 AUC (Dec stress test) | 📉 Regression RMSE (Dec) | 🧮 Rows processed |
|:---:|:---:|:---:|:---:|
| **0.9295** | **0.9254** | **15.30 min** | **1.72 M** (+ 582 K holdout) |

</div>

---

## 📌 Table of Contents
1. [Overview](#-overview)
2. [Key Contributions](#-key-contributions)
3. [Results at a Glance](#-results-at-a-glance)
4. [Methodology](#-methodology)
5. [Repository Structure](#-repository-structure)
6. [Getting Started](#-getting-started)
7. [Reproducing the Results](#-reproducing-the-results)
8. [Explainability (SHAP)](#-explainability-shap)
9. [Limitations & Future Work](#-limitations--future-work)
10. [Paper & Citation](#-paper--citation)
11. [Authors](#-authors)
12. [License & Acknowledgements](#-license--acknowledgements)

---

## 🔎 Overview

Flight delays cost the U.S. aviation industry roughly **$33 billion a year** (FAA). Airlines, airports and passengers all benefit from an early warning, but most published work has two weaknesses:

1. **Random train/test splits** that leak seasonal information from the future and inflate scores.
2. **Classification only** — it says *if* a flight is late, not *how late*.

This project fixes both. It builds one shared set of **26 engineered features** from U.S. Bureau of Transportation Statistics (BTS) On-Time Performance data and uses them for two tasks:

| Task | Target | Output |
|---|---|---|
| **Classification** | `IS_DELAYED` = arrival delay **> 15 min** | Probability + yes/no alert |
| **Regression** | `ARR_DELAY` | Expected delay in minutes |

Everything is trained on **January + June 2025** and tested on **November 2025** and **December 2025** — months the models never saw. December is a deliberate *holiday stress test* with a delay rate of 25.5 %, versus 22.6 % in training and 19.4 % in November.

---

## 🏆 Key Contributions

- **Dual-task framework** — one feature pipeline, two prediction problems (alert + magnitude).
- **Strict temporal validation** — train on Jan/Jun, test on Nov, stress-test on Dec. Random splits are shown to inflate AUC by **0.016 – 0.032**.
- **Novel `TAIL_DELAY_RATE` feature** — the historical delay rate of each individual aircraft (by tail number), computed on training data only. SHAP ranks it **#3 of 26** features on unseen data.
- **Five-model stacking ensemble** — Logistic Regression, Random Forest, LightGBM, XGBoost and TabNet feed a Logistic Regression meta-learner.
- **Leakage-safe by design** — post-flight delay-cause columns are dropped, and all target-encoding maps are fitted on train only, saved to disk, and re-applied to later months.
- **Honest statistics** — McNemar's tests, a SMOTE-vs-cost-sensitive ablation, and SHAP explanations.

---

## 📊 Results at a Glance

### Classification — delayed (> 15 min) vs. on time

| Model | Nov AUC | Nov F1 | Dec AUC | Dec F1 |
|---|:---:|:---:|:---:|:---:|
| **Stacking Ensemble** | **0.9295** | 0.8070 | 0.9254 | 0.8071 |
| Logistic Regression | 0.9290 | 0.8097 | **0.9258** | **0.8111** |
| Random Forest | 0.9197 | 0.7984 | 0.9169 | 0.7989 |
| XGBoost | 0.9185 | 0.8010 | 0.9139 | 0.8023 |
| LightGBM | 0.9159 | 0.7988 | 0.9110 | 0.7997 |
| TabNet | 0.9094 | 0.7921 | 0.9026 | 0.7935 |

> Decision thresholds were tuned for F1 on the **validation** set only, then frozen before testing.
> The stack beats XGBoost, TabNet, Random Forest and LightGBM with McNemar's p < 0.001. Against Logistic Regression it is statistically comparable (p = 0.768) — expected, since LR *is* the meta-learner.

**Operational view (XGBoost, threshold 0.74, November):** 14,288 delays caught, 5,016 missed, and only **2,084 false alarms** → **2.6 % false-alarm rate**, 74 % recall on delayed flights.

### Regression — minutes of arrival delay

| Model | Nov RMSE | Nov MAE | Nov R² | Dec RMSE | Dec MAE | Dec R² |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | **14.26** | **10.10** | **0.878** | **15.30** | **10.94** | **0.887** |
| XGBoost | 14.30 | 10.11 | 0.877 | 15.33 | 10.95 | 0.887 |
| Random Forest | 14.89 | 10.62 | 0.866 | 15.91 | 11.48 | 0.878 |
| TabNet | 14.94 | 10.70 | 0.866 | 16.02 | 11.60 | 0.877 |
| Linear Regression | 15.21 | 11.06 | 0.861 | 16.24 | 11.97 | 0.873 |

### Random split vs. temporal split (AUC)

| Model | Random | Nov | Dec | Drop |
|---|:---:|:---:|:---:|:---:|
| Random Forest | 0.9327 | 0.9197 | 0.9169 | −0.016 |
| XGBoost | 0.9341 | 0.9185 | 0.9139 | −0.020 |
| LightGBM | 0.9311 | 0.9159 | 0.9110 | −0.020 |
| TabNet | 0.9344 | 0.9094 | 0.9026 | −0.032 |
| **Stacking** | — | **0.9295** | **0.9254** | **smallest drop (−0.004)** |

Random splits flatter every tree and deep model; the stacking ensemble degrades the least when the data shifts.

---

## 🧠 Methodology

```
 BTS raw CSVs (Jan, Jun, Nov, Dec 2025)
          │
          ▼
 ┌──────────────────────┐   ┌─────────────────────────────────────────┐
 │ 1. Pre-EDA & Cleaning│──▶│ missing values · outliers · skew ·      │
 │    (notebook 1)      │   │ leakage check · winsorisation 1–99 %    │
 └──────────┬───────────┘   └─────────────────────────────────────────┘
            ▼
 ┌──────────────────────┐   300 K stratified sample (month × delay status)
 │ 2. Temporal split    │   Train/Val: Jan+Jun (85/15) · Test: Nov · Stress: Dec
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   26 pre-departure features, encoding maps fitted
 │ 3. Feature engineering│  on TRAIN ONLY and saved to artifacts/
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   5 classifiers ─▶ Logistic-Regression meta-learner
 │ 4. Modelling         │   5 regressors  ─▶ best: LightGBM
 │    (notebook 2)      │   cost-sensitive learning (+ SMOTE ablation)
 └──────────┬───────────┘
            ▼
 Threshold tuning (val) → Nov test (touched once) → Dec stress test → SHAP + McNemar
```

### Data

| Month (2025) | Role | Rows |
|---|---|---:|
| January | Train / Val | 539,747 |
| June | Train / Val | 611,575 |
| November | Temporal test | 570,550 |
| December | Holiday stress test | 582,304 |

The classification target is imbalanced (**3.65 : 1**, ≈ 21.5 % delayed), handled with `scale_pos_weight` / `class_weight='balanced'`.

### Preprocessing highlights
- Cancelled flights → delay set to 0, cancellation code → sentinel `N`; **zero missing values** after the pipeline.
- Winsorisation instead of row deletion: `DEP_DELAY` → [−15, 226], `ARR_DELAY` → [−37, 226] min (skew 10.1 → ~3).
- **Dropped as leakage:** `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY` — only known *after* landing.
- Detected multicollinearity (`CRS_ELAPSED_TIME` / `AIR_TIME` / `DISTANCE`, r ≈ 0.98–0.99).

### The 26 features

| Group | Features |
|---|---|
| **Time** | `DEP_HOUR`, `ARR_HOUR`, `IS_RUSH_HOUR`, `IS_EARLY_MORNING`, `MONTH`, `YEAR` |
| **Route / distance** | `ROUTE`, `DISTANCE`, `DIST_BIN`, `CRS_ELAPSED_TIME`, `ORIGIN`, `DEST`, `ORIGIN_STATE_ABR`, `DEST_STATE_ABR` |
| **Carrier** | `OP_UNIQUE_CARRIER` |
| **Departure status** | `DEP_DELAY`, `IS_DEP_LATE`, `DEP_DELAY_BUCKET` |
| **Disruption flags** | `CANCELLED`, `IS_CANCELLED`, `DIVERTED`, `CANCELLATION_CODE` |
| **Target-encoded rates** | `CARRIER_DELAY_RATE`, `ROUTE_DELAY_RATE`, `HOUR_DELAY_RATE`, **`TAIL_DELAY_RATE`** ⭐ |

⭐ **`TAIL_DELAY_RATE`** — mean of `1[ARR_DELAY > 15]` over an aircraft's training flights. Unseen tail numbers fall back to the global training rate (0.2255). Coverage: 97.3 % (Nov) / 96.5 % (Dec).

### Models
| Task | Models |
|---|---|
| Classification | Logistic Regression · Random Forest (200 trees) · LightGBM · XGBoost (300 trees, depth 6, lr 0.05, GPU) · TabNet (n_d = n_a = 64, 5 steps, entmax) · **Stacking** (LR meta-learner) |
| Regression | Linear Regression · Random Forest · LightGBM · XGBoost · TabNet |

Meta-learner coefficients: RF 3.83 · LR 2.39 · XGB 0.42 · TabNet 0.27 · LightGBM 0.16 (raw log-odds coefficients, not normalised importances).

---

## 📁 Repository Structure

```
flight-delay-prediction/
├── artifacts/        # Saved encoding maps & bounds (carrier/route/tail/hour maps,
│                     #   global_rate, winsor_bounds) and exported figures
├── data/
│   ├── README.md     # How to download the BTS data
│   └── raw/          # ← place the 4 CSVs here (not tracked by git)
├── notebooks/
│   ├── datascience05-pre-eda.ipynb   # EDA, cleaning, sampling, split, feature engineering
│   └── datascience05-model.ipynb     # Training, tuning, stacking, evaluation, SHAP
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & install
```bash
git clone https://github.com/ammar2almomani/flight-delay-prediction.git
cd flight-delay-prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the data
The raw BTS files are hundreds of MB and are **not** in the repo.

1. Go to <https://www.transtats.bts.gov/> → *Aviation → Airline On-Time Statistics and Delay Causes → Reporting Carrier On-Time Performance (1987–present)*.
2. Select year **2025**, then download **January, June, November, December**.
3. Tick the fields listed in [`data/README.md`](data/README.md), unzip, and rename into `data/raw/`:

| Month | File name |
|---|---|
| January | `T_ONTIME_REPORTING1.csv` |
| June | `T_ONTIME_REPORTING6.csv` |
| November | `T_ONTIME_REPORTING11.csv` |
| December | `T_ONTIME_REPORTING12.csv` |

> Row counts vary slightly with the extract date — small differences are normal.

---

## 🔁 Reproducing the Results

### ▶️ Quick start: how to run the models

```bash
# 1. Install everything (inside your virtual environment)
pip install -r requirements.txt

# 2. Launch Jupyter from the repo root
jupyter lab            # or: jupyter notebook

# 3. Run, in this order, using "Run All Cells"
#    notebooks/datascience05-pre-eda.ipynb   -> cleans data, builds features, saves maps to artifacts/
#    notebooks/datascience05-model.ipynb     -> trains + evaluates all models
```

Or run them headless from the terminal:

```bash
jupyter nbconvert --to notebook --execute notebooks/datascience05-pre-eda.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/datascience05-model.ipynb   --inplace
```

> ⚠️ Notebook 2 depends on the files saved by notebook 1, so always run notebook 1 first. Expect a long run on CPU (1.7 M rows, TabNet, SHAP); a GPU makes it much faster.

**Detailed order:**

| # | Notebook | What it does | Produces |
|---|---|---|---|
| 1 | `datascience05-pre-eda.ipynb` | Loads Jan/Jun/Nov, EDA, cleaning, winsorisation, 300 K stratified sample, temporal split, feature engineering | Train/val/test matrices, `winsor_bounds.csv`, `global_rate.csv`, `carrier_map.csv`, `route_map.csv`, `tail_map.csv`, `hour_map.csv` |
| 2 | `datascience05-model.ipynb` | Trains all classifiers & regressors, tunes thresholds, builds the stack, evaluates on Nov and Dec, runs McNemar + SHAP | Metrics tables, ROC curves, confusion matrix, SHAP plots |

**Notes**
- Paths in the notebooks point to a Kaggle environment (`/kaggle/input/...`). Edit them to `data/raw/` and `artifacts/` for local runs.
- A **GPU** is recommended (trained on an NVIDIA Tesla T4). XGBoost uses `device='cuda'`; switch to `device='cpu'` if you don't have one.
- All random seeds are fixed (`SEED`, `random_state=42`) for repeatability.
- The December file is processed with the **saved** maps and bounds from notebook 1 — nothing is re-fitted on holdout data.

---

## 🔍 Explainability (SHAP)

Mean |SHAP| on the November holdout (XGBoost classifier):

| Rank | Feature | Approx. mean \|SHAP\| | Why it matters |
|:---:|---|:---:|---|
| 1 | `DEP_DELAY` | ~2.5 | Late departure is the main driver of late arrival |
| 2 | `ROUTE_DELAY_RATE` | ~0.60 | Persistent route-level congestion |
| 3 | **`TAIL_DELAY_RATE`** ⭐ | ~0.50 | Aircraft-specific reliability — a signal absent from prior work |
| 4 | `ARR_HOUR` | — | Time-of-day schedule pressure |
| 5 | `CANCELLED` | — | Disruption status |

`IS_DEP_LATE` drops out of the top 10 because it is just a threshold on `DEP_DELAY`, so its signal is already captured.

---

## ⚠️ Limitations & Future Work

- **`DEP_DELAY` is a model input.** It is the strongest feature and is known when the aircraft pushes back, not at booking time. The model is therefore best read as a *"flight just left the gate — will it arrive late, and by how much?"* predictor rather than a day-ahead forecaster. Re-running without `DEP_DELAY` (and `IS_DEP_LATE`, `DEP_DELAY_BUCKET`) would give a stricter long-lead variant.
- **Cold-start aircraft:** ~3.5 % of December flights have unseen tail numbers and fall back to the global rate. A carrier-level fallback is a natural next step.
- **Limited calendar coverage:** four months of a single year (2025). More seasons and years would test generalisation further.
- **No weather / ATC / airport-congestion feeds** — only schedule and history features.
- **Sampling:** models train on a 300 K stratified sample rather than all 1.7 M rows.

**Ideas:** carrier-level tail fallback · live weather & NAS data · delay-propagation features along an aircraft's daily rotation · probability calibration · a small API / dashboard for real-time scoring.

---

## 📄 Paper & Citation

This repository accompanies the paper **"Flight Delay Prediction Using Stacking Ensemble and Novel Aircraft-Level Features Under Temporal Validation."**

```bibtex
@article{flightdelay2026,
  title  = {Flight Delay Prediction Using Stacking Ensemble and Novel Aircraft-Level Features Under Temporal Validation},
  author = {Almomani Almomani, Anas Odat, Mahmoud Al Waqfi , Ali Al Ekish},
  year   = {2026},
  note   = {Jordan University of Science and Technology}
}
```

---

## 👥 Authors

Department of Computer Science, **Jordan University of Science and Technology**, Irbid, Jordan.

| Name | Contact |
|---|---|
| Ammar Almomani | amalmomani23@cit.just.edu.jo |
| Anas Odat | amodat23@cit.just.edu.jo |
| Mahmoud Al Waqfi | mralwaqfi23@cit.just.edu.jo |
| Ali Al Ekish | asalekish23@cit.just.edu.jo |

---

## 📜 License & Acknowledgements

Released under the terms in [`LICENSE`](LICENSE).

Data: [U.S. Bureau of Transportation Statistics](https://www.transtats.bts.gov/). Built with scikit-learn, XGBoost, LightGBM, PyTorch-TabNet, SHAP, pandas and matplotlib.

<div align="center">

⭐ If this project helps your work, please consider starring the repo.

</div>
