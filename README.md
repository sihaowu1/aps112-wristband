# Portable Biofeedback Wristband — Software Prototype

Binary classifier that predicts whether a wristband should **vibrate** (stress detected)
or **remain silent** (rest/no stress), using physiological and motion data from an
Empatica E4 wristband.

---

## Assumptions and Data Discovery

### Dataset
- **Source**: PhysioNet Wearable Device Dataset (Empatica E4)
- **Sessions used**: `STRESS` only — each session contains both stress tasks and
  rest/baseline periods, providing natural stress vs. rest labels from a single recording.
- **Excluded**: `AEROBIC` and `ANAEROBIC` sessions — exercise physiology differs from
  psychological stress and would introduce a confounded positive class if mixed.

### Subjects
| Group | IDs | Protocol | Stress tasks |
|-------|-----|----------|--------------|
| V1    | S01–S18 | 12 button-press tags | Stroop, TMCT, Real Opinion, Opposite Opinion, Subtract |
| V2    | f01–f18 | 9 button-press tags  | TMCT, Real Opinion, Opposite Opinion, Subtract |

Special cases handled:
- **f14**: split into `f14_a` (baseline only, no usable stress tags) and `f14_b`
  (full protocol, 9 tags). `f14_a` is skipped; `f14_b` is loaded as subject `f14`.
- **f07**: PPG and TEMP sensors covered during recording; only EDA and ACC are valid.
  These subjects are still included — missing HR/IBI/TEMP features are NaN-imputed.
- **S02**: duplicated signal rows in the raw file (noted in `data_constraints.txt`).
  Loaded as-is; duplication does not affect label assignment from timestamps.

---

## Label Mapping

Tags in each session mark protocol stage boundaries. The **augmented tag array**
prepends the session start time (t=0) to the button-press timestamps, producing:

```
tags[0] = 0.0   (session start, prepended)
tags[1..N] = button-press times in seconds from session start
```

### V1 (subjects starting with "S") — 12 button presses → 13 elements

| Interval | Label |
|----------|-------|
| `[tags[2], tags[3]]` | **0 — rest** (baseline) |
| `[tags[3], tags[4]]` | **1 — stress** (Stroop) |
| `[tags[4], tags[5]]` | **0 — rest** (First Rest) |
| `[tags[5], tags[6]]` | **1 — stress** (TMCT) |
| `[tags[6], tags[7]]` | **0 — rest** (Second Rest) |
| `[tags[7], tags[8]]` | **1 — stress** (Real Opinion) |
| `[tags[9], tags[10]]`| **1 — stress** (Opposite Opinion) |
| `[tags[11],tags[12]]`| **1 — stress** (Subtract) |

Short inter-task intervals `[tags[8],tags[9]]`, `[tags[10],tags[11]]`,
`[tags[12],end]` are excluded (stress-level reporting breaks; ambiguous).

### V2 (subjects starting with "f") — 9 button presses → 10 elements

| Interval | Label |
|----------|-------|
| `[tags[1], tags[2]]` | **0 — rest** (baseline) |
| `[tags[2], tags[3]]` | **1 — stress** (TMCT) |
| `[tags[3], tags[4]]` | **0 — rest** (First Rest) |
| `[tags[4], tags[5]]` | **1 — stress** (Real Opinion) |
| `[tags[6], tags[7]]` | **1 — stress** (Opposite Opinion) |
| `[tags[7], tags[8]]` | **0 — rest** (Second Rest) |
| `[tags[8], tags[9]]` | **1 — stress** (Subtract) |

Short interval `[tags[5], tags[6]]` is excluded (inter-task reporting break).

Binary encoding: **1 = vibrate** (stress), **0 = do not vibrate** (rest).

---

## Preprocessing

### Windowing
- **Window size**: 20 seconds
- **Step size**: 10 seconds (50 % overlap)
- A window is only included if it fits entirely within a labeled interval.
  Partial windows at interval boundaries are discarded.

### NaN handling
IBI features are set to NaN when fewer than 3 IBI observations fall within a window
(e.g., short stress intervals or connectivity gaps). All NaN values are imputed with
per-column training-set means before XGBoost training; the same means are stored in
`outputs/model_meta.json` and applied at inference.

---

## Feature Extraction

21 features are computed per 20-second window:

| Group | Features |
|-------|----------|
| EDA (4)  | `eda_mean`, `eda_std`, `eda_slope`, `eda_range` |
| HR (4)   | `hr_mean`, `hr_std`, `hr_min`, `hr_max` |
| IBI (5)  | `ibi_mean`, `ibi_std`, `ibi_rmssd`, `ibi_sdnn`, `ibi_pnn50` |
| ACC (6)  | `acc_mag_mean`, `acc_mag_std`, `acc_var_x`, `acc_var_y`, `acc_var_z`, `acc_mean_jerk` |
| TEMP (2) | `temp_mean`, `temp_slope` |

---

## Train/Test Split

- **Strategy**: subject-wise split (no window-level leakage across subjects)
- **Ratio**: approximately 70 % train / 30 % test
- **Stratification**: V1 and V2 subjects are split independently so both protocol
  versions are represented in both partitions
- **Class imbalance**: training windows are balanced with **SMOTE** (imbalanced-learn)
  after NaN imputation; XGBoost is then trained with `scale_pos_weight = 1.0` because
  the resampled training set is class-balanced

---

## Model

Single XGBoost binary classifier (`binary:logistic`):

| Hyperparameter | Value |
|----------------|-------|
| `num_boost_round` | 300 |
| `max_depth` | 6 |
| `eta` (learning rate) | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 5 |
| `scale_pos_weight` | 1.0 (after SMOTE balancing) |
| Decision threshold | 0.5 |

Training pipeline: impute NaNs → SMOTE → XGBoost.

No baseline model. No model comparison.

---

## How to Run

### Requirements
Python 3.8+.

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

Core packages: `numpy`, `scipy`, `scikit-learn`, `imbalanced-learn` (SMOTE), `xgboost`.
For the local web UI (`serve.py`): `fastapi`, `uvicorn`, `python-multipart` (also listed in `requirements.txt`).

### Training and evaluation (combined)
```bash
# From the project root (APS112/)
python train.py

# With explicit paths
python train.py --data-dir data/PhysioNet --output-dir outputs
```

On Linux/macOS you can use `python3` instead of `python`.

The script performs these steps automatically:
1. Load and parse STRESS sessions
2. Extract 20-second windowed features (10 s step)
3. Subject-wise ~70/30 split
4. Train XGBoost (NaN imputation → SMOTE → boosting)
5. Evaluate on the held-out test set and save metrics, predictions, confusion matrix, methods summary
6. Write `subject_profiles.json` for the web UI (per-subject baseline norms and demographics)

### Inference (one 20 s window)

After `train.py` has produced `outputs/model.json` and `outputs/model_meta.json`, you can classify a single **20 s** window (same length and feature pipeline as training; per-subject baseline z-scoring uses rest intervals from `tags.csv` or `--rest-intervals`).

**CLI** — point at a STRESS session folder (Empatica CSVs) and a `[t_start, t_end]` window of exactly 20 seconds:

```bash
python predict.py --subject S01 --t-start 180 --t-end 200
# or
python predict.py --subject-dir path/to/session --t-start 0 --t-end 20
```

Optional: `--rest-intervals "[[0,120],[300,400]]"` overrides `tags.csv` for baseline z-scoring. You need at least one rest interval long enough to form a full 20 s window.

**Local web UI** — upload `EDA.csv` (required), plus `HR.csv`, `ACC.csv`, `IBI.csv`, and optionally `tags.csv`; set window times and optional rest-interval JSON:

```bash
pip install -r requirements.txt
python serve.py
# Open http://127.0.0.1:8000
```

Set `MODEL_DIR` if the model lives outside `./outputs`.

---

## Outputs

All files are written to `outputs/` (created automatically):

| File | Description |
|------|-------------|
| `model.json` | Trained XGBoost model (XGBoost JSON format) |
| `model_meta.json` | Feature names + NaN imputation means |
| `subject_profiles.json` | Per-subject baseline norm stats + demographics (for `serve.py`) |
| `metrics.json` | Accuracy, sensitivity, specificity, precision, processing time |
| `predictions.csv` | Per-window predictions on held-out test set |
| `confusion_matrix.txt` | Human-readable confusion matrix |
| `methods_summary.txt` | Concise methods description |

---

## Label Ambiguity Note

The stress_level CSVs (`Stress_Level_v1.csv`, `Stress_Level_v2.csv`) record
*self-reported* stress scores per task (0–10 scale). These are not used for labeling —
the **protocol-based tag intervals** provide ground-truth temporal windows because
self-reported scores capture retrospective perceived stress, while the tags define
the actual stimulus onset/offset times. This is the standard approach in physiological
stress detection research.
