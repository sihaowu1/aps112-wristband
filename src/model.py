"""
model.py - XGBoost classifier training, evaluation, and persistence.

Binary output:
  1 = vibrate  (stress detected)
  0 = no vibrate (rest / no stress)

SMOTE is applied to the training set to balance the minority stress class before
XGBoost training. With a balanced training set, scale_pos_weight is set to 1.0.
"""

import json
import math
import time

import xgboost as xgb
from imblearn.over_sampling import SMOTE

from .features import FEATURE_NAMES


# ---------------------------------------------------------------------------
# NaN imputation (column-mean from training set)
# ---------------------------------------------------------------------------

def _impute_nan(X):
    """
    Replace NaN with per-column training-set mean.
    Returns (X_imputed, col_means).
    """
    n_feat = len(X[0])
    col_sums = [0.0] * n_feat
    col_counts = [0] * n_feat
    for row in X:
        for j, v in enumerate(row):
            if not math.isnan(v):
                col_sums[j] += v
                col_counts[j] += 1
    col_means = [
        col_sums[j] / col_counts[j] if col_counts[j] > 0 else 0.0
        for j in range(n_feat)
    ]
    return [[v if not math.isnan(v) else col_means[j] for j, v in enumerate(row)]
            for row in X], col_means


def _apply_means(X, col_means):
    return [[v if not math.isnan(v) else col_means[j] for j, v in enumerate(row)]
            for row in X]


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train):
    """
    Train a single XGBoost binary classifier with SMOTE oversampling.

    Pipeline:
      1. Impute NaN with column means (from training set).
      2. Apply SMOTE to balance stress vs. rest classes.
      3. Train XGBoost on the balanced, imputed data.

    Returns (model, col_means) where col_means must be applied to the test set.
    """
    # Step 1: impute NaN
    X_imp, col_means = _impute_nan(X_train)

    # Step 2: SMOTE — balance classes (k_neighbors capped at minority class size - 1)
    n_stress = sum(y_train)
    k = min(5, n_stress - 1) if n_stress > 1 else 1
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_bal, y_bal = smote.fit_resample(X_imp, y_train)

    # Step 3: XGBoost on balanced data (scale_pos_weight=1 since classes are equal)
    dtrain = xgb.DMatrix(X_bal, label=y_bal, feature_names=FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": 1.0,
        "seed": 42,
        "verbosity": 0,
    }

    model = xgb.train(params, dtrain, num_boost_round=300, verbose_eval=False)
    return model, col_means


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate(model, col_means, X_test, y_test):
    """
    Run inference on X_test and return metrics dict.
    """
    X_imp = _apply_means(X_test, col_means)
    dtest = xgb.DMatrix(X_imp, feature_names=FEATURE_NAMES)

    t0 = time.perf_counter()
    probs = model.predict(dtest)
    t1 = time.perf_counter()
    processing_ms = (t1 - t0) * 1000.0

    preds = [1 if p >= 0.5 else 0 for p in probs]

    tp = sum(1 for p, t in zip(preds, y_test) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(preds, y_test) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(preds, y_test) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, y_test) if p == 0 and t == 1)

    accuracy = (tp + tn) / len(y_test) if y_test else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "processing_time_ms": round(processing_ms, 3),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_test": len(y_test),
        "n_test_stress": sum(y_test),
        "n_test_rest": len(y_test) - sum(y_test),
    }, preds, [float(p) for p in probs]


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_model(model, col_means, model_path, meta_path):
    model.save_model(model_path)
    with open(meta_path, "w") as f:
        json.dump({"col_means": col_means, "feature_names": FEATURE_NAMES}, f, indent=2)


def load_model(model_path, meta_path):
    model = xgb.Booster()
    model.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta["col_means"]


def predict_proba(model, col_means, X_rows):
    """
    Return numpy-like list of probabilities P(class=1) for each row.
    Applies training-set NaN imputation (col_means) before inference.
    """
    X_imp = _apply_means(X_rows, col_means)
    d = xgb.DMatrix(X_imp, feature_names=FEATURE_NAMES)
    return model.predict(d)
