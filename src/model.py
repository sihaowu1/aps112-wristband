"""
model.py - XGBoost classifier training, evaluation, and persistence.

Binary output:
  1 = vibrate  (stress detected)
  0 = no vibrate (rest / no stress)

Class imbalance is handled via scale_pos_weight (ratio of rest to stress windows).
Decision threshold is optimized on training-set predictions to maximize the
geometric mean of sensitivity and specificity (Youden's J statistic).
"""

import json
import math
import random
import time

import xgboost as xgb

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
# threshold optimization
# ---------------------------------------------------------------------------

def _find_optimal_threshold(probs, y_true, lo=0.25, hi=0.55, steps=61):
    """
    Sweep thresholds and pick the one maximizing the geometric mean of
    sensitivity and specificity (equivalent to maximizing Youden's J).
    """
    best_thresh, best_gmean = 0.5, -1.0
    for i in range(steps):
        t = lo + (hi - lo) * i / (steps - 1)
        tp = fn = tn = fp = 0
        for p, y in zip(probs, y_true):
            if y == 1:
                if p >= t:
                    tp += 1
                else:
                    fn += 1
            else:
                if p < t:
                    tn += 1
                else:
                    fp += 1
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        gmean = math.sqrt(sens * spec)
        if gmean > best_gmean:
            best_gmean = gmean
            best_thresh = t
    return round(best_thresh, 4)


def _threshold_via_cv(X_train, y_train, groups_train, params, num_boost_round, n_folds=3):
    """
    Subject-wise k-fold CV on the training set to select the optimal decision
    threshold without overfitting it to the test set.
    """
    unique_subs = sorted(set(groups_train))
    rng = random.Random(42)
    rng.shuffle(unique_subs)

    # assign subjects to folds
    fold_of = {}
    for i, s in enumerate(unique_subs):
        fold_of[s] = i % n_folds

    all_oof_probs = []
    all_oof_labels = []

    for fold_idx in range(n_folds):
        tr_X, tr_y, val_X, val_y = [], [], [], []
        for x, label, sid in zip(X_train, y_train, groups_train):
            if fold_of[sid] == fold_idx:
                val_X.append(x)
                val_y.append(label)
            else:
                tr_X.append(x)
                tr_y.append(label)

        if not tr_X or not val_X:
            continue

        tr_imp, tr_means = _impute_nan(tr_X)
        val_imp = _apply_means(val_X, tr_means)

        n_stress = sum(tr_y)
        n_rest = len(tr_y) - n_stress
        fold_params = dict(params)
        fold_params["scale_pos_weight"] = n_rest / n_stress if n_stress > 0 else 1.0

        dtrain = xgb.DMatrix(tr_imp, label=tr_y, feature_names=FEATURE_NAMES)
        dval = xgb.DMatrix(val_imp, feature_names=FEATURE_NAMES)
        m = xgb.train(fold_params, dtrain, num_boost_round=num_boost_round, verbose_eval=False)
        probs = m.predict(dval)
        all_oof_probs.extend(float(p) for p in probs)
        all_oof_labels.extend(val_y)

    if not all_oof_probs:
        return 0.5
    return _find_optimal_threshold(all_oof_probs, all_oof_labels)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "seed": 42,
    "verbosity": 0,
}
NUM_BOOST_ROUND = 300


def train_xgboost(X_train, y_train, groups_train=None):
    """
    Train a single XGBoost binary classifier.

    Pipeline:
      1. Impute NaN with column means (from training set).
      2. Train XGBoost with scale_pos_weight to handle class imbalance.
      3. Optimize decision threshold via subject-wise CV (if groups provided).

    Returns (model, col_means, threshold).
    """
    # Step 1: impute NaN
    X_imp, col_means = _impute_nan(X_train)

    # Step 2: compute class weight
    n_stress = sum(y_train)
    n_rest = len(y_train) - n_stress
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = n_rest / n_stress if n_stress > 0 else 1.0

    # Step 3: train
    dtrain = xgb.DMatrix(X_imp, label=y_train, feature_names=FEATURE_NAMES)
    model = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND, verbose_eval=False)

    # Step 4: find optimal threshold via subject-wise CV
    if groups_train is not None:
        threshold = _threshold_via_cv(X_train, y_train, groups_train, XGB_PARAMS, NUM_BOOST_ROUND)
        print(f"  Optimal threshold (CV): {threshold}")
    else:
        threshold = 0.5

    return model, col_means, threshold


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate(model, col_means, X_test, y_test, threshold=0.5):
    """
    Run inference on X_test and return metrics dict.
    """
    X_imp = _apply_means(X_test, col_means)
    dtest = xgb.DMatrix(X_imp, feature_names=FEATURE_NAMES)

    t0 = time.perf_counter()
    probs = model.predict(dtest)
    t1 = time.perf_counter()
    processing_ms = (t1 - t0) * 1000.0

    preds = [1 if p >= threshold else 0 for p in probs]

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
        "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_test": len(y_test),
        "n_test_stress": sum(y_test),
        "n_test_rest": len(y_test) - sum(y_test),
    }, preds, [float(p) for p in probs]


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_model(model, col_means, model_path, meta_path, threshold=0.5):
    model.save_model(model_path)
    with open(meta_path, "w") as f:
        json.dump({
            "col_means": col_means,
            "feature_names": FEATURE_NAMES,
            "threshold": threshold,
        }, f, indent=2)


def load_model(model_path, meta_path):
    model = xgb.Booster()
    model.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    threshold = meta.get("threshold", 0.5)
    return model, meta["col_means"], threshold


def predict_proba(model, col_means, X_rows):
    """
    Return numpy-like list of probabilities P(class=1) for each row.
    Applies training-set NaN imputation (col_means) before inference.
    """
    X_imp = _apply_means(X_rows, col_means)
    d = xgb.DMatrix(X_imp, feature_names=FEATURE_NAMES)
    return model.predict(d)
