"""
inference.py - Single-window prediction pipeline for the biofeedback wristband web UI.

Flow:
  1. User enters profile (gender, age, physical activity) and two sets of readings:
       - Baseline: sensor values when calm/at rest  (sets the personal zero)
       - Current:  sensor values right now
  2. Synthesize a 20-second window from each set of point-in-time readings.
  3. Z-score current features using baseline means + population stds matched to profile.
  4. NaN imputation (training col_means) → XGBoost predict → VIBRATE / NO VIBRATE.

For simulation:
  Generate N synthetic stress and N rest windows (with physiological noise),
  run all through the pipeline, and report a confusion matrix.
"""

import json
import math
import os
import random

import xgboost as xgb

from .features import (
    extract_window_features,
    z_score_query_window,
    FEATURE_NAMES,
    WINDOW_S,
)
from .model import _apply_means, load_model


# ---------------------------------------------------------------------------
# window synthesis from instantaneous sensor readings
# ---------------------------------------------------------------------------

def _gauss(val, frac, min_std):
    return val + random.gauss(0, max(abs(val) * frac, min_std))


def synthesize_window(eda_val, hr_val, acc_xyz, ibi_vals=None, noise=0.04):
    """
    Build a synthetic 20-second recording dict from single-point sensor readings.

    EDA and HR are replicated with small Gaussian noise; IBI is generated from
    the mean beat period (derived from HR if ibi_vals not supplied); ACC is
    replicated with small noise around the supplied x/y/z values.

    noise : fraction of each value used as Gaussian noise std.
    """
    ax, ay, az = acc_xyz

    # EDA: 4 Hz -> 80 samples
    eda = [_gauss(eda_val, noise, 0.002) for _ in range(80)]

    # HR: 1 Hz -> 20 samples
    hr = [_gauss(hr_val, noise * 0.5, 0.1) for _ in range(20)]

    # ACC: 32 Hz -> 640 samples
    acc_noise = max(abs(ax) + abs(ay) + abs(az), 10.0) * noise * 0.15
    acc = [[_gauss(ax, 0, acc_noise), _gauss(ay, 0, acc_noise), _gauss(az, 0, acc_noise)]
           for _ in range(640)]

    # IBI: generate beats across the 20 s window
    if ibi_vals and len(ibi_vals) >= 1:
        mean_ibi = sum(ibi_vals) / len(ibi_vals)
    else:
        mean_ibi = 60.0 / max(hr_val, 20.0)
    ibi_noise = max(mean_ibi * noise, 0.005)
    ibi_entries = []
    t = _gauss(mean_ibi, 0, ibi_noise)
    while t < WINDOW_S:
        dur = max(0.25, _gauss(mean_ibi, 0, ibi_noise))
        ibi_entries.append((t, dur))
        t += dur

    return {
        "eda": eda, "hr": hr, "acc": acc, "ibi": ibi_entries,
        "bvp": [], "temp": [],
        "fs": {"eda": 4.0, "hr": 1.0, "acc": 32.0, "bvp": 64.0, "temp": 4.0},
        "stress_intervals": [], "rest_intervals": [],
        "subject_id": "live", "signal_duration_s": WINDOW_S,
    }


# ---------------------------------------------------------------------------
# profile-aware normalization
# ---------------------------------------------------------------------------

def _pop_stds_for_profile(profiles, gender, activity):
    """
    Average the norm_stds of training subjects whose gender and activity match.
    Falls back progressively: gender-only -> all subjects.
    """
    subs = list(profiles["subjects"].values())
    matched = [s for s in subs
               if s["gender"].lower() == gender.lower()
               and s["activity"].lower() == activity.lower()]
    if not matched:
        matched = [s for s in subs if s["gender"].lower() == gender.lower()]
    if not matched:
        matched = subs

    n = len(FEATURE_NAMES)
    avgs = []
    for j in range(n):
        vals = [s["norm_stds"][j] for s in matched if s["norm_stds"][j] > 1e-8]
        avgs.append(sum(vals) / len(vals) if vals else 1.0)
    return avgs


def _compute_baseline_means(baseline_eda, baseline_hr, baseline_acc):
    """
    Synthesize a low-noise baseline window and extract its features as the
    personal mean vector. NaN entries are replaced with 0.
    """
    win = synthesize_window(baseline_eda, baseline_hr, baseline_acc, noise=0.02)
    feats = extract_window_features(win, 0, WINDOW_S)
    return [v if not math.isnan(v) else 0.0 for v in feats]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def predict_one(model, col_means, profiles,
                eda, hr, acc_xyz, ibi_vals,
                baseline_eda, baseline_hr, baseline_acc,
                gender="m", activity="Yes", n_mc=30):
    """
    Classify a single point-in-time reading.

    Runs n_mc Monte Carlo samples (each with independent synthesis noise) and
    averages the probabilities, so P(stress) is stable across repeated calls.

    Returns dict with:
      prediction  : 0 or 1
      label       : "VIBRATE" | "DO NOT VIBRATE"
      probability : float P(stress)  (mean over n_mc samples)
      features    : {name: z_scored_value}  (from the first sample)
    """
    b_means = _compute_baseline_means(baseline_eda, baseline_hr, baseline_acc)
    pop_stds = _pop_stds_for_profile(profiles, gender, activity)

    all_z_imp = []
    first_z = None
    for i in range(n_mc):
        win = synthesize_window(eda, hr, acc_xyz, ibi_vals, noise=0.04)
        feats = extract_window_features(win, 0, WINDOW_S)
        z = [(v - m) / s if not math.isnan(v) else float("nan")
             for v, m, s in zip(feats, b_means, pop_stds)]
        z_imp = _apply_means([z], col_means)[0]
        all_z_imp.append(z_imp)
        if first_z is None:
            first_z = z

    d = xgb.DMatrix(all_z_imp, feature_names=FEATURE_NAMES)
    probs = model.predict(d)
    prob = float(sum(probs) / len(probs))
    pred = 1 if prob >= 0.5 else 0

    return {
        "prediction": pred,
        "label": "VIBRATE" if pred == 1 else "DO NOT VIBRATE",
        "probability": round(prob, 4),
        "features": {name: (round(float(zv), 3) if not math.isnan(zv) else None)
                     for name, zv in zip(FEATURE_NAMES, first_z)},
    }


def simulate_stress_detection(model, col_means, profiles,
                               eda, hr, acc_xyz, ibi_vals,
                               baseline_eda, baseline_hr, baseline_acc,
                               gender="m", activity="Yes",
                               n_sim=60, noise=0.07):
    """
    Simulate n_sim stress windows (around current readings) and n_sim rest windows
    (around baseline readings), run through the model, and return a confusion matrix.

    Simulated ground-truth labels:
      stress windows -> 1 (should vibrate)
      rest windows   -> 0 (should not vibrate)
    """
    random.seed(42)
    b_means = _compute_baseline_means(baseline_eda, baseline_hr, baseline_acc)
    pop_stds = _pop_stds_for_profile(profiles, gender, activity)

    pairs = []
    for (val_eda, val_hr, val_acc, val_ibi, true_label) in [
        (eda, hr, acc_xyz, ibi_vals, 1),
        (baseline_eda, baseline_hr, baseline_acc, None, 0),
    ]:
        for _ in range(n_sim):
            win = synthesize_window(val_eda, val_hr, val_acc, val_ibi, noise=noise)
            feats = extract_window_features(win, 0, WINDOW_S)
            z = [(v - m) / s if not math.isnan(v) else float("nan")
                 for v, m, s in zip(feats, b_means, pop_stds)]
            z_imp = _apply_means([z], col_means)[0]
            d = xgb.DMatrix([z_imp], feature_names=FEATURE_NAMES)
            prob = float(model.predict(d)[0])
            pairs.append((1 if prob >= 0.5 else 0, true_label))

    tp = sum(1 for p, t in pairs if p == 1 and t == 1)
    tn = sum(1 for p, t in pairs if p == 0 and t == 0)
    fp = sum(1 for p, t in pairs if p == 1 and t == 0)
    fn = sum(1 for p, t in pairs if p == 0 and t == 1)
    total = tp + tn + fp + fn

    def _safe(num, den):
        return round(num / den, 4) if den else None

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_sim": n_sim,
        "accuracy":    _safe(tp + tn, total),
        "sensitivity": _safe(tp, tp + fn),
        "specificity": _safe(tn, tn + fp),
        "precision":   _safe(tp, tp + fp),
    }


# ---------------------------------------------------------------------------
# model / profile loading helpers
# ---------------------------------------------------------------------------

def load_trained_model(output_dir):
    """
    Load the trained XGBoost model, imputation means, and (optionally)
    subject profiles from the output directory.

    Returns (model, col_means, profiles).
    profiles is None when subject_profiles.json is absent.
    """
    model_path = os.path.join(output_dir, "model.json")
    meta_path = os.path.join(output_dir, "model_meta.json")
    model, col_means = load_model(model_path, meta_path)

    profiles_path = os.path.join(output_dir, "subject_profiles.json")
    profiles = None
    if os.path.isfile(profiles_path):
        with open(profiles_path) as f:
            profiles = json.load(f)
    return model, col_means, profiles


_FALLBACK_PROFILES = {
    "feature_names": FEATURE_NAMES,
    "subjects": {},
}


def get_profiles_or_fallback(profiles):
    """Return profiles if populated, otherwise a neutral fallback."""
    if profiles and profiles.get("subjects"):
        return profiles
    return _FALLBACK_PROFILES


# ---------------------------------------------------------------------------
# full-session window prediction (used by predict.py / old serve.py)
# ---------------------------------------------------------------------------

def predict_stress_window(model, col_means, subj, t_start, t_end):
    """
    Predict stress for a single [t_start, t_end] window from a loaded session.

    Uses per-subject rest-baseline z-scoring (same as training) and then
    applies training column-mean imputation before XGBoost inference.
    """
    z = z_score_query_window(subj, t_start, t_end)
    z_imp = _apply_means([z], col_means)[0]
    d = xgb.DMatrix([z_imp], feature_names=FEATURE_NAMES)
    prob = float(model.predict(d)[0])
    pred = 1 if prob >= 0.5 else 0
    return {
        "pred_prob": round(prob, 4),
        "pred_class": "stress" if pred == 1 else "rest",
        "feedback": "VIBRATE" if pred == 1 else "DO NOT VIBRATE",
    }
