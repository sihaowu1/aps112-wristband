#!/usr/bin/env python3
"""
train.py - End-to-end training pipeline for the Portable Biofeedback Wristband prototype.

Usage:
    python3 train.py [--data-dir PATH] [--output-dir PATH]

Defaults:
    --data-dir   : <script_dir>/data/PhysioNet
    --output-dir : <script_dir>/outputs

Outputs written to output-dir:
    model.json         - trained XGBoost model
    model_meta.json    - imputation column means + feature names
    metrics.json       - test-set evaluation metrics
    predictions.csv    - per-window predictions on the held-out test set
    confusion_matrix.txt
    methods_summary.txt
"""

import argparse
import csv
import json
import os
import random
import sys
import time

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all_stress_subjects
from src.features import build_dataset, compute_subject_norm_stats, FEATURE_NAMES, WINDOW_S, STEP_S
from src.model import train_xgboost, evaluate, save_model


# ---------------------------------------------------------------------------
# subject profile helpers (for web UI normalization and matching)
# ---------------------------------------------------------------------------

def _load_subject_info(data_dir):
    """Load subject-info.csv → {subject_id: {gender, age, activity}} dict."""
    path = os.path.join(data_dir, "subject-info.csv")
    info = {}
    if not os.path.exists(path):
        return info
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("Info ", row.get("Info", "")).strip()
            if not sid:
                continue
            info[sid] = {
                "gender": row.get("Gender", "").strip().lower(),
                "age": row.get("Age", "").strip(),
                "activity": row.get("Does physical activity regularly?", "").strip(),
            }
    return info


def _save_subject_profiles(subjects, data_dir, output_dir):
    """
    Compute per-subject baseline norm stats and merge with demographic info.
    Saves outputs/subject_profiles.json used by the web inference UI.
    """
    from src.features import compute_subject_norm_stats, FEATURE_NAMES
    subject_info = _load_subject_info(data_dir)
    norm_stats = compute_subject_norm_stats(subjects)

    profiles = {}
    for sid, stats in norm_stats.items():
        demo = subject_info.get(sid, {})
        profiles[sid] = {
            "gender":   demo.get("gender", "unknown"),
            "age":      demo.get("age", ""),
            "activity": demo.get("activity", "unknown"),
            "norm_means": stats["norm_means"],
            "norm_stds":  stats["norm_stds"],
        }

    path = os.path.join(output_dir, "subject_profiles.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"feature_names": FEATURE_NAMES, "subjects": profiles}, f, indent=2)
    print(f"  Subject profiles saved: {path}  ({len(profiles)} subjects)")


# ---------------------------------------------------------------------------
# subject-wise train/test split
# ---------------------------------------------------------------------------

def subject_split(X, y, groups, test_fraction=0.30, seed=42):
    """
    Split X, y into train and test by unique subject ID (no window-level leakage).

    Stratified by protocol version (S-prefix = V1, f-prefix = V2) to ensure
    both versions appear in both splits.
    """
    rng = random.Random(seed)

    unique_subjects = sorted(set(groups))
    v1 = [s for s in unique_subjects if s.startswith("S")]
    v2 = [s for s in unique_subjects if s.startswith("f")]

    rng.shuffle(v1)
    rng.shuffle(v2)

    def split_list(lst, fraction):
        n_test = max(1, round(len(lst) * fraction))
        return lst[:n_test], lst[n_test:]

    v1_test, v1_train = split_list(v1, test_fraction)
    v2_test, v2_train = split_list(v2, test_fraction)

    test_subjects = set(v1_test + v2_test)
    train_subjects = set(v1_train + v2_train)

    X_train, y_train = [], []
    X_test, y_test = [], []
    groups_test = []

    for feats, label, sid in zip(X, y, groups):
        if sid in test_subjects:
            X_test.append(feats)
            y_test.append(label)
            groups_test.append(sid)
        elif sid in train_subjects:
            X_train.append(feats)
            y_train.append(label)

    return X_train, y_train, X_test, y_test, groups_test, sorted(train_subjects), sorted(test_subjects)


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------

def print_metrics(metrics):
    print(f"\n{'='*50}")
    print("  TEST SET METRICS")
    print(f"{'='*50}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Sensitivity  : {metrics['sensitivity']:.4f}  (recall for stress class)")
    print(f"  Specificity  : {metrics['specificity']:.4f}  (recall for rest class)")
    print(f"  Precision    : {metrics['precision']:.4f}  (PPV for stress class)")
    print(f"  Proc. time   : {metrics['processing_time_ms']:.2f} ms  "
          f"({metrics['n_test']} windows)")
    print(f"{'='*50}")
    print(f"  Confusion matrix (actual → predicted):")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}")
    print(f"  FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"{'='*50}\n")


def save_outputs(output_dir, metrics, preds, probs, groups_test, y_test,
                 train_subjects, test_subjects):
    os.makedirs(output_dir, exist_ok=True)

    # metrics.json
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # predictions.csv
    pred_path = os.path.join(output_dir, "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "true_label", "pred_label", "pred_prob",
                          "true_class", "pred_class"])
        for sid, true, pred, prob in zip(groups_test, y_test, preds, probs):
            writer.writerow([
                sid, true, pred, f"{prob:.4f}",
                "stress" if true == 1 else "rest",
                "vibrate" if pred == 1 else "no_vibrate",
            ])

    # confusion_matrix.txt
    cm_path = os.path.join(output_dir, "confusion_matrix.txt")
    with open(cm_path, "w") as f:
        f.write("Confusion Matrix — Rows: Actual, Columns: Predicted\n\n")
        f.write("              Pred:rest  Pred:stress\n")
        f.write(f"Actual:rest     {metrics['tn']:6d}       {metrics['fp']:6d}\n")
        f.write(f"Actual:stress   {metrics['fn']:6d}       {metrics['tp']:6d}\n\n")
        f.write(f"Accuracy    : {metrics['accuracy']:.4f}\n")
        f.write(f"Sensitivity : {metrics['sensitivity']:.4f}\n")
        f.write(f"Specificity : {metrics['specificity']:.4f}\n")
        f.write(f"Precision   : {metrics['precision']:.4f}\n")
        f.write(f"Proc. time  : {metrics['processing_time_ms']:.2f} ms "
                f"({metrics['n_test']} windows)\n")

    # methods_summary.txt
    ms_path = os.path.join(output_dir, "methods_summary.txt")
    with open(ms_path, "w") as f:
        f.write("METHODS SUMMARY — Portable Biofeedback Wristband Prototype\n")
        f.write("="*60 + "\n\n")
        f.write("Dataset\n")
        f.write("  Source  : PhysioNet Wearable Device Dataset (Empatica E4)\n")
        f.write("  Sessions: STRESS only (aerobic/anaerobic excluded)\n")
        f.write("  Subjects: 18 V1 (S01–S18) + 18 V2 (f01–f18)\n\n")
        f.write("Label Mapping\n")
        f.write("  1 = vibrate    (stress task active)\n")
        f.write("  0 = no vibrate (rest / baseline period)\n")
        f.write("  V1 stress intervals: tag-pairs [3,4],[5,6],[7,8],[9,10],[11,12]\n")
        f.write("  V1 rest intervals  : tag-pairs [2,3],[4,5],[6,7]\n")
        f.write("  V2 stress intervals: tag-pairs [2,3],[4,5],[6,7],[8,9]\n")
        f.write("  V2 rest intervals  : tag-pairs [1,2],[3,4],[7,8]\n\n")
        f.write("Windowing\n")
        f.write(f"  Window : {WINDOW_S} s, Step : {STEP_S} s (50% overlap)\n\n")
        f.write("Features (21 total)\n")
        f.write("  EDA  : mean, std, slope, range\n")
        f.write("  HR   : mean, std, min, max\n")
        f.write("  IBI  : mean, std, RMSSD, SDNN, pNN50\n")
        f.write("  ACC  : mag_mean, mag_std, var_x, var_y, var_z, mean_jerk\n")
        f.write("  TEMP : mean, slope\n\n")
        f.write("Train/Test Split\n")
        f.write(f"  Strategy : subject-wise 70/30, stratified by protocol version\n")
        f.write(f"  Train subjects ({len(train_subjects)}): {', '.join(train_subjects)}\n")
        f.write(f"  Test  subjects ({len(test_subjects)}): {', '.join(test_subjects)}\n\n")
        f.write("Model\n")
        f.write("  Algorithm    : XGBoost (binary:logistic)\n")
        f.write("  Rounds       : 300 boosting rounds\n")
        f.write("  Depth        : 6\n")
        f.write("  Learning rate: 0.1\n")
        f.write("  Imbalance    : scale_pos_weight = n_rest / n_stress\n")
        f.write("  NaN handling : column-mean imputation from training set\n")

    print(f"  Outputs saved to: {output_dir}/")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train the Portable Biofeedback Wristband stress classifier."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--data-dir",
        default=os.path.join(script_dir, "data", "PhysioNet"),
        help="Path to the PhysioNet dataset root (default: ./data/PhysioNet)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(script_dir, "outputs"),
        help="Directory for model and evaluation outputs (default: ./outputs)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    t_pipeline_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading STRESS session data...")
    subjects = load_all_stress_subjects(data_dir, verbose=True)
    if not subjects:
        print("ERROR: No subjects loaded. Check data directory.", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(subjects)} subjects.")

    # ------------------------------------------------------------------
    # 2. Feature extraction
    # ------------------------------------------------------------------
    print("\n[2/5] Extracting windowed features...")
    X, y, groups = build_dataset(subjects, verbose=True)
    if not X:
        print("ERROR: Feature matrix is empty.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Subject-wise train/test split
    # ------------------------------------------------------------------
    print("\n[3/5] Splitting by subject (70/30)...")
    X_train, y_train, X_test, y_test, groups_test, train_subs, test_subs = \
        subject_split(X, y, groups)

    n_train_stress = sum(y_train)
    n_train_rest = len(y_train) - n_train_stress
    n_test_stress = sum(y_test)
    n_test_rest = len(y_test) - n_test_stress

    print(f"  Train: {len(y_train)} windows  (stress={n_train_stress}, rest={n_train_rest})")
    print(f"  Test : {len(y_test)} windows  (stress={n_test_stress}, rest={n_test_rest})")
    print(f"  Train subjects ({len(train_subs)}): {', '.join(train_subs)}")
    print(f"  Test  subjects ({len(test_subs)}): {', '.join(test_subs)}")

    if n_train_stress == 0:
        print("ERROR: No stress windows in training set.", file=sys.stderr)
        sys.exit(1)
    if n_test_stress == 0:
        print("ERROR: No stress windows in test set.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print("\n[4/5] Training XGBoost classifier...")
    t_train_start = time.perf_counter()
    model, col_means = train_xgboost(X_train, y_train)
    t_train_end = time.perf_counter()
    print(f"  Training time: {(t_train_end - t_train_start):.2f}s")

    # Save model
    model_path = os.path.join(output_dir, "model.json")
    meta_path = os.path.join(output_dir, "model_meta.json")
    os.makedirs(output_dir, exist_ok=True)
    save_model(model, col_means, model_path, meta_path)
    print(f"  Model saved: {model_path}")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating on held-out test set...")
    metrics, preds, probs = evaluate(model, col_means, X_test, y_test)

    # Add split metadata to metrics
    metrics["train_subjects"] = train_subs
    metrics["test_subjects"] = test_subs
    metrics["n_train"] = len(y_train)
    metrics["n_train_stress"] = n_train_stress
    metrics["n_train_rest"] = n_train_rest

    print_metrics(metrics)
    save_outputs(output_dir, metrics, preds, probs, groups_test, y_test,
                 train_subs, test_subs)

    # ------------------------------------------------------------------
    # 6. Save subject profiles for web inference
    # ------------------------------------------------------------------
    print("[6/6] Saving subject profiles for web UI...")
    _save_subject_profiles(subjects, data_dir, output_dir)

    t_pipeline_end = time.perf_counter()
    print(f"  Total pipeline time: {(t_pipeline_end - t_pipeline_start):.1f}s\n")


if __name__ == "__main__":
    main()
