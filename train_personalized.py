#!/usr/bin/env python3
"""
train_personalized.py - Personalized (mixed-split) training pipeline.

Instead of a pure subject-wise split, this randomly allocates 70% of each
subject's windows to training and 30% to testing.  The model sees some data
from every subject, which acts as calibration — mirroring a real wristband
deployment where the device would collect baseline data on first use.

Usage:
    python3 train_personalized.py [--data-dir PATH] [--output-dir PATH]

Defaults:
    --data-dir   : <script_dir>/data/PhysioNet
    --output-dir : <script_dir>/outputs_personalized
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all_stress_subjects
from src.features import build_dataset, FEATURE_NAMES, WINDOW_S, STEP_S
from src.model import train_xgboost, evaluate, save_model


# ---------------------------------------------------------------------------
# within-subject mixed split
# ---------------------------------------------------------------------------

def mixed_split(X, y, groups, test_fraction=0.30, seed=42):
    """
    For each subject, randomly assign 70% of their windows to train and 30%
    to test.  Every subject appears in both splits.
    """
    rng = random.Random(seed)

    # Group windows by subject
    subject_indices = {}
    for i, sid in enumerate(groups):
        subject_indices.setdefault(sid, []).append(i)

    train_idx, test_idx = [], []

    for sid in sorted(subject_indices):
        indices = subject_indices[sid][:]
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * test_fraction))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    groups_test = [groups[i] for i in test_idx]
    groups_train = [groups[i] for i in train_idx]

    all_subjects = sorted(subject_indices.keys())
    return X_train, y_train, groups_train, X_test, y_test, groups_test, all_subjects


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------

def print_metrics(metrics):
    print(f"\n{'='*50}")
    print("  TEST SET METRICS (personalized mixed-split)")
    print(f"{'='*50}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Sensitivity  : {metrics['sensitivity']:.4f}  (recall for stress class)")
    print(f"  Specificity  : {metrics['specificity']:.4f}  (recall for rest class)")
    print(f"  Precision    : {metrics['precision']:.4f}  (PPV for stress class)")
    print(f"  Proc. time   : {metrics['processing_time_ms']:.2f} ms  "
          f"({metrics['n_test']} windows)")
    print(f"{'='*50}")
    print(f"  Confusion matrix (actual -> predicted):")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}")
    print(f"  FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"{'='*50}\n")


def save_outputs(output_dir, metrics, preds, probs, groups_test, y_test,
                 all_subjects):
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
        f.write("Confusion Matrix (Personalized Mixed-Split)\n")
        f.write("Rows: Actual, Columns: Predicted\n\n")
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
        f.write("METHODS SUMMARY — Personalized Mixed-Split Model\n")
        f.write("=" * 60 + "\n\n")
        f.write("Dataset\n")
        f.write("  Source  : PhysioNet Wearable Device Dataset (Empatica E4)\n")
        f.write("  Sessions: STRESS only (aerobic/anaerobic excluded)\n")
        f.write(f"  Subjects: {len(all_subjects)} total — all appear in both train and test\n\n")
        f.write("Split Strategy\n")
        f.write("  Type: within-subject mixed split (NOT subject-wise)\n")
        f.write("  For each subject, 70% of windows -> train, 30% -> test (random)\n")
        f.write("  Rationale: simulates a wristband that calibrates on the user's own\n")
        f.write("  baseline data before making predictions. Every subject has some\n")
        f.write("  representation in training, so the model learns individual baselines.\n\n")
        f.write("Label Mapping\n")
        f.write("  1 = vibrate    (stress task active)\n")
        f.write("  0 = no vibrate (rest / baseline period)\n\n")
        f.write("Windowing\n")
        f.write(f"  Window : {WINDOW_S} s, Step : {STEP_S} s (50% overlap)\n\n")
        f.write(f"Features ({len(FEATURE_NAMES)} total)\n")
        f.write("  EDA  : mean, std, slope, range, scr_count, deriv_mean, delta\n")
        f.write("  HR   : mean, std, min, max, delta\n")
        f.write("  IBI  : mean, std, RMSSD, SDNN, pNN50, LF, HF, LF/HF\n")
        f.write("  ACC  : mag_mean, mag_std, var_x, var_y, var_z, mean_jerk\n\n")
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
        description="Train personalized (mixed-split) stress classifier."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--data-dir",
        default=os.path.join(script_dir, "data", "PhysioNet"),
        help="Path to the PhysioNet dataset root (default: ./data/PhysioNet)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(script_dir, "outputs_personalized"),
        help="Directory for outputs (default: ./outputs_personalized)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    t_start = time.perf_counter()

    # 1. Load data
    print("\n[1/5] Loading STRESS session data...")
    subjects = load_all_stress_subjects(data_dir, verbose=True)
    if not subjects:
        print("ERROR: No subjects loaded.", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(subjects)} subjects.")

    # 2. Feature extraction
    print("\n[2/5] Extracting windowed features...")
    X, y, groups = build_dataset(subjects, verbose=True)
    if not X:
        print("ERROR: Feature matrix is empty.", file=sys.stderr)
        sys.exit(1)

    # 3. Within-subject mixed split
    print("\n[3/5] Mixed split (70/30 within each subject)...")
    X_train, y_train, groups_train, X_test, y_test, groups_test, all_subjects = \
        mixed_split(X, y, groups)

    n_train_stress = sum(y_train)
    n_train_rest = len(y_train) - n_train_stress
    n_test_stress = sum(y_test)
    n_test_rest = len(y_test) - n_test_stress

    print(f"  Train: {len(y_train)} windows  (stress={n_train_stress}, rest={n_train_rest})")
    print(f"  Test : {len(y_test)} windows  (stress={n_test_stress}, rest={n_test_rest})")
    print(f"  All {len(all_subjects)} subjects appear in both train and test.")

    # 4. Train
    print("\n[4/5] Training XGBoost classifier...")
    t_train = time.perf_counter()
    model, col_means, threshold = train_xgboost(X_train, y_train, groups_train)
    print(f"  Training time: {(time.perf_counter() - t_train):.2f}s")

    # Save model
    model_path = os.path.join(output_dir, "model.json")
    meta_path = os.path.join(output_dir, "model_meta.json")
    os.makedirs(output_dir, exist_ok=True)
    save_model(model, col_means, model_path, meta_path, threshold)
    print(f"  Model saved: {model_path}")

    # 5. Evaluate
    print("\n[5/5] Evaluating on held-out test windows...")
    metrics, preds, probs = evaluate(model, col_means, X_test, y_test, threshold)

    metrics["split_strategy"] = "within-subject mixed 70/30"
    metrics["subjects"] = all_subjects
    metrics["n_train"] = len(y_train)
    metrics["n_train_stress"] = n_train_stress
    metrics["n_train_rest"] = n_train_rest

    print_metrics(metrics)
    save_outputs(output_dir, metrics, preds, probs, groups_test, y_test,
                 all_subjects)

    print(f"  Total pipeline time: {(time.perf_counter() - t_start):.1f}s\n")


if __name__ == "__main__":
    main()
