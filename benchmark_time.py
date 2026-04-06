#!/usr/bin/env python3
"""
benchmark_time.py - Measure inference processing time statistics.

Loads the trained model and test data, then runs inference N times to compute
mean and standard deviation of processing time.

Usage:
    python3 benchmark_time.py [--runs 1000] [--data-dir data/PhysioNet] [--output-dir outputs]

Outputs:
    outputs/processing_time_stats.json
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all_stress_subjects
from src.features import build_dataset
from src.model import load_model, _apply_means, FEATURE_NAMES
from train import subject_split

import xgboost as xgb


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Benchmark inference processing time.")
    parser.add_argument("--runs", type=int, default=1000, help="Number of inference runs (default: 1000)")
    parser.add_argument("--data-dir", default=os.path.join(script_dir, "data", "PhysioNet"))
    parser.add_argument("--output-dir", default=os.path.join(script_dir, "outputs"))
    args = parser.parse_args()

    model_path = os.path.join(args.output_dir, "model.json")
    meta_path = os.path.join(args.output_dir, "model_meta.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print("ERROR: Model not found. Run train.py first.", file=sys.stderr)
        sys.exit(1)

    # Load model
    model, col_means, threshold = load_model(model_path, meta_path)

    # Load test data using same split as training
    print("Loading data and rebuilding test split...")
    subjects = load_all_stress_subjects(args.data_dir, verbose=False)
    X, y, groups = build_dataset(subjects, verbose=False)
    _, _, X_test, y_test, _, _, _ = subject_split(X, y, groups)

    X_imp = _apply_means(X_test, col_means)
    dtest = xgb.DMatrix(X_imp, feature_names=FEATURE_NAMES)

    n_windows = len(y_test)
    n_runs = args.runs
    print(f"Running {n_runs} inference passes over {n_windows} test windows...")

    # Warmup
    for _ in range(5):
        model.predict(dtest)

    # Timed runs
    times_ms = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dtest)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    mean_ms = sum(times_ms) / len(times_ms)
    variance = sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)
    std_ms = math.sqrt(variance)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    per_window_ms = mean_ms / n_windows if n_windows else 0

    results = {
        "n_runs": n_runs,
        "n_test_windows": n_windows,
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "min_ms": round(min_ms, 4),
        "max_ms": round(max_ms, 4),
        "per_window_mean_ms": round(per_window_ms, 6),
        "raw_times_ms": [round(t, 4) for t in times_ms],
    }

    out_path = os.path.join(args.output_dir, "processing_time_stats.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessing Time Benchmark ({n_runs} runs, {n_windows} windows)")
    print(f"  Mean : {mean_ms:.4f} ms")
    print(f"  Std  : {std_ms:.4f} ms")
    print(f"  Min  : {min_ms:.4f} ms")
    print(f"  Max  : {max_ms:.4f} ms")
    print(f"  Per-window mean: {per_window_ms:.6f} ms")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
