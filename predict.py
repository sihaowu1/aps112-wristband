#!/usr/bin/env python3
"""
predict.py - Classify stress vs rest for one 20 s window using a trained model.

Requires:
  - outputs/model.json and outputs/model_meta.json from train.py
  - PhysioNet-format CSVs under a STRESS session folder (or --subject-dir)
  - Per-subject baseline: rest_intervals from tags.csv, or --rest-intervals JSON

Example (dataset subject):
    python predict.py --subject S01 --t-start 180 --t-end 200

Example (explicit folder + custom baseline):
    python predict.py --subject-dir path/to/session --rest-intervals "[[0,120]]" --t-start 300 --t-end 320
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_subject, load_subject_with_intervals, resolve_stress_subject_dir
from src.features import WINDOW_S
from src.inference import load_trained_model, predict_stress_window


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Predict stressed vs not for one window (binary vibrate / no vibrate)."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(script_dir, "data", "PhysioNet"),
        help="PhysioNet root (default: ./data/PhysioNet)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(script_dir, "outputs"),
        help="Directory with model.json and model_meta.json (default: ./outputs)",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Subject id (e.g. S01, f02). Used with --data-dir to locate STRESS session.",
    )
    parser.add_argument(
        "--subject-dir",
        default=None,
        help="Path to a session folder containing EDA.csv, HR.csv, ACC.csv, IBI.csv (overrides --subject).",
    )
    parser.add_argument(
        "--subject-id",
        default="upload",
        help="Label for logging when using --subject-dir (default: upload).",
    )
    parser.add_argument(
        "--t-start",
        type=float,
        required=True,
        help="Window start time (seconds from session start).",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        required=True,
        help=f"Window end time (seconds from start). Must span exactly {WINDOW_S} s.",
    )
    parser.add_argument(
        "--rest-intervals",
        default=None,
        help='Optional JSON list of [start,end] pairs in seconds, e.g. "[[0,60],[120,180]]" '
        "to override tags.csv for baseline normalization.",
    )
    args = parser.parse_args()

    if args.subject_dir:
        subject_dir = os.path.abspath(args.subject_dir)
        subject_id = args.subject_id
    elif args.subject:
        subject_dir = resolve_stress_subject_dir(args.data_dir, args.subject)
        subject_id = args.subject
    else:
        print("ERROR: Provide --subject (e.g. S01) or --subject-dir /path/to/session", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(subject_dir):
        print(f"ERROR: Not a directory: {subject_dir}", file=sys.stderr)
        sys.exit(1)

    rest_iv = None
    if args.rest_intervals:
        try:
            rest_iv = json.loads(args.rest_intervals)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid --rest-intervals JSON: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading session: {subject_dir} (id={subject_id})")
    if rest_iv is not None:
        subj = load_subject_with_intervals(subject_dir, subject_id, rest_intervals=rest_iv)
    else:
        subj = load_subject(subject_dir, subject_id)

    print(f"Loading model from: {args.output_dir}")
    model, col_means, threshold, _profiles = load_trained_model(args.output_dir)

    out = predict_stress_window(model, col_means, subj, args.t_start, args.t_end, threshold)

    print()
    print("=" * 50)
    print("  PREDICTION")
    print("=" * 50)
    print(f"  Window        : [{args.t_start:.1f}, {args.t_end:.1f}] s")
    print(f"  P(stress)     : {out['pred_prob']:.4f}")
    print(f"  Decision      : {out['pred_class']}  ({out['feedback']})")
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()
