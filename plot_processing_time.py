#!/usr/bin/env python3
"""
plot_processing_time.py - Visualize inference processing time distribution.

Reads processing_time_stats.json (with raw_times_ms) and produces a histogram
with mean and +/- 1 std deviation markers.

Usage:
    python3 plot_processing_time.py [--output-dir outputs]
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Plot processing time distribution.")
    parser.add_argument("--output-dir", default=os.path.join(script_dir, "outputs"))
    args = parser.parse_args()

    stats_path = os.path.join(args.output_dir, "processing_time_stats.json")
    if not os.path.exists(stats_path):
        print("ERROR: processing_time_stats.json not found. Run benchmark_time.py first.", file=sys.stderr)
        sys.exit(1)

    with open(stats_path) as f:
        stats = json.load(f)

    times = stats["raw_times_ms"]
    mean = stats["mean_ms"]
    std = stats["std_ms"]
    n_runs = stats["n_runs"]
    n_windows = stats["n_test_windows"]

    green = "#196b24"
    green_light = "#2d9e3f"

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(times, bins=50, color=green, edgecolor="white", linewidth=0.5, alpha=0.9)

    # Mean line
    ax.axvline(mean, color="#0d3d12", linewidth=2, linestyle="-", label=f"Mean: {mean:.2f} ms")
    # +/- 1 std
    ax.axvline(mean - std, color=green_light, linewidth=1.5, linestyle="--", label=f"\u00b11 Std: {std:.2f} ms")
    ax.axvline(mean + std, color=green_light, linewidth=1.5, linestyle="--")

    ax.set_xlabel("Processing Time (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title(f"Inference Processing Time Distribution ({n_runs} runs, {n_windows} windows)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "processing_time.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
