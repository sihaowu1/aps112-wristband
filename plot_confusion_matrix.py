#!/usr/bin/env python3
"""
plot_confusion_matrix.py - Generate a confusion matrix visualization.

Reads metrics.json for TP/TN/FP/FN and produces a green-themed confusion
matrix image saved to outputs/confusion_matrix.png.

Usage:
    python3 plot_confusion_matrix.py [--output-dir outputs]
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
    parser = argparse.ArgumentParser(description="Plot confusion matrix from metrics.json.")
    parser.add_argument("--output-dir", default=os.path.join(script_dir, "outputs"))
    args = parser.parse_args()

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print("ERROR: metrics.json not found. Run train.py first.", file=sys.stderr)
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    tp = metrics["tp"]
    tn = metrics["tn"]
    fp = metrics["fp"]
    fn = metrics["fn"]

    # Matrix: rows = actual, cols = predicted
    # Row 0 = Rest (actual), Row 1 = Stress (actual)
    # Col 0 = Rest (pred),   Col 1 = Stress (pred)
    cm = np.array([[tn, fp],
                    [fn, tp]])

    # Green colormap from white to #196b24
    green = "#196b24"
    cmap = mcolors.LinearSegmentedColormap.from_list("green_cm", ["#ffffff", green])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    # Annotate cells with count and percentage
    n_total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / n_total * 100
            # Use white text on dark cells, dark text on light cells
            color = "white" if count > n_total * 0.25 else "#1a1a1a"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color=color)

    labels = ["Rest\n(Do Not Vibrate)", "Stress\n(Vibrate)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix — Test Set (1 504 windows)", fontsize=13, fontweight="bold", pad=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Count", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "confusion_matrix.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
