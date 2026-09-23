#!/usr/bin/env python3
"""Plot per-image evoked beta magnitude, broken down by source and valence.

One figure per ROI set. Each figure has two panels (Natural, AI); each panel
shows that source's 60 unique images on the x-axis, grouped into three
valence blocks of 20. Points are the across-subject mean of the image's beta
(after averaging that image's 5 repetitions within each subject), with
between-subject SEM error bars (n=30).

Companion summary figure: cell means (source x valence) with between-subject
SEM, which is the actual test of "do unpleasant images evoke less response".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NEW_PIPELINE = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")

VALENCE_ORDER = ["pleasant", "neutral", "unpleasant"]
VALENCE_COLORS = {"pleasant": "#1f4e79", "neutral": "#7a7a7a", "unpleasant": "#b45f06"}
SOURCE_TITLES = {"natural": "Natural images", "ai": "AI-generated images"}
ROI_SET_TITLES = {
    "visual_kastner": "Visual cortex (Kastner/Wang 17-ROI union, 8 mm)",
    "wholebrain_aal3": "Whole-brain gray matter (AAL3, 154 regions, 8 mm)",
}


def plot_per_image(image_df: pd.DataFrame, roi_set: str, out_file: Path, n_subjects: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharey=True)

    for ax, source in zip(axes, ["natural", "ai"]):
        sub = image_df[image_df["source"] == source]
        x_cursor = 0
        block_centers = []
        for valence in VALENCE_ORDER:
            block = sub[sub["valence"] == valence].sort_values("image_id").reset_index(drop=True)
            xs = np.arange(x_cursor, x_cursor + len(block))
            ax.errorbar(xs, block["mean_beta"], yerr=block["sem_beta"],
                        fmt="o", markersize=4.5, color=VALENCE_COLORS[valence],
                        ecolor=VALENCE_COLORS[valence], elinewidth=1.0, capsize=2,
                        alpha=0.9, label=valence.capitalize(), zorder=3)
            # block mean line
            ax.hlines(block["mean_beta"].mean(), xs[0] - 0.5, xs[-1] + 0.5,
                      color=VALENCE_COLORS[valence], linewidth=2.0, linestyle="-", alpha=0.55, zorder=2)
            block_centers.append((xs.mean(), valence, block["mean_beta"].mean()))
            x_cursor += len(block)
            if valence != VALENCE_ORDER[-1]:
                ax.axvline(x_cursor - 0.5, color="#cccccc", linewidth=1.0, linestyle="--", zorder=1)

        ax.axhline(0, color="#4A4A4A", linewidth=1.0, linestyle="-", alpha=0.7, zorder=1)
        ax.set_title(SOURCE_TITLES[source], fontsize=13, fontweight="bold")
        ax.set_ylabel("Mean beta (% signal change)", fontsize=11)
        ax.set_xlim(-1, x_cursor)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.7, zorder=0)
        ax.set_xticks([c for c, _, _ in block_centers])
        ax.set_xticklabels([f"{v.capitalize()}\n(20 images)" for _, v, _ in block_centers], fontsize=11)
        for c, v, m in block_centers:
            ax.annotate(f"block mean {m:+.3f}", xy=(c, m), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=VALENCE_COLORS[v], fontweight="bold")

    axes[0].legend(loc="upper right", fontsize=10, frameon=True, facecolor="white", framealpha=0.9, ncol=3)
    axes[1].set_xlabel("Unique image (60 per source: 20 per valence)", fontsize=11)
    fig.suptitle(f"Per-image evoked response by source and valence — {ROI_SET_TITLES[roi_set]}",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.005, 0.005,
             f"Each point = one unique image. Within each subject the image's 5 repetitions are averaged; "
             f"points are the mean across {n_subjects} subjects and error bars are between-subject SEM (n={n_subjects}). "
             f"GLMsingle Type-D single-trial betas, 8 mm smoothed, averaged across the ROI set's voxels.",
             fontsize=8.5, color="#444444")
    plt.tight_layout(rect=(0, 0.035, 1, 0.955))
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_file}")


def plot_cell_summary(cell_df: pd.DataFrame, roi_set: str, out_file: Path, n_subjects: int) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    width = 0.35
    x = np.arange(len(VALENCE_ORDER))
    for i, source in enumerate(["natural", "ai"]):
        sub = cell_df[cell_df["source"] == source].set_index("valence").loc[VALENCE_ORDER]
        offset = (i - 0.5) * width
        ax.bar(x + offset, sub["mean_beta"], width, yerr=sub["sem_beta"],
               color=["#1f4e79" if i == 0 else "#8ecae6"] * 3,
               hatch=None if i == 0 else "///",
               capsize=4, edgecolor="black", linewidth=0.5,
               label=SOURCE_TITLES[source], alpha=0.9, zorder=3)
    ax.axhline(0, color="#4A4A4A", linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([v.capitalize() for v in VALENCE_ORDER], fontsize=12)
    ax.set_ylabel("Mean beta (% signal change)", fontsize=12)
    ax.set_title(f"Evoked response by valence\n{ROI_SET_TITLES[roi_set]}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, frameon=True)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.7, zorder=0)
    fig.text(0.005, 0.005, f"Error bars: between-subject SEM (n={n_subjects}).",
             fontsize=8.5, color="#444444")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_file}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=NEW_PIPELINE / "trial_beta_magnitude")
    ap.add_argument("--roi-sets", nargs="+", default=["visual_kastner", "wholebrain_aal3"])
    args = ap.parse_args()

    for roi_set in args.roi_sets:
        ROI_SET_TITLES.setdefault(roi_set, roi_set.replace("_", " "))
        image_df = pd.read_csv(args.in_dir / f"{roi_set}_image_level_betas.csv")
        cell_df = pd.read_csv(args.in_dir / f"{roi_set}_cell_summary.csv")
        n_subjects = int(image_df["n_subjects"].iloc[0])
        plot_per_image(image_df, roi_set, args.in_dir / f"{roi_set}_per_image_betas.png", n_subjects)
        plot_cell_summary(cell_df, roi_set, args.in_dir / f"{roi_set}_cell_summary.png", n_subjects)

    print("TRIAL_BETA_PLOTS_COMPLETE")


if __name__ == "__main__":
    main()
