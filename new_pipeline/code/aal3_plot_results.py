#!/usr/bin/env python3
"""Phase 4 of the whole-brain AAL3 cross-source decoding plan.

Per contrast (4), at the 8mm headline smoothing level:
  - a full ranked horizontal bar chart across all 154 regions, FDR-starred
  - a whole-brain accuracy-map NIfTI (each region painted with its group-mean
    accuracy, on the native beta grid)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/aal3_crosssource")
CROSS_CONTRASTS = (
    "train_natural_pleasant_vs_neutral_test_ai",
    "train_ai_pleasant_vs_neutral_test_natural",
    "train_natural_unpleasant_vs_neutral_test_ai",
    "train_ai_unpleasant_vs_neutral_test_natural",
)
CONTRAST_TITLES = {
    "train_natural_pleasant_vs_neutral_test_ai": "Pleasant vs Neutral, trained on Natural, tested on AI",
    "train_ai_pleasant_vs_neutral_test_natural": "Pleasant vs Neutral, trained on AI, tested on Natural",
    "train_natural_unpleasant_vs_neutral_test_ai": "Unpleasant vs Neutral, trained on Natural, tested on AI",
    "train_ai_unpleasant_vs_neutral_test_natural": "Unpleasant vs Neutral, trained on AI, tested on Natural",
}
BAR_COLOR = "#1f4e79"
SIG_COLOR = "#a6491f"


def plot_ranked_bar(df: pd.DataFrame, contrast: str, out_file: Path) -> None:
    sub = df[df["contrast"] == contrast].sort_values("mean_accuracy", ascending=True).reset_index(drop=True)
    n = len(sub)
    fig, ax = plt.subplots(figsize=(9, max(14, n * 0.135)))
    y = np.arange(n)
    colors = [SIG_COLOR if s else BAR_COLOR for s in sub["significant_fdr05"]]
    ax.barh(y, sub["mean_accuracy"], xerr=sub["sem"], color=colors, height=0.72,
            capsize=1.5, edgecolor="none", alpha=0.9, zorder=3)
    ax.axvline(0.5, color="#4A4A4A", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["roi"], fontsize=6.2)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title(f"AAL3 whole-brain cross-source decoding, n=30, 8mm\n{CONTRAST_TITLES[contrast]}",
                 fontsize=11.5, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(0.35, 0.68)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.6, zorder=0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BAR_COLOR, label="not significant"),
        plt.Rectangle((0, 0), 1, 1, color=SIG_COLOR, label="significant (FDR q<0.05)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True, facecolor="white", framealpha=0.9)
    fig.text(0.01, 0.005,
              "One-sample t-test vs 50% chance, BH-FDR corrected across 154 AAL3 regions within this contrast. "
              "Cross-source: single train-on-all-200/test-on-all-200 split, no cross-validation.",
              fontsize=7.5, color="#444444")
    plt.tight_layout(rect=(0, 0.012, 1, 1))
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_file}")


def build_accuracy_map(df: pd.DataFrame, contrast: str, atlas_native_labels: Path,
                        manifest: pd.DataFrame, out_file: Path) -> None:
    atlas_img = nib.load(atlas_native_labels)
    vals = np.rint(np.asarray(atlas_img.dataobj)).astype(np.int32)
    acc_map = np.full(vals.shape, np.nan, dtype=np.float32)

    sub = df[df["contrast"] == contrast].set_index("roi")
    name_to_id = dict(zip(manifest["roi_name"], manifest["id"].astype(int)))
    for roi_name, row in sub.iterrows():
        rid = name_to_id[roi_name]
        acc_map[vals == rid] = row["mean_accuracy"]

    header = atlas_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(acc_map, atlas_img.affine, header), out_file)
    print(f"saved {out_file}")


def main() -> None:
    manifest = pd.read_csv(ROOT / "phase0_manifest" / "roi_manifest.csv")
    manifest = manifest[manifest["kept"]]
    atlas_native_labels = ROOT / "phase0_manifest" / "aal3_native_labels.nii.gz"

    group_df = pd.read_csv(ROOT / "smooth08" / "group" / "group_aal3_stats.csv")

    charts_dir = ROOT / "smooth08" / "group" / "charts"
    maps_dir = ROOT / "smooth08" / "group" / "accuracy_maps"
    charts_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    for contrast in CROSS_CONTRASTS:
        plot_ranked_bar(group_df, contrast, charts_dir / f"aal3_ranked_{contrast}.png")
        build_accuracy_map(group_df, contrast, atlas_native_labels, manifest,
                            maps_dir / f"aal3_accuracy_map_{contrast}.nii.gz")

    print("AAL3_PLOTS_COMPLETE")


if __name__ == "__main__":
    main()
