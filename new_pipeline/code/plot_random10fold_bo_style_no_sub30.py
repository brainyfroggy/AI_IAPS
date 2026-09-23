#!/usr/bin/env python3
"""Same as plot_random10fold_bo_style.py, but recomputes group mean/SEM per ROI x contrast
after excluding Sub30, then plots with the identical Bo et al.-style layout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

NP = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")
ALL_SUBJECT_CSV = NP / "roi_decoding_random10fold" / "group" / "all_subject_results.csv"
OUT_DIR = NP / "roi_decoding_random10fold" / "group"
EXCLUDE_SUBJECTS = {"Sub30"}

ROI_ORDER = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "V3a", "V3b",
             "IPS", "LO1", "LO2", "hMT", "VO1", "VO2", "PHC1", "PHC2"]

WITHIN_ORDER = [
    "within_natural_pleasant_vs_neutral",
    "within_ai_pleasant_vs_neutral",
    "within_natural_unpleasant_vs_neutral",
    "within_ai_unpleasant_vs_neutral",
]
CROSS_ORDER = [
    "train_natural_pleasant_vs_neutral_test_ai",
    "train_ai_pleasant_vs_neutral_test_natural",
    "train_natural_unpleasant_vs_neutral_test_ai",
    "train_ai_unpleasant_vs_neutral_test_natural",
]
WITHIN_LABELS = {
    "within_natural_pleasant_vs_neutral": "PL vs Nt, Natural",
    "within_ai_pleasant_vs_neutral": "PL vs Nt, AI",
    "within_natural_unpleasant_vs_neutral": "UP vs Nt, Natural",
    "within_ai_unpleasant_vs_neutral": "UP vs Nt, AI",
}
CROSS_LABELS = {
    "train_natural_pleasant_vs_neutral_test_ai": "PL vs Nt, trained on NA and tested on AI",
    "train_ai_pleasant_vs_neutral_test_natural": "PL vs Nt, trained on AI and tested on NA",
    "train_natural_unpleasant_vs_neutral_test_ai": "UP vs Nt, trained on NA and tested on AI",
    "train_ai_unpleasant_vs_neutral_test_natural": "UP vs Nt, trained on AI and tested on NA",
}
WITHIN_COLORS = {
    "within_natural_pleasant_vs_neutral": "#1f4e79",
    "within_ai_pleasant_vs_neutral": "#8ecae6",
    "within_natural_unpleasant_vs_neutral": "#b45f06",
    "within_ai_unpleasant_vs_neutral": "#f6b26b",
}
CROSS_COLORS = {
    "train_natural_pleasant_vs_neutral_test_ai": "#1b7837",
    "train_ai_pleasant_vs_neutral_test_natural": "#a6dba0",
    "train_natural_unpleasant_vs_neutral_test_ai": "#8b0000",
    "train_ai_unpleasant_vs_neutral_test_natural": "#f4a3a3",
}


def hatch_for(comp: str) -> str | None:
    return "///" if ("_ai_" in comp or comp.endswith("test_natural")) else None


def add_guide_lines(ax, low: float, high: float, step: float) -> None:
    v = low
    while v <= high + 1e-9:
        is_chance = abs(v - 0.50) < 1e-9
        ax.axhline(y=v, color="#4A4A4A" if is_chance else "#D0D0D0",
                   linestyle="--", linewidth=2.0 if is_chance else 0.8,
                   alpha=0.95 if is_chance else 0.75, zorder=0)
        v += step


def plot_panel(df: pd.DataFrame, comp_order: list[str], colors: dict, labels: dict,
               title: str, out_file: Path, y_low: float, y_high: float, step: float) -> None:
    rois = ROI_ORDER
    x = range(len(rois))
    width = min(0.18, 0.80 / len(comp_order))
    fig, ax = plt.subplots(figsize=(max(18, len(rois) * 1.15), 7))

    for ci, comp in enumerate(comp_order):
        sub = df[df["contrast"] == comp].set_index("roi").loc[rois]
        means = sub["mean_accuracy"].to_numpy()
        sems = sub["sem"].to_numpy()
        offset = (ci - (len(comp_order) - 1) / 2) * width
        xs = [xi + offset for xi in x]
        ax.bar(xs, means, width, yerr=sems, color=colors[comp], label=labels[comp],
               hatch=hatch_for(comp), capsize=3, alpha=0.9, edgecolor="black",
               linewidth=0.4, zorder=3)

    add_guide_lines(ax, y_low, y_high, step)
    ax.set_ylabel("Accuracy", fontsize=24)
    ax.set_xlabel("Region of Interest", fontsize=24)
    ax.set_title(title, fontsize=26, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xticks(list(x))
    ax.set_xticklabels(rois, rotation=45, ha="right", fontsize=18)
    ax.set_ylim([y_low, y_high])
    legend = ax.legend(fontsize=15, loc="upper left", frameon=True, facecolor="white",
                       framealpha=0.9, edgecolor="gray")
    legend.get_frame().set_linewidth(0.8)
    plt.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_file}")


def main() -> None:
    all_df = pd.read_csv(ALL_SUBJECT_CSV)
    n_before = all_df["subject"].nunique()
    all_df = all_df[~all_df["subject"].isin(EXCLUDE_SUBJECTS)]
    n_after = all_df["subject"].nunique()
    print(f"subjects: {n_before} -> {n_after} (excluded {EXCLUDE_SUBJECTS})")

    rows = []
    for contrast in WITHIN_ORDER + CROSS_ORDER:
        sub = all_df[all_df["contrast"] == contrast]
        for roi in ROI_ORDER:
            acc = sub[sub["roi"] == roi]["accuracy"]
            rows.append({
                "contrast": contrast, "roi": roi, "n_subjects": len(acc),
                "mean_accuracy": acc.mean(), "sem": acc.std(ddof=1) / (len(acc) ** 0.5),
            })
    df = pd.DataFrame(rows)

    all_vals = df["mean_accuracy"].to_numpy()
    print(f"data range: {all_vals.min()*100:.1f}% - {all_vals.max()*100:.1f}%")

    y_low, y_high, step = 0.45, 0.60, 0.025

    plot_panel(df, WITHIN_ORDER, WITHIN_COLORS, WITHIN_LABELS,
              f"Within Source Decoding (GLMsingle, Random 10-fold x30, n={n_after}, no Sub30)",
              OUT_DIR / "within_kastner_wang_random10fold_glmsingle_no_sub30.png", y_low, y_high, step)
    plot_panel(df, CROSS_ORDER, CROSS_COLORS, CROSS_LABELS,
              f"Cross Source Decoding (GLMsingle, Random 10-fold x30, n={n_after}, no Sub30)",
              OUT_DIR / "cross_kastner_wang_random10fold_glmsingle_no_sub30.png", y_low, y_high, step)


if __name__ == "__main__":
    main()
