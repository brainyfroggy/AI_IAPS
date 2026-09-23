#!/usr/bin/env python3
"""Generic Bo et al.-style within/cross source decoding bar charts.

Supersedes plot_random10fold_bo_style.py and plot_random10fold_bo_style_no_sub30.py by
taking the group stats CSV, titles, axis range and optional subject exclusions as CLI args.

Input is either a precomputed group_roi_stats.csv (columns: contrast, roi, mean_accuracy,
sem) or, with --all-subject-csv, a per-subject all_subject_results.csv from which group
mean/SEM are recomputed after applying --exclude-subjects.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

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
               title: str, out_file: Path, y_low: float, y_high: float, step: float,
               sig_df: pd.DataFrame | None = None, sig_col: str = "sig_fdr_q05",
               method_note: str | None = None) -> None:
    x = range(len(ROI_ORDER))
    width = min(0.18, 0.80 / len(comp_order))
    fig, ax = plt.subplots(figsize=(max(18, len(ROI_ORDER) * 1.15), 7.6))

    for ci, comp in enumerate(comp_order):
        sub = df[df["contrast"] == comp].set_index("roi").loc[ROI_ORDER]
        offset = (ci - (len(comp_order) - 1) / 2) * width
        xs = [xi + offset for xi in x]
        means = sub["mean_accuracy"].to_numpy()
        sems = sub["sem"].to_numpy()
        ax.bar(xs, means, width,
               yerr=sems, color=colors[comp], label=labels[comp],
               hatch=hatch_for(comp), capsize=3, alpha=0.9, edgecolor="black",
               linewidth=0.4, zorder=3)

        if sig_df is not None:
            sig_sub = sig_df[sig_df["contrast"] == comp].set_index("roi").loc[ROI_ORDER]
            is_sig = sig_sub[sig_col].to_numpy()
            marker_y = means + sems + (y_high - y_low) * 0.015
            for xi, yi, s in zip(xs, marker_y, is_sig):
                if s:
                    ax.plot(xi, yi, marker="*", markersize=9, color="#1a1a1a", zorder=4,
                           markeredgewidth=0)

    add_guide_lines(ax, y_low, y_high, step)
    ax.set_ylabel("Accuracy", fontsize=24)
    ax.set_xlabel("Region of Interest", fontsize=24)
    ax.set_title(title, fontsize=24, fontweight="bold", pad=22)
    decimals = 0 if abs(round(step * 100) - step * 100) < 1e-9 else 1
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=decimals))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ROI_ORDER, rotation=45, ha="right", fontsize=18)
    ax.set_ylim([y_low, y_high])
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if sig_df is not None:
        star_handle = plt.Line2D([0], [0], marker="*", color="#1a1a1a", linestyle="None",
                                 markersize=9, label="significant (FDR q<0.05)")
        legend_handles.append(star_handle)
        legend_labels.append("significant (FDR q<0.05)")
    legend = ax.legend(legend_handles, legend_labels, fontsize=13, loc="upper left",
                       frameon=True, facecolor="white", framealpha=0.9, edgecolor="gray")
    legend.get_frame().set_linewidth(0.8)

    bottom_margin = 0.0
    if method_note:
        wrapped = "\n".join(textwrap.wrap(method_note, width=185))
        n_lines = wrapped.count("\n") + 1
        fig.text(0.005, 0.005, wrapped, fontsize=8.5, color="#444444", ha="left", va="bottom",
                 linespacing=1.4)
        bottom_margin = 0.05 + 0.018 * n_lines

    plt.tight_layout(rect=(0, bottom_margin, 1, 0.94))
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_file}")


def group_from_subjects(path: Path, exclude: set[str]) -> tuple[pd.DataFrame, int]:
    all_df = pd.read_csv(path)
    n_before = all_df["subject"].nunique()
    all_df = all_df[~all_df["subject"].isin(exclude)]
    n_after = all_df["subject"].nunique()
    print(f"subjects: {n_before} -> {n_after} (excluded: {sorted(exclude) or 'none'})")
    rows = []
    for contrast in WITHIN_ORDER + CROSS_ORDER:
        sub = all_df[all_df["contrast"] == contrast]
        for roi in ROI_ORDER:
            acc = sub[sub["roi"] == roi]["accuracy"]
            rows.append({"contrast": contrast, "roi": roi, "n_subjects": len(acc),
                         "mean_accuracy": acc.mean(),
                         "sem": acc.std(ddof=1) / (len(acc) ** 0.5)})
    return pd.DataFrame(rows), n_after


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--group-csv", type=Path, help="precomputed group_roi_stats.csv")
    src.add_argument("--all-subject-csv", type=Path, help="all_subject_results.csv to re-aggregate")
    ap.add_argument("--exclude-subjects", nargs="*", default=[])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--title", required=True, help="e.g. 'SPM LS-A, StratifiedKFold(4)x20, n=28'")
    ap.add_argument("--title-within", default=None,
                    help="overrides --title for the within-source panel only")
    ap.add_argument("--title-cross", default=None,
                    help="overrides --title for the cross-source panel only")
    ap.add_argument("--y-low", type=float, default=0.45)
    ap.add_argument("--y-high", type=float, default=0.60)
    ap.add_argument("--step", type=float, default=0.025)
    ap.add_argument("--sig-csv", type=Path, default=None,
                    help="per-cell significance CSV (contrast, roi, + --sig-col boolean)")
    ap.add_argument("--sig-col", default="sig_fdr_q05")
    ap.add_argument("--method-note", default=None, help="small footnote text describing the significance method")
    args = ap.parse_args()

    if args.group_csv:
        df = pd.read_csv(args.group_csv)
        n = int(df["n_subjects"].iloc[0]) if "n_subjects" in df.columns else -1
        print(f"loaded {args.group_csv} (n_subjects={n})")
    else:
        df, n = group_from_subjects(args.all_subject_csv, set(args.exclude_subjects))

    vals = df["mean_accuracy"]
    lo, hi = (vals - df["sem"]).min(), (vals + df["sem"]).max()
    print(f"data range (mean): {vals.min()*100:.1f}% - {vals.max()*100:.1f}%")
    print(f"data range (+/-SEM): {lo*100:.1f}% - {hi*100:.1f}%")
    if lo < args.y_low or hi > args.y_high:
        print(f"WARNING: data extends past axis [{args.y_low*100:.0f}%, {args.y_high*100:.0f}%] "
              f"- bars/error bars will be clipped")

    sig_df = pd.read_csv(args.sig_csv) if args.sig_csv else None
    if sig_df is not None:
        n_sig = int(sig_df[args.sig_col].sum())
        print(f"loaded {args.sig_csv} ({n_sig}/{len(sig_df)} significant at {args.sig_col})")

    title_within = args.title_within or args.title
    title_cross = args.title_cross or args.title
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_panel(df, WITHIN_ORDER, WITHIN_COLORS, WITHIN_LABELS,
               f"Within Source Decoding ({title_within})",
               args.out_dir / f"within_{args.out_prefix}.png", args.y_low, args.y_high, args.step,
               sig_df=sig_df, sig_col=args.sig_col, method_note=args.method_note)
    plot_panel(df, CROSS_ORDER, CROSS_COLORS, CROSS_LABELS,
               f"Cross Source Decoding ({title_cross})",
               args.out_dir / f"cross_{args.out_prefix}.png", args.y_low, args.y_high, args.step,
               sig_df=sig_df, sig_col=args.sig_col, method_note=args.method_note)


if __name__ == "__main__":
    main()
