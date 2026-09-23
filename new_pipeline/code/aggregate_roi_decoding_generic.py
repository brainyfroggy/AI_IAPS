#!/usr/bin/env python3
"""Generic version of aggregate_roi_decoding_fixedhrf_full.py -- takes ROOT as a CLI
arg instead of hardcoding it, so the same aggregation logic (t-test vs chance +
BH-FDR across 17 ROIs within each contrast) can run against any of the random10fold
decode trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROI_ORDER = (
    "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "V3a", "V3b",
    "IPS", "LO1", "LO2", "hMT", "VO1", "VO2", "PHC1", "PHC2",
)
CONTRAST_ORDER = (
    "within_natural_pleasant_vs_neutral",
    "within_ai_pleasant_vs_neutral",
    "within_natural_unpleasant_vs_neutral",
    "within_ai_unpleasant_vs_neutral",
    "train_natural_pleasant_vs_neutral_test_ai",
    "train_ai_pleasant_vs_neutral_test_natural",
    "train_natural_unpleasant_vs_neutral_test_ai",
    "train_ai_unpleasant_vs_neutral_test_natural",
)
WITHIN = CONTRAST_ORDER[:4]
CHANCE = 0.5


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    out = root / "group"
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for sub_dir in sorted(root.glob("sub-*")):
        f = sub_dir / "subject_results.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv(out / "all_subject_results.csv", index=False)
    n_subjects = all_results["subject"].nunique()
    print(f"aggregated {len(all_results)} rows across {n_subjects} subjects")

    rows = []
    for contrast in CONTRAST_ORDER:
        sub_frame = all_results[all_results["contrast"] == contrast]
        pvals = []
        cell_data = []
        for roi in ROI_ORDER:
            acc = sub_frame[sub_frame["roi"] == roi].sort_values("subject")["accuracy"].to_numpy()
            if len(acc) != n_subjects:
                raise RuntimeError(f"{contrast}/{roi}: expected {n_subjects} subjects, got {len(acc)}")
            t_stat, p_val = stats.ttest_1samp(acc, popmean=CHANCE)
            mean = float(acc.mean())
            sem = float(acc.std(ddof=1) / np.sqrt(len(acc)))
            pvals.append(p_val)
            cell_data.append({
                "contrast": contrast, "roi": roi, "n_subjects": len(acc),
                "mean_accuracy": mean, "sem": sem, "std": float(acc.std(ddof=1)),
                "t_stat": float(t_stat), "p_value": float(p_val),
            })
        qvals = benjamini_hochberg(np.asarray(pvals))
        for cell, q in zip(cell_data, qvals):
            cell["q_value_fdr"] = float(q)
            cell["significant_fdr05"] = bool(q < 0.05)
            rows.append(cell)

    group_df = pd.DataFrame(rows)
    group_df.to_csv(out / "group_roi_stats.csv", index=False)

    n_sig = int(group_df["significant_fdr05"].sum())
    within_mean = group_df[group_df["contrast"].isin(WITHIN)]["mean_accuracy"].mean()
    cross_mean = group_df[~group_df["contrast"].isin(WITHIN)]["mean_accuracy"].mean()
    print(f"group stats: {len(group_df)} cells, {n_sig} significant at FDR q<0.05")
    print(f"within-source grand mean: {within_mean*100:.2f}%")
    print(f"cross-source grand mean: {cross_mean*100:.2f}%")
    print("AGGREGATE_COMPLETE")


if __name__ == "__main__":
    main()
