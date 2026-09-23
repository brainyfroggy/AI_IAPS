#!/usr/bin/env python3
"""Same as aggregate_roi_decoding_100rep_crossfix.py, pointed at the fixedhrf_full
tree (GLMsingle wantlibrary=0 + GLMdenoise + ridge, random-10-fold single-trial
decoding, corrected no-CV cross-source design)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/roi_decoding_random10fold_fixedhrf_full")
OUT = ROOT / "group"

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
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for sub_dir in sorted(ROOT.glob("sub-*")):
        f = sub_dir / "subject_results.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv(OUT / "all_subject_results.csv", index=False)
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
                "contrast": contrast,
                "roi": roi,
                "n_subjects": len(acc),
                "mean_accuracy": mean,
                "sem": sem,
                "std": float(acc.std(ddof=1)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
            })
        qvals = benjamini_hochberg(np.asarray(pvals))
        for cell, q in zip(cell_data, qvals):
            cell["q_value_fdr"] = float(q)
            cell["significant_fdr05"] = bool(q < 0.05)
            rows.append(cell)

    group_df = pd.DataFrame(rows)
    group_df.to_csv(OUT / "group_roi_stats.csv", index=False)

    n_sig = int(group_df["significant_fdr05"].sum())
    within_mean = group_df[group_df["contrast"].isin(WITHIN)]["mean_accuracy"].mean()
    cross_mean = group_df[~group_df["contrast"].isin(WITHIN)]["mean_accuracy"].mean()
    grand_mean = group_df["mean_accuracy"].mean()
    print(f"group stats: {len(group_df)} ROI x contrast cells, {n_sig} significant at FDR q<0.05")
    print(f"within-source grand mean: {within_mean*100:.2f}%")
    print(f"cross-source grand mean: {cross_mean*100:.2f}%")
    print(f"overall grand mean: {grand_mean*100:.2f}%")

    chart_data = {
        "n_subjects": int(n_subjects),
        "chance": CHANCE,
        "cv_scheme": ("GLMsingle wantlibrary=0 (fixed HRF) + GLMdenoise + ridge; "
                      "within-source: random 10-fold x100 repeats (Bo et al. 2021 scheme); "
                      "cross-source: single train-on-all-200/test-on-all-200 split, no CV"),
        "roi_order": list(ROI_ORDER),
        "contrast_order": list(CONTRAST_ORDER),
        "cells": rows,
    }
    (OUT / "roi_decoding_chart_data.json").write_text(json.dumps(chart_data, indent=2) + "\n")
    print(f"saved {OUT / 'roi_decoding_chart_data.json'}")
    print("AGGREGATE_COMPLETE")


if __name__ == "__main__":
    main()
