#!/usr/bin/env python3
"""Aggregate per-subject whitened random-10-fold Kastner ROI decoding results (n=29, Bo et al.
2021 CV scheme, 30 repeats, Ledoit-Wolf multivariate noise whitening) and add group-level
statistics. Mirrors aggregate_roi_decoding_random10fold.py exactly, pointed at the whitened
output tree, plus a whitening-diagnostics rollup.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/roi_decoding_random10fold_whitened")
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
    diag_frames = []
    for sub_dir in sorted(ROOT.glob("sub-*")):
        f = sub_dir / "subject_results.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
        d = sub_dir / "whitening_diagnostics.csv"
        if d.exists():
            ddf = pd.read_csv(d)
            if "subject" not in ddf.columns:
                ddf.insert(0, "subject", sub_dir.name)
            diag_frames.append(ddf)
    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv(OUT / "all_subject_results.csv", index=False)
    n_subjects = all_results["subject"].nunique()
    print(f"aggregated {len(all_results)} rows across {n_subjects} subjects")

    all_diag = pd.concat(diag_frames, ignore_index=True)
    all_diag.to_csv(OUT / "all_whitening_diagnostics.csv", index=False)
    all_diag["eigval_ratio"] = all_diag["max_eigval_raw"] / all_diag["min_eigval_raw"].clip(lower=1e-12)
    print(f"whitening diagnostics: {len(all_diag)} subject x ROI rows")
    print(f"  shrinkage: min={all_diag['shrinkage'].min():.4f} median={all_diag['shrinkage'].median():.4f} max={all_diag['shrinkage'].max():.4f}")
    print(f"  eigval_ratio: min={all_diag['eigval_ratio'].min():.1f} median={all_diag['eigval_ratio'].median():.1f} max={all_diag['eigval_ratio'].max():.1f}")
    print(f"  n_negative_or_tiny_eigval_clipped: sum={int(all_diag['n_negative_or_tiny_eigval_clipped'].sum())} max_per_cell={int(all_diag['n_negative_or_tiny_eigval_clipped'].max())}")

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
    print(f"group stats: {len(group_df)} ROI x contrast cells, {n_sig} significant at FDR q<0.05")

    chart_data = {
        "n_subjects": int(n_subjects),
        "chance": CHANCE,
        "cv_scheme": "random 10-fold x30 repeats, whitened (Bo et al. 2021 CV scheme + Ledoit-Wolf multivariate noise normalization)",
        "roi_order": list(ROI_ORDER),
        "contrast_order": list(CONTRAST_ORDER),
        "cells": rows,
    }
    (OUT / "roi_decoding_chart_data.json").write_text(json.dumps(chart_data, indent=2) + "\n")
    print(f"saved {OUT / 'roi_decoding_chart_data.json'}")
    print("AGGREGATE_COMPLETE")


if __name__ == "__main__":
    main()
