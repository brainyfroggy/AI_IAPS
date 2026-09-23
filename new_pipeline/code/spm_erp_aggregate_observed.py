#!/usr/bin/env python3
"""Same as erp_aggregate_observed.py, pointed at the SPM ERP-style output tree
(spm_erp_permutation/) instead of the GLMsingle tree (erp_permutation/)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER, CONTRASTS  # noqa: E402

CONTRAST_ORDER = tuple(c[0] for c in CONTRASTS)
WITHIN = tuple(c[0] for c in CONTRASTS if c[1] == "within")
CHANCE = 0.5

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/spm_erp_permutation")


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
    obs_dir = ROOT / "observed"
    out_dir = ROOT / "group"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(obs_dir.glob("sub*_*.csv"))
    print(f"found {len(files)} observed unit files")
    all_results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    all_results.to_csv(out_dir / "all_subject_results.csv", index=False)
    n_subjects = all_results["subject"].nunique()
    print(f"aggregated {len(all_results)} rows across {n_subjects} subjects")

    rows = []
    for contrast in CONTRAST_ORDER:
        sub_frame = all_results[all_results["contrast"] == contrast]
        pvals, cell_data = [], []
        for roi in ROI_ORDER:
            acc = sub_frame[sub_frame["roi"] == roi].sort_values("subject")["accuracy"].to_numpy()
            if len(acc) != n_subjects:
                raise RuntimeError(f"{contrast}/{roi}: expected {n_subjects} subjects, got {len(acc)}")
            t_stat, p_val = stats.ttest_1samp(acc, popmean=CHANCE)
            pvals.append(p_val)
            cell_data.append({
                "contrast": contrast, "roi": roi, "n_subjects": len(acc),
                "mean_accuracy": float(acc.mean()),
                "sem": float(acc.std(ddof=1) / np.sqrt(len(acc))),
                "std": float(acc.std(ddof=1)),
                "t_stat": float(t_stat), "p_value": float(p_val),
            })
        qvals = benjamini_hochberg(np.asarray(pvals))
        for cell, q in zip(cell_data, qvals):
            cell["q_value_fdr"] = float(q)
            cell["significant_fdr05"] = bool(q < 0.05)
            rows.append(cell)

    group_df = pd.DataFrame(rows)
    group_df.to_csv(out_dir / "group_roi_stats.csv", index=False)

    n_sig = int(group_df["significant_fdr05"].sum())
    within_mean = group_df[group_df["contrast"].isin(WITHIN)]["mean_accuracy"].mean()
    cross_mean = group_df[~group_df["contrast"].isin(WITHIN)]["mean_accuracy"].mean()
    print(f"group stats: {len(group_df)} cells, {n_sig} significant at t-test FDR q<0.05")
    print(f"within-source grand mean: {within_mean*100:.2f}%")
    print(f"cross-source grand mean : {cross_mean*100:.2f}%")

    chart_data = {
        "n_subjects": int(n_subjects),
        "chance": CHANCE,
        "cv_scheme": "ERP-style Avg(random) on SPM LSS betas: voxel-source z-score, "
                     "KFold(4) x 30 repeats, 3-group train averaging (within); "
                     "4-chunk averaging both sides (cross)",
        "roi_order": list(ROI_ORDER),
        "contrast_order": list(CONTRAST_ORDER),
        "cells": rows,
    }
    (out_dir / "roi_decoding_chart_data.json").write_text(json.dumps(chart_data, indent=2) + "\n")
    print("SPM_ERP_AGGREGATE_COMPLETE")


if __name__ == "__main__":
    main()
