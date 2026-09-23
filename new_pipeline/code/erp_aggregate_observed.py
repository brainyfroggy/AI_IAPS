#!/usr/bin/env python3
"""Aggregate the per-(subject, ROI) ERP-style observed CSVs written by erp_permute_worker.py
into an all-subject table plus group stats (one-sample t-test vs chance + BH-FDR across the
17 ROIs within each contrast).

The group_roi_stats.csv this writes is the file stelzer_group_resample.py consumes via
--observed-csv, so this must run before the permutation group stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER, CONTRASTS  # noqa: E402
from aal3_common import load_aal3_kept_regions  # noqa: E402

CHANCE = 0.5

DEFAULT_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/erp_permutation")


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
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--roi-manifest", type=Path, default=None,
                    help="AAL3-style kept-regions CSV (id, roi_name, kept); overrides Kastner ROI_ORDER")
    ap.add_argument("--contrast-kind", choices=["within", "cross"], default=None)
    args = ap.parse_args()
    root = args.root

    roi_order = ([name for _rid, name in load_aal3_kept_regions(args.roi_manifest)]
                 if args.roi_manifest is not None else list(ROI_ORDER))
    contrasts = [c for c in CONTRASTS if args.contrast_kind is None or c[1] == args.contrast_kind]
    contrast_order = tuple(c[0] for c in contrasts)
    within = tuple(c[0] for c in contrasts if c[1] == "within")

    obs_dir = root / "observed"
    out_dir = root / "group"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(obs_dir.glob("sub*_*.csv"))
    print(f"found {len(files)} observed unit files")
    all_results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    all_results.to_csv(out_dir / "all_subject_results.csv", index=False)
    n_subjects = all_results["subject"].nunique()
    print(f"aggregated {len(all_results)} rows across {n_subjects} subjects")

    rows = []
    for contrast in contrast_order:
        sub_frame = all_results[all_results["contrast"] == contrast]
        pvals, cell_data = [], []
        for roi in roi_order:
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
    within_mean = group_df[group_df["contrast"].isin(within)]["mean_accuracy"].mean()
    cross_mean = group_df[~group_df["contrast"].isin(within)]["mean_accuracy"].mean()
    print(f"group stats: {len(group_df)} cells, {n_sig} significant at t-test FDR q<0.05")
    print(f"within-source grand mean: {within_mean*100:.2f}%")
    print(f"cross-source grand mean : {cross_mean*100:.2f}%")

    n_repeats_used = int(all_results["n_repeats"].iloc[0])
    chart_data = {
        "n_subjects": int(n_subjects),
        "chance": CHANCE,
        "cv_scheme": f"ERP-style Avg(random): voxel-source z-score, KFold(4) x {n_repeats_used} repeats, "
                     "3-group train averaging (within); 4-chunk averaging both sides (cross)",
        "roi_order": list(roi_order),
        "contrast_order": list(contrast_order),
        "cells": rows,
    }
    (out_dir / "roi_decoding_chart_data.json").write_text(json.dumps(chart_data, indent=2) + "\n")
    print("ERP_AGGREGATE_COMPLETE")


if __name__ == "__main__":
    main()
