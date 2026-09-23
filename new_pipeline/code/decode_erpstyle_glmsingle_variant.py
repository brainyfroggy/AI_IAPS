#!/usr/bin/env python3
"""Generic ERP-style observed decode for any GLMsingle HDF5 variant tree with the
standard per-subject layout (analysis_mask.nii.gz, flat_mask_indices.npy,
trial_manifest.tsv, glmsingle/<hdf5-name>). Reuses load_glmsingle_variant/decode_one
from glmsingle_typeb_erp_pilot.py. Observed only, no permutation, 8mm smoothing,
Kastner ROIs, n_repeats=30 (matching the established ERP convention this session).

Writes one subject_results.csv per subject under --output/sub-XX/, then aggregates
with t-test+FDR into --output/group/group_roi_stats.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER, CONTRASTS, validate_manifest  # noqa: E402
from glmsingle_typeb_erp_pilot import load_glmsingle_variant, decode_one  # noqa: E402

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")
KASTNER = ROOT / ".." / "Decoding" / "full_cohort_raw_pipeline_28" / "resources" / "kastner"
CONTRAST_ORDER = tuple(c[0] for c in CONTRASTS)
WITHIN = tuple(c[0] for c in CONTRASTS if c[1] == "within")
CHANCE = 0.5
SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 33, 34]


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5-root", type=Path, required=True, help="dir with sub-XX/ subdirs")
    ap.add_argument("--hdf5-name", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-repeats", type=int, default=30,
                    help="random fold/chunk re-splits averaged per subject/cell (default 30, "
                         "matching the established ERP convention this session)")
    args = ap.parse_args()

    obs_dir = args.output / "observed"
    obs_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for s in SUBJECTS:
        input_root = args.hdf5_root / f"sub-{s:02d}"
        data, positions, manifest = load_glmsingle_variant(
            input_root, KASTNER / "kastner.nii.gz", KASTNER / "kastner.nii.txt", 8.0, args.hdf5_name
        )
        manifest = validate_manifest(manifest)
        df = decode_one(data, positions, manifest, n_repeats=args.n_repeats)
        df["subject"] = f"Sub{s:02d}"
        df.to_csv(obs_dir / f"sub-{s:02d}.csv", index=False)
        all_rows.append(df)
        print(f"sub-{s:02d}: mean accuracy {df['accuracy'].mean():.4f}", flush=True)

    all_results = pd.concat(all_rows, ignore_index=True)
    group_dir = args.output / "group"
    group_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(group_dir / "all_subject_results.csv", index=False)
    n_subjects = all_results["subject"].nunique()

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
            })
        qvals = benjamini_hochberg(np.asarray(pvals))
        for cell, q in zip(cell_data, qvals):
            cell["q_value_fdr"] = float(q)
            cell["significant_fdr05"] = bool(q < 0.05)
            rows.append(cell)

    group_df = pd.DataFrame(rows)
    group_df.to_csv(group_dir / "group_roi_stats.csv", index=False)

    n_sig = int(group_df["significant_fdr05"].sum())
    within_mean = group_df[group_df["contrast"].isin(WITHIN)]["mean_accuracy"].mean()
    cross_mean = group_df[~group_df["contrast"].isin(WITHIN)]["mean_accuracy"].mean()
    print(f"\ngroup stats: {len(group_df)} cells, {n_sig} significant at t-test FDR q<0.05")
    print(f"within-source grand mean: {within_mean*100:.2f}%")
    print(f"cross-source grand mean : {cross_mean*100:.2f}%")
    print("ERPSTYLE_VARIANT_COMPLETE")


if __name__ == "__main__":
    main()
