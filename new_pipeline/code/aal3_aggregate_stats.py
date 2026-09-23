#!/usr/bin/env python3
"""Phase 3 of the whole-brain AAL3 cross-source decoding plan.

Per smoothing level: one-sample t-test vs chance (50%) per (region, contrast)
cell, BH-FDR corrected across the 154 regions WITHIN each contrast (matches
the Kastner/Wang convention). Also emits an 8mm-vs-0mm comparison: per-cell
accuracy delta and significance agreement/disagreement between the two
smoothing trees.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CROSS_CONTRASTS = (
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


def group_stats(subject_csv: Path, roi_order: list[str]) -> pd.DataFrame:
    all_results = pd.read_csv(subject_csv)
    n_subjects = all_results["subject"].nunique()

    rows = []
    for contrast in CROSS_CONTRASTS:
        sub_frame = all_results[all_results["contrast"] == contrast]
        pvals = []
        cell_data = []
        for roi in roi_order:
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
    return pd.DataFrame(rows)


def main() -> None:
    root = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/aal3_crosssource")
    manifest = pd.read_csv(root / "phase0_manifest" / "roi_manifest.csv")
    roi_order = manifest.loc[manifest["kept"], "roi_name"].tolist()

    levels = {"smooth08": root / "decode_smooth08" / "subject_results.csv",
              "smooth00": root / "decode_smooth00" / "subject_results.csv"}

    group_dfs = {}
    for label, csv_path in levels.items():
        out_dir = root / label / "group"
        out_dir.mkdir(parents=True, exist_ok=True)
        gdf = group_stats(csv_path, roi_order)
        gdf.to_csv(out_dir / "group_aal3_stats.csv", index=False)
        group_dfs[label] = gdf

        n_sig = int(gdf["significant_fdr05"].sum())
        grand_mean = gdf["mean_accuracy"].mean()
        print(f"[{label}] {len(gdf)} cells, {n_sig} significant at FDR q<0.05, "
              f"grand mean {grand_mean*100:.2f}%")

        chart_data = {
            "n_subjects": int(gdf["n_subjects"].iloc[0]),
            "chance": CHANCE,
            "smoothing_fwhm_mm": 8.0 if label == "smooth08" else 0.0,
            "cv_scheme": "single train-on-all-200/test-on-all-200 split, no CV (corrected cross-source design)",
            "roi_order": roi_order,
            "contrast_order": list(CROSS_CONTRASTS),
            "cells": gdf.to_dict(orient="records"),
        }
        (out_dir / "aal3_chart_data.json").write_text(json.dumps(chart_data, indent=2) + "\n")

    # 8mm vs 0mm comparison
    merged = group_dfs["smooth08"].merge(
        group_dfs["smooth00"], on=["contrast", "roi"], suffixes=("_08mm", "_00mm")
    )
    merged["accuracy_delta_08_minus_00"] = merged["mean_accuracy_08mm"] - merged["mean_accuracy_00mm"]
    merged["sig_agree"] = merged["significant_fdr05_08mm"] == merged["significant_fdr05_00mm"]
    comp_dir = root / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(comp_dir / "smoothing_comparison.csv", index=False)

    n_agree = int(merged["sig_agree"].sum())
    print(f"\n[comparison] {len(merged)} cells; significance agreement 8mm vs 0mm: "
          f"{n_agree}/{len(merged)} ({100*n_agree/len(merged):.1f}%)")
    print(f"mean |accuracy delta| (8mm - 0mm): {merged['accuracy_delta_08_minus_00'].abs().mean()*100:.2f} pts")
    print(f"8mm-only significant: {int(((merged['significant_fdr05_08mm']) & ~(merged['significant_fdr05_00mm'])).sum())}")
    print(f"0mm-only significant: {int((~(merged['significant_fdr05_08mm']) & (merged['significant_fdr05_00mm'])).sum())}")
    print(f"both significant    : {int(((merged['significant_fdr05_08mm']) & (merged['significant_fdr05_00mm'])).sum())}")

    print("\nAAL3_AGGREGATE_COMPLETE")


if __name__ == "__main__":
    main()
