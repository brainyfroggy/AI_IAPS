#!/usr/bin/env python3
"""Build the ROI x cross-source-contrast significance table for the AAL3 whole-brain
permutation result: rows = AAL3 ROIs (in roi-manifest order), columns = the 4
cross-source contrasts, cell = significant at FDR q<0.05 (permutation test). Adds a
n_significant_contrasts count per ROI so "shared across contrasts" vs. "contrast-specific"
ROIs are visible directly, and writes out three views: the full pivot, ROIs significant
in all 4 contrasts (fully shared), and ROIs significant in exactly 1 (contrast-specific).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CROSS_CONTRASTS = [
    "train_natural_pleasant_vs_neutral_test_ai",
    "train_ai_pleasant_vs_neutral_test_natural",
    "train_natural_unpleasant_vs_neutral_test_ai",
    "train_ai_unpleasant_vs_neutral_test_natural",
]
CONTRAST_SHORT = {
    "train_natural_pleasant_vs_neutral_test_ai": "PL: NA to AI",
    "train_ai_pleasant_vs_neutral_test_natural": "PL: AI to NA",
    "train_natural_unpleasant_vs_neutral_test_ai": "UP: NA to AI",
    "train_ai_unpleasant_vs_neutral_test_natural": "UP: AI to NA",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-permutation-csv", type=Path, required=True,
                    help="group_permutation_results.csv from stelzer_group_resample.py")
    ap.add_argument("--roi-manifest", type=Path, required=True,
                    help="AAL3 roi_manifest.csv (id, roi_name, kept) for row order")
    ap.add_argument("--sig-col", default="sig_fdr_q05")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.roi_manifest)
    roi_order = manifest[manifest["kept"]].sort_values("id")["roi_name"].tolist()

    df = pd.read_csv(args.group_permutation_csv)
    missing = set(CROSS_CONTRASTS) - set(df["contrast"].unique())
    if missing:
        raise RuntimeError(f"group-permutation csv is missing cross contrasts: {missing}")

    pivot = df[df["contrast"].isin(CROSS_CONTRASTS)].pivot(
        index="roi", columns="contrast", values=args.sig_col
    ).reindex(roi_order)[CROSS_CONTRASTS]
    pivot = pivot.rename(columns=CONTRAST_SHORT)
    pivot["n_significant_contrasts"] = pivot.sum(axis=1).astype(int)

    acc_pivot = df[df["contrast"].isin(CROSS_CONTRASTS)].pivot(
        index="roi", columns="contrast", values="observed_accuracy"
    ).reindex(roi_order)[CROSS_CONTRASTS].rename(columns=CONTRAST_SHORT)

    full = pivot.reset_index().rename(columns={"roi": "ROI"})
    full.to_csv(args.out_dir / "aal3_crosssource_significance_table.csv", index=False)

    acc_out = acc_pivot.reset_index().rename(columns={"roi": "ROI"})
    acc_out.to_csv(args.out_dir / "aal3_crosssource_accuracy_table.csv", index=False)

    shared_all4 = full[full["n_significant_contrasts"] == 4]
    shared_all4.to_csv(args.out_dir / "aal3_shared_all4_contrasts.csv", index=False)

    specific_1 = full[full["n_significant_contrasts"] == 1]
    specific_1.to_csv(args.out_dir / "aal3_specific_1contrast.csv", index=False)

    n_rois = len(full)
    n_any = int((full["n_significant_contrasts"] > 0).sum())
    print(f"{n_rois} ROIs total")
    print(f"significant in >=1 cross contrast: {n_any}")
    print(f"significant in all 4 (fully shared): {len(shared_all4)}")
    for k in range(4, -1, -1):
        n_k = int((full["n_significant_contrasts"] == k).sum())
        print(f"  n_significant_contrasts == {k}: {n_k} ROIs")
    print(f"significant in exactly 1 (contrast-specific): {len(specific_1)}")
    print("AAL3_SIGNIFICANCE_TABLE_COMPLETE")


if __name__ == "__main__":
    main()
