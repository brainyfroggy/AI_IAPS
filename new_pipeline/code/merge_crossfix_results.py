#!/usr/bin/env python3
"""Merge the corrected cross-source results (single train-on-all/test-on-all,
no CV) with the unaffected within-source results, for both the observed-accuracy
decode tree and the Stelzer permutation null pools.

Within-source contrasts are untouched by the cross-source CV fix, so their rows
are carried over verbatim from the original 100-repeat / null_pools trees. Only
the 4 cross-source contrasts are replaced with the new no-fold-CV results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")

CROSS_CONTRASTS = {
    "train_natural_pleasant_vs_neutral_test_ai",
    "train_ai_pleasant_vs_neutral_test_natural",
    "train_natural_unpleasant_vs_neutral_test_ai",
    "train_ai_unpleasant_vs_neutral_test_natural",
}

ROI_ORDER = (
    "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "V3a", "V3b",
    "IPS", "LO1", "LO2", "hMT", "VO1", "VO2", "PHC1", "PHC2",
)


def merge_decode() -> None:
    src_within = ROOT / "roi_decoding_random10fold_100rep"
    src_cross = ROOT / "roi_decoding_random10fold_100rep_crossfix"
    out_root = ROOT / "roi_decoding_random10fold_100rep_fixed"
    out_root.mkdir(exist_ok=True)

    n_subjects = 0
    for sub_dir in sorted(src_within.glob("sub-*")):
        sub = sub_dir.name
        within_df = pd.read_csv(sub_dir / "subject_results.csv")
        within_df = within_df[~within_df["contrast"].isin(CROSS_CONTRASTS)]
        cross_df = pd.read_csv(src_cross / sub / "subject_results.csv")
        assert set(cross_df["contrast"]) == CROSS_CONTRASTS, f"{sub}: unexpected cross contrasts"

        merged = pd.concat([within_df, cross_df], ignore_index=True)
        assert len(merged) == 136, f"{sub}: expected 136 rows, got {len(merged)}"

        out_dir = out_root / sub
        out_dir.mkdir(exist_ok=True)
        merged.to_csv(out_dir / "subject_results.csv", index=False)
        n_subjects += 1

    print(f"merge_decode: wrote {n_subjects} subjects to {out_root}")


def merge_null_pools() -> None:
    src_within = ROOT / "stelzer_permutation" / "null_pools"
    src_cross = ROOT / "stelzer_permutation" / "null_pools_crossfix"
    out_dir = ROOT / "stelzer_permutation" / "null_pools_fixed"
    out_dir.mkdir(exist_ok=True)

    files = sorted(src_within.glob("sub*_*.csv"))
    n_files = 0
    for f in files:
        within_df = pd.read_csv(f)
        within_df = within_df[~within_df["contrast"].isin(CROSS_CONTRASTS)]
        cross_f = src_cross / f.name
        cross_df = pd.read_csv(cross_f)
        assert set(cross_df["contrast"]) == CROSS_CONTRASTS, f"{f.name}: unexpected cross contrasts"

        merged = pd.concat([within_df, cross_df], ignore_index=True)
        assert len(merged) == 800, f"{f.name}: expected 800 rows (8 contrasts x 100 perms), got {len(merged)}"

        merged.to_csv(out_dir / f.name, index=False)
        n_files += 1

    print(f"merge_null_pools: wrote {n_files} files to {out_dir}")


if __name__ == "__main__":
    merge_decode()
    merge_null_pools()
    print("MERGE_CROSSFIX_COMPLETE")
