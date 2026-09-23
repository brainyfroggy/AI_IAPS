#!/usr/bin/env python3
"""Full 30-subject ERP decoding comparison: fixedhrf_full (wantlibrary=0 + GLMdenoise
+ ridge) vs production typeD (wantlibrary=1 + GLMdenoise + ridge). Isolates JUST the
HRF-library factor at production-quality settings (denoise+ridge on in both arms).

No permutation test -- observed accuracy only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import validate_manifest  # noqa: E402
from glmsingle_typeb_erp_pilot import load_glmsingle_variant, decode_one  # noqa: E402

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")
KASTNER = ROOT / ".." / "Decoding" / "full_cohort_raw_pipeline_28" / "resources" / "kastner"
FIXEDHRF_FULL_ROOT = ROOT / "glmsingle_fixedhrf_full_pilot"
TYPED_ROOT = ROOT / "glmsingle"
SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 33, 34]


def decode_variant(subject: int, hdf5_root: Path) -> pd.DataFrame:
    input_root = hdf5_root / f"sub-{subject:02d}"
    data, positions, manifest = load_glmsingle_variant(
        input_root, KASTNER / "kastner.nii.gz", KASTNER / "kastner.nii.txt", 8.0,
        "TYPED_FITHRF_GLMDENOISE_RR.hdf5"
    )
    manifest = validate_manifest(manifest)
    return decode_one(data, positions, manifest)


def main() -> None:
    all_rows = []
    for s in SUBJECTS:
        for variant, root in [("fixedhrf_full", FIXEDHRF_FULL_ROOT), ("typeD", TYPED_ROOT)]:
            df = decode_variant(s, root)
            df["subject"] = f"Sub{s:02d}"
            df["variant"] = variant
            all_rows.append(df)
            print(f"sub-{s:02d} {variant}: mean accuracy {df['accuracy'].mean():.4f}", flush=True)

    result = pd.concat(all_rows, ignore_index=True)
    out = ROOT / "fixedhrf_full_vs_typed_n30.csv"
    result.to_csv(out, index=False)
    print(f"\nsaved {out}")

    print("\n=== mean accuracy by variant ===")
    print((result.groupby("variant")["accuracy"].mean() * 100).round(2).to_string())

    within = result[result["contrast"].str.startswith("within")]
    cross = result[result["contrast"].str.startswith("train")]
    print("\nwithin-source:")
    print((within.groupby("variant")["accuracy"].mean() * 100).round(2).to_string())
    print("cross-source:")
    print((cross.groupby("variant")["accuracy"].mean() * 100).round(2).to_string())

    pivot = result.pivot_table(index=["subject", "roi", "contrast"], columns="variant", values="accuracy")
    pivot["diff"] = pivot["fixedhrf_full"] - pivot["typeD"]
    print(f"\nmean(fixedhrf_full - typeD): {pivot['diff'].mean()*100:+.2f} pts")
    print(f"fixedhrf_full wins: {(pivot['diff'] > 0).sum()}/{len(pivot)} cells")

    within_pivot = within.pivot_table(index=["subject", "roi", "contrast"], columns="variant", values="accuracy")
    within_pivot["diff"] = within_pivot["fixedhrf_full"] - within_pivot["typeD"]
    print(f"within-source mean(fixedhrf_full - typeD): {within_pivot['diff'].mean()*100:+.2f} pts, "
          f"wins {(within_pivot['diff'] > 0).sum()}/{len(within_pivot)}")

    cross_pivot = cross.pivot_table(index=["subject", "roi", "contrast"], columns="variant", values="accuracy")
    cross_pivot["diff"] = cross_pivot["fixedhrf_full"] - cross_pivot["typeD"]
    print(f"cross-source mean(fixedhrf_full - typeD): {cross_pivot['diff'].mean()*100:+.2f} pts, "
          f"wins {(cross_pivot['diff'] > 0).sum()}/{len(cross_pivot)}")

    print("\nCOMPARE_FIXEDHRF_FULL_N30_COMPLETE")


if __name__ == "__main__":
    main()
