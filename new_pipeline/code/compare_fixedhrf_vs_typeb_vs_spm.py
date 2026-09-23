#!/usr/bin/env python3
"""Three/four-way ERP decoding comparison for the matched 10-subject subset
(1,2,3,4,5,6,7,9,11,12):

  - fixedhrf : GLMsingle, wantlibrary=0 (ONE fixed HRF for every voxel), no denoise/ridge
  - typeB    : GLMsingle, wantlibrary=1 (per-voxel HRF library), no denoise/ridge (production)
  - typeD    : GLMsingle, wantlibrary=1 + GLMdenoise + ridge (production, the pipeline
               used throughout this project's headline results)
  - spm      : SPM LSS, canonical HRF, no denoise/ridge (from spm_erp_permutation/cache)

fixedhrf vs typeB isolates ONLY the HRF-library factor (identical settings otherwise).
typeB vs spm isolates HRF library vs canonical HRF, but confounded with the different
upstream preprocessing pipelines (fMRIPrep vs SPM-native) -- reported for context, not
as a clean ablation.

No permutation test -- observed accuracy only, per user instruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER, validate_manifest  # noqa: E402
from glmsingle_typeb_erp_pilot import load_glmsingle_variant, decode_one  # noqa: E402

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")
KASTNER = ROOT / ".." / "Decoding" / "full_cohort_raw_pipeline_28" / "resources" / "kastner"
FIXEDHRF_ROOT = ROOT / "glmsingle_fixedhrf_pilot"
PILOT_SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13]  # Sub3 has no GLMsingle data; swapped for Sub13


def decode_glmsingle_variant(subject: int, hdf5_root: Path, hdf5_name: str, smoothing: float) -> pd.DataFrame:
    input_root = hdf5_root / f"sub-{subject:02d}"
    data, positions, manifest = load_glmsingle_variant(
        input_root, KASTNER / "kastner.nii.gz", KASTNER / "kastner.nii.txt", smoothing, hdf5_name
    )
    manifest = validate_manifest(manifest)
    return decode_one(data, positions, manifest)


def decode_spm(subject: int) -> pd.DataFrame:
    from erp_permute_worker import load_cache  # reuse existing cache loader
    from decode_roi_erpstyle import compute_voxel_source_zscore, condition_label, decode_within_avg, decode_cross_avg
    from decode_roi_singletrial import CONTRASTS

    cache_path = ROOT / "spm_erp_permutation" / "cache" / f"sub-{subject:02d}.npz"
    data, positions, source, valence = load_cache(cache_path)
    rows = []
    for roi in ROI_ORDER:
        if roi not in positions:
            continue
        roi_data = data[:, positions[roi]]
        finite = np.all(np.isfinite(roi_data), axis=0)
        roi_data = roi_data[:, finite]
        if roi_data.shape[1] < 10:
            continue
        cond_data = {}
        for src in ("natural", "ai"):
            for val in ("pleasant", "neutral", "unpleasant"):
                idx = np.flatnonzero((source == src) & (valence == val))
                cond_data[condition_label(src, val)] = roi_data[idx]
        voxel_source_z = compute_voxel_source_zscore(cond_data)
        for name, kind, train_source, test_source, positive in CONTRASTS:
            if kind == "within":
                d1 = voxel_source_z[condition_label(train_source, positive)]
                d2 = voxel_source_z[condition_label(train_source, "neutral")]
                acc = decode_within_avg(d1, d2)
            else:
                tr1 = voxel_source_z[condition_label(train_source, positive)]
                tr2 = voxel_source_z[condition_label(train_source, "neutral")]
                te1 = voxel_source_z[condition_label(test_source, positive)]
                te2 = voxel_source_z[condition_label(test_source, "neutral")]
                acc = decode_cross_avg(tr1, tr2, te1, te2)
            rows.append({"roi": roi, "contrast": name, "accuracy": acc})
    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    for s in PILOT_SUBJECTS:
        for variant, fn in [
            ("fixedhrf", lambda s=s: decode_glmsingle_variant(s, FIXEDHRF_ROOT, "TYPEB_FITHRF.hdf5", 8.0)),
            ("typeB", lambda s=s: decode_glmsingle_variant(s, ROOT / "glmsingle", "TYPEB_FITHRF.hdf5", 8.0)),
            ("typeD", lambda s=s: decode_glmsingle_variant(s, ROOT / "glmsingle", "TYPED_FITHRF_GLMDENOISE_RR.hdf5", 8.0)),
            ("spm", lambda s=s: decode_spm(s)),
        ]:
            df = fn()
            df["subject"] = f"Sub{s:02d}"
            df["variant"] = variant
            all_rows.append(df)
            print(f"sub-{s:02d} {variant}: mean accuracy {df['accuracy'].mean():.4f}", flush=True)

    result = pd.concat(all_rows, ignore_index=True)
    out = ROOT / "fixedhrf_vs_typeb_vs_spm_n10.csv"
    result.to_csv(out, index=False)
    print(f"\nsaved {out}")

    print("\n=== mean accuracy by variant ===")
    print((result.groupby("variant")["accuracy"].mean() * 100).round(2).to_string())

    print("\n=== mean accuracy by variant x contrast_kind ===")
    within = result[result["contrast"].str.startswith("within")]
    cross = result[result["contrast"].str.startswith("train")]
    print("within-source:")
    print((within.groupby("variant")["accuracy"].mean() * 100).round(2).to_string())
    print("cross-source:")
    print((cross.groupby("variant")["accuracy"].mean() * 100).round(2).to_string())

    pivot = result.pivot_table(index=["subject", "roi", "contrast"], columns="variant", values="accuracy")
    pivot["fixedhrf_minus_typeB"] = pivot["fixedhrf"] - pivot["typeB"]
    print(f"\nmean(fixedhrf - typeB) [isolates JUST the HRF-library factor]: "
          f"{pivot['fixedhrf_minus_typeB'].mean()*100:+.2f} pts")
    print(f"fixedhrf wins: {(pivot['fixedhrf_minus_typeB'] > 0).sum()}/{len(pivot)} cells")

    print("\nCOMPARE_FIXEDHRF_COMPLETE")


if __name__ == "__main__":
    main()
