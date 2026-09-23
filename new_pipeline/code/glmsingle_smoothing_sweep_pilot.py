#!/usr/bin/env python3
"""Pilot: does increasing GLMsingle's post-hoc smoothing kernel (currently 8mm, applied
to already-estimated single-trial Type-D betas) improve ERP decoding accuracy?

Motivation: SPM applies smoothing to the BOLD timeseries BEFORE fitting the single-trial
GLM (pre-modeling smoothing reduces per-voxel noise that then feeds into the regression
itself). GLMsingle's pipeline here does the opposite -- it estimates single-trial betas
from UNSMOOTHED voxel timeseries (each fit is nonlinear: per-voxel HRF selection,
GLMdenoise nuisance-regressor selection, per-voxel ridge tuning), and only smooths the
resulting beta MAPS afterward. Because those per-voxel steps are nonlinear, smoothing
before vs after does not commute here, so post-hoc smoothing may recover less SNR than
pre-modeling smoothing did for SPM. If that mechanism matters, decoding accuracy should
keep improving as the post-hoc kernel widens (up to the point ROI-mixing costs outweigh
noise averaging). This sweeps 8/12/16/20mm on the same Type-D betas, same 3 pilot
subjects used in the Type-B vs Type-D test, to see whether that's true.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER, CONTRASTS, validate_manifest  # noqa: E402
from glmsingle_typeb_erp_pilot import load_glmsingle_variant, decode_one  # noqa: E402


def main() -> None:
    ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")
    KASTNER = ROOT / ".." / "Decoding" / "full_cohort_raw_pipeline_28" / "resources" / "kastner"
    atlas = KASTNER / "kastner.nii.gz"
    labels = KASTNER / "kastner.nii.txt"

    pilot_subjects = [1, 2, 4]
    smoothing_levels = [8.0, 12.0, 16.0, 20.0]
    all_rows = []
    for s in pilot_subjects:
        input_root = ROOT / "glmsingle" / f"sub-{s:02d}"
        for mm in smoothing_levels:
            data, positions, manifest = load_glmsingle_variant(
                input_root, atlas, labels, mm, "TYPED_FITHRF_GLMDENOISE_RR.hdf5"
            )
            manifest = validate_manifest(manifest)
            df = decode_one(data, positions, manifest)
            df["subject"] = f"Sub{s:02d}"
            df["smoothing_mm"] = mm
            all_rows.append(df)
            print(f"sub-{s:02d} {mm:g}mm: mean accuracy {df['accuracy'].mean():.4f}", flush=True)

    result = pd.concat(all_rows, ignore_index=True)
    out = ROOT / "glmsingle_smoothing_sweep_results.csv"
    result.to_csv(out, index=False)
    print(f"\nsaved {out}")

    summary = result.groupby("smoothing_mm")["accuracy"].mean()
    print("\nmean accuracy by smoothing level:")
    print((summary * 100).round(2).to_string())


if __name__ == "__main__":
    main()
