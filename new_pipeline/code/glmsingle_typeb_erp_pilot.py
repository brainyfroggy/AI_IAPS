#!/usr/bin/env python3
"""Pilot: does GLMsingle's Type-B (FITHRF, no GLMdenoise, no ridge) beat Type-D
(FITHRF+GLMdenoise+ridge) for ERP-style decoding? Type-D's ridge regularization is
chosen to maximize single-trial beta reliability/predictability -- a criterion that
need not align with preserving weak condition-discriminative variance. If ridge
shrinkage is disproportionately removing valence signal along with noise, Type-B
(less regularized) should decode better despite noisier single-trial estimates,
since the ERP-style pipeline averages trials anyway (which cancels random noise
regardless of the estimator) and z-scores before classification (which removes
pure scale differences) -- so a genuine accuracy gap here would implicate SNR,
not amplitude.

Mirrors decode_roi_erpstyle.py exactly (same load_glmsingle logic, same
compute_voxel_source_zscore / decode_within_avg / decode_cross_avg), with only the
GLMsingle HDF5 filename swapped.
"""

from __future__ import annotations

import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import (  # noqa: E402
    ROI_ORDER, CONTRASTS, roi_indices, load_hdf5_matrix, validate_manifest,
)
from decode_roi_erpstyle import (  # noqa: E402
    compute_voxel_source_zscore, condition_label, decode_within_avg, decode_cross_avg,
    N_REPEATS, N_FOLDS, N_AVG_GROUPS,
)


def load_glmsingle_variant(root: Path, atlas: Path, labels: Path, smoothing: float, hdf5_name: str):
    mask_image = nib.load(root / "analysis_mask.nii.gz")
    mask = np.asarray(mask_image.dataobj) > 0
    flat = np.load(root / "flat_mask_indices.npy").astype(np.int64)
    union, positions, counts = roi_indices(atlas, labels, mask_image, mask)
    selected = np.searchsorted(flat, union)

    matrix = load_hdf5_matrix(root / "glmsingle" / hdf5_name, len(flat))
    if smoothing == 0:
        data = matrix[:, selected].copy()
    else:
        sigma_mm = smoothing / math.sqrt(8 * math.log(2))
        sigma_vox = sigma_mm / nib.affines.voxel_sizes(mask_image.affine)
        denominator = gaussian_filter(mask.astype(np.float32), sigma=sigma_vox, mode="constant", cval=0.0)
        denominator = denominator.ravel()[union]
        data = np.empty((600, len(union)), dtype=np.float32)
        volume = np.zeros(mask.shape, dtype=np.float32)
        for trial in range(600):
            volume.fill(0)
            volume.ravel()[flat] = matrix[trial]
            smoothed = gaussian_filter(volume, sigma=sigma_vox, mode="constant", cval=0.0)
            data[trial] = smoothed.ravel()[union] / denominator
    manifest = pd.read_csv(root / "trial_manifest.tsv", sep="\t").sort_values("beta_index")
    return data, positions, manifest


def load_glmsingle_variant_full(root: Path, atlas: Path, labels: Path, smoothing: float, hdf5_name: str):
    """Same as load_glmsingle_variant but returns the full 5-tuple
    (data, positions, counts, manifest, reference) that decode_roi_singletrial's
    load_glmsingle returns -- a drop-in monkeypatch target for scripts (like
    decode_roi_random10fold.py) that consume that exact signature."""
    mask_image = nib.load(root / "analysis_mask.nii.gz")
    mask = np.asarray(mask_image.dataobj) > 0
    flat = np.load(root / "flat_mask_indices.npy").astype(np.int64)
    union, positions, counts = roi_indices(atlas, labels, mask_image, mask)
    selected = np.searchsorted(flat, union)

    matrix = load_hdf5_matrix(root / "glmsingle" / hdf5_name, len(flat))
    if smoothing == 0:
        data = matrix[:, selected].copy()
    else:
        sigma_mm = smoothing / math.sqrt(8 * math.log(2))
        sigma_vox = sigma_mm / nib.affines.voxel_sizes(mask_image.affine)
        denominator = gaussian_filter(mask.astype(np.float32), sigma=sigma_vox, mode="constant", cval=0.0)
        denominator = denominator.ravel()[union]
        data = np.empty((600, len(union)), dtype=np.float32)
        volume = np.zeros(mask.shape, dtype=np.float32)
        for trial in range(600):
            volume.fill(0)
            volume.ravel()[flat] = matrix[trial]
            smoothed = gaussian_filter(volume, sigma=sigma_vox, mode="constant", cval=0.0)
            data[trial] = smoothed.ravel()[union] / denominator
    manifest = pd.read_csv(root / "trial_manifest.tsv", sep="\t").sort_values("beta_index")
    return data, positions, counts, manifest, mask_image


def decode_one(data, positions, manifest, n_repeats=N_REPEATS, n_folds=N_FOLDS, n_avg_groups=N_AVG_GROUPS):
    source_values = manifest["source"].astype(str).str.lower().to_numpy()
    valence_values = manifest["valence"].astype(str).str.lower().to_numpy()
    rows = []
    for roi in ROI_ORDER:
        roi_data = data[:, positions[roi]]
        finite = np.all(np.isfinite(roi_data), axis=0)
        roi_data = roi_data[:, finite]
        if roi_data.shape[1] < 10:
            continue
        cond_data = {}
        for source in ("natural", "ai"):
            for valence in ("pleasant", "neutral", "unpleasant"):
                idx = np.flatnonzero((source_values == source) & (valence_values == valence))
                cond_data[condition_label(source, valence)] = roi_data[idx]
        voxel_source_z = compute_voxel_source_zscore(cond_data)
        for name, kind, train_source, test_source, positive in CONTRASTS:
            if kind == "within":
                d1 = voxel_source_z[condition_label(train_source, positive)]
                d2 = voxel_source_z[condition_label(train_source, "neutral")]
                acc = decode_within_avg(d1, d2, n_repeats, n_folds, n_avg_groups)
            else:
                tr1 = voxel_source_z[condition_label(train_source, positive)]
                tr2 = voxel_source_z[condition_label(train_source, "neutral")]
                te1 = voxel_source_z[condition_label(test_source, positive)]
                te2 = voxel_source_z[condition_label(test_source, "neutral")]
                acc = decode_cross_avg(tr1, tr2, te1, te2, n_repeats, n_folds)
            rows.append({"roi": roi, "contrast": name, "accuracy": acc})
    return pd.DataFrame(rows)


def main() -> None:
    ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")
    KASTNER = ROOT / ".." / "Decoding" / "full_cohort_raw_pipeline_28" / "resources" / "kastner"
    atlas = KASTNER / "kastner.nii.gz"
    labels = KASTNER / "kastner.nii.txt"

    pilot_subjects = [1, 2, 4]
    all_rows = []
    for s in pilot_subjects:
        input_root = ROOT / "glmsingle" / f"sub-{s:02d}"
        for variant, hdf5_name in [("typeB", "TYPEB_FITHRF.hdf5"),
                                    ("typeD", "TYPED_FITHRF_GLMDENOISE_RR.hdf5")]:
            data, positions, manifest = load_glmsingle_variant(input_root, atlas, labels, 8.0, hdf5_name)
            manifest = validate_manifest(manifest)
            df = decode_one(data, positions, manifest)
            df["subject"] = f"Sub{s:02d}"
            df["variant"] = variant
            all_rows.append(df)
            print(f"sub-{s:02d} {variant}: mean accuracy {df['accuracy'].mean():.4f}", flush=True)

    result = pd.concat(all_rows, ignore_index=True)
    out = ROOT / "glmsingle_typeb_pilot_results.csv"
    result.to_csv(out, index=False)
    print(f"\nsaved {out}")

    pivot = result.pivot_table(index=["subject", "roi", "contrast"], columns="variant", values="accuracy")
    pivot["typeB_minus_typeD"] = pivot["typeB"] - pivot["typeD"]
    print(f"\nmean(typeB - typeD) across all cells: {pivot['typeB_minus_typeD'].mean()*100:+.2f} pts")
    print(f"typeB wins: {(pivot['typeB_minus_typeD'] > 0).sum()}/{len(pivot)} cells")
    print(f"typeB mean accuracy: {pivot['typeB'].mean()*100:.2f}%")
    print(f"typeD mean accuracy: {pivot['typeD'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
