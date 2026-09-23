#!/usr/bin/env python3
"""Shared AAL3 whole-brain loading helpers for the cross-source decoding plan
(docs/AAL3_WHOLEBRAIN_CROSSSOURCE_PLAN.md). Reuses decode_roi_singletrial.py's
generic HDF5 loader and mask-normalized smoothing, but with the frozen
154-region AAL3 manifest (Phase 0) instead of the Kastner-specific
ROI_ORDER/ROI_LABEL_NAMES globals -- AAL3 regions are 1:1 with atlas ids here
(L/R kept separate), so no label-name grouping is needed.
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
from decode_roi_singletrial import load_hdf5_matrix  # noqa: E402


def load_aal3_kept_regions(manifest_csv: Path) -> list[tuple[int, str]]:
    df = pd.read_csv(manifest_csv)
    kept = df[df["kept"]].sort_values("id")
    return list(zip(kept["id"].astype(int).tolist(), kept["roi_name"].tolist()))


def aal3_roi_indices(
    atlas_native_labels: Path,
    kept_regions: list[tuple[int, str]],
    reference: nib.spatialimages.SpatialImage,
    analysis_mask: np.ndarray,
    min_voxels: int = 10,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
    atlas_img = nib.load(atlas_native_labels)
    if atlas_img.shape[:3] != reference.shape[:3] or not np.allclose(
        atlas_img.affine, reference.affine, atol=1e-5
    ):
        raise RuntimeError("AAL3 native-labels atlas grid does not match this subject's reference grid")
    values = np.rint(np.asarray(atlas_img.dataobj)).astype(np.int32)

    full: dict[str, np.ndarray] = {}
    counts = []
    for rid, name in kept_regions:
        mask = values == rid
        before = int(mask.sum())
        indices = np.flatnonzero((mask & analysis_mask).ravel()).astype(np.int64)
        if len(indices) < min_voxels:
            raise RuntimeError(f"{name} (id={rid}) has only {len(indices)} analysis voxels (<{min_voxels})")
        full[name] = indices
        counts.append({
            "roi": name, "id": rid,
            "atlas_voxels_on_target_grid": before,
            "analysis_voxels": int(len(indices)),
        })
    union = np.unique(np.concatenate(list(full.values())))
    positions = {name: np.searchsorted(union, full[name]) for _, name in kept_regions}
    return union, positions, counts


def load_aal3_glmsingle(
    root: Path,
    atlas_native_labels: Path,
    kept_regions: list[tuple[int, str]],
    smoothing: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict], pd.DataFrame, nib.Nifti1Image]:
    mask_image = nib.load(root / "analysis_mask.nii.gz")
    mask = np.asarray(mask_image.dataobj) > 0
    flat = np.load(root / "flat_mask_indices.npy").astype(np.int64)
    if not np.array_equal(flat, np.flatnonzero(mask.ravel())):
        raise RuntimeError("GLMsingle flat-mask index set is inconsistent")

    union, positions, counts = aal3_roi_indices(atlas_native_labels, kept_regions, mask_image, mask)
    selected = np.searchsorted(flat, union)
    if np.any(selected >= len(flat)) or not np.array_equal(flat[selected], union):
        raise RuntimeError("ROI union is not contained in the GLMsingle mask")

    matrix = load_hdf5_matrix(root / "glmsingle" / "TYPED_FITHRF_GLMDENOISE_RR.hdf5", len(flat))
    if smoothing == 0:
        data = matrix[:, selected].copy()
    else:
        sigma_mm = smoothing / math.sqrt(8 * math.log(2))
        sigma_vox = sigma_mm / nib.affines.voxel_sizes(mask_image.affine)
        denominator = gaussian_filter(mask.astype(np.float32), sigma=sigma_vox, mode="constant", cval=0.0)
        denominator = denominator.ravel()[union]
        if np.any(denominator <= np.finfo(np.float32).eps):
            raise RuntimeError("Zero mask-normalized smoothing denominator")
        data = np.empty((600, len(union)), dtype=np.float32)
        volume = np.zeros(mask.shape, dtype=np.float32)
        for trial in range(600):
            volume.fill(0)
            volume.ravel()[flat] = matrix[trial]
            smoothed = gaussian_filter(volume, sigma=sigma_vox, mode="constant", cval=0.0)
            data[trial] = smoothed.ravel()[union] / denominator
            if (trial + 1) % 200 == 0:
                print(f"smoothed {trial + 1}/600 betas at {smoothing:g} mm", flush=True)

    manifest = pd.read_csv(root / "trial_manifest.tsv", sep="\t").sort_values("beta_index")
    return data, positions, counts, manifest, mask_image
