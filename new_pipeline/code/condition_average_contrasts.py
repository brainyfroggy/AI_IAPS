#!/usr/bin/env python3
"""Condition-averaged univariate contrast maps from GLMsingle single-trial betas.

Averages single-trial Type-D betas within each of the 6 conditions (source x
valence), forms 4 contrasts (Natural/AI x Pleasant/Unpleasant vs Neutral), and
saves each contrast at multiple post-GLM smoothing levels as .nii.gz.

Smoothing-then-average and average-then-smooth are mathematically identical
for a linear (Gaussian) kernel with a trial-independent mask-normalization
denominator, so this averages the raw betas first (600 trials -> 6 condition
means -> 4 contrasts, cheap) and smooths only the 4 resulting contrast maps
per smoothing level, rather than smoothing all 600 trials individually. The
smoothing implementation (sigma-from-FWHM, mask-normalized denominator) is
copied verbatim from decode_roi_singletrial.py's load_glmsingle() for
methodological consistency with the rest of this project.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

SMOOTHING_LEVELS_MM = [0, 3, 5, 8]

CONTRASTS = {
    "natural_pleasant_vs_neutral": (("natural", "pleasant"), ("natural", "neutral")),
    "natural_unpleasant_vs_neutral": (("natural", "unpleasant"), ("natural", "neutral")),
    "ai_pleasant_vs_neutral": (("ai", "pleasant"), ("ai", "neutral")),
    "ai_unpleasant_vs_neutral": (("ai", "unpleasant"), ("ai", "neutral")),
}


def load_betas(root: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image, pd.DataFrame]:
    mask_image = nib.load(root / "analysis_mask.nii.gz")
    mask = np.asarray(mask_image.dataobj) > 0
    flat = np.load(root / "flat_mask_indices.npy").astype(np.int64)
    if not np.array_equal(flat, np.flatnonzero(mask.ravel())):
        raise RuntimeError("flat_mask_indices inconsistent with analysis_mask")

    hdf5_path = root / "glmsingle" / "TYPED_FITHRF_GLMDENOISE_RR.hdf5"
    n_voxels = len(flat)
    matrix = np.empty((600, n_voxels), dtype=np.float32)
    with h5py.File(hdf5_path, "r") as handle:
        dataset = handle["betasmd"]
        if dataset.shape != (n_voxels, 1, 1, 600):
            raise RuntimeError(f"Unexpected betasmd shape: {dataset.shape}")
        block = 8192
        for start in range(0, n_voxels, block):
            stop = min(start + block, n_voxels)
            matrix[:, start:stop] = np.asarray(dataset[start:stop, 0, 0, :]).T
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("GLMsingle betas contain nonfinite values")

    manifest = pd.read_csv(root / "trial_manifest.tsv", sep="\t").sort_values("beta_index")
    if manifest["beta_index"].astype(int).tolist() != list(range(600)):
        raise RuntimeError("trial_manifest beta_index is not exactly 0..599")
    manifest = manifest.reset_index(drop=True)
    manifest["source"] = manifest["source"].astype(str).str.lower()
    manifest["valence"] = manifest["valence"].astype(str).str.lower()

    return matrix, flat, mask_image, manifest


def condition_mean(matrix: np.ndarray, manifest: pd.DataFrame, source: str, valence: str) -> np.ndarray:
    rows = (manifest["source"] == source) & (manifest["valence"] == valence)
    n = int(rows.sum())
    if n == 0:
        raise RuntimeError(f"No trials found for source={source} valence={valence}")
    return matrix[rows.to_numpy()].mean(axis=0), n


def smooth_full_volume(
    values_at_flat: np.ndarray,
    flat: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    fwhm_mm: float,
) -> np.ndarray:
    volume = np.zeros(mask.shape, dtype=np.float32)
    if fwhm_mm == 0:
        volume.ravel()[flat] = values_at_flat
        return volume
    sigma_mm = fwhm_mm / math.sqrt(8 * math.log(2))
    sigma_vox = sigma_mm / nib.affines.voxel_sizes(affine)
    denominator = gaussian_filter(mask.astype(np.float32), sigma=sigma_vox, mode="constant", cval=0.0)
    if np.any(denominator[mask] <= np.finfo(np.float32).eps):
        raise RuntimeError("Zero mask-normalized smoothing denominator")
    raw = np.zeros(mask.shape, dtype=np.float32)
    raw.ravel()[flat] = values_at_flat
    smoothed = gaussian_filter(raw, sigma=sigma_vox, mode="constant", cval=0.0)
    out = np.zeros(mask.shape, dtype=np.float32)
    out[mask] = smoothed[mask] / denominator[mask]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--glmsingle-root", type=Path, required=True,
                         help="Directory containing analysis_mask.nii.gz, flat_mask_indices.npy, "
                              "trial_manifest.tsv, and glmsingle/TYPED_FITHRF_GLMDENOISE_RR.hdf5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoothing-mm", type=float, nargs="+", default=SMOOTHING_LEVELS_MM)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    matrix, flat, mask_image, manifest = load_betas(args.glmsingle_root)
    mask = np.asarray(mask_image.dataobj) > 0
    affine = mask_image.affine

    condition_means = {}
    condition_counts = {}
    for source in ("natural", "ai"):
        for valence in ("pleasant", "neutral", "unpleasant"):
            mean_vec, n = condition_mean(matrix, manifest, source, valence)
            condition_means[(source, valence)] = mean_vec
            condition_counts[(source, valence)] = n
            print(f"condition {source}/{valence}: n={n} trials")

    provenance_rows = []
    for contrast_name, (pos_key, neg_key) in CONTRASTS.items():
        contrast_vec = condition_means[pos_key] - condition_means[neg_key]
        if not np.all(np.isfinite(contrast_vec)):
            raise RuntimeError(f"{contrast_name}: nonfinite values in contrast")
        for fwhm in args.smoothing_mm:
            volume = smooth_full_volume(contrast_vec, flat, mask, affine, fwhm)
            out_path = args.output / f"sub-{args.subject:02d}_contrast-{contrast_name}_smoothing-{fwhm:g}mm.nii.gz"
            img = nib.Nifti1Image(volume, affine, header=mask_image.header.copy())
            img.header.set_data_dtype(np.float32)
            nib.save(img, out_path)
            print(f"saved {out_path.name}  (min={volume[mask].min():.4f} max={volume[mask].max():.4f})")
            provenance_rows.append({
                "contrast": contrast_name,
                "smoothing_fwhm_mm": fwhm,
                "positive_condition": f"{pos_key[0]}-{pos_key[1]}",
                "negative_condition": f"{neg_key[0]}-{neg_key[1]}",
                "n_positive_trials": condition_counts[pos_key],
                "n_negative_trials": condition_counts[neg_key],
                "output_file": out_path.name,
                "n_voxels_in_mask": int(mask.sum()),
            })

    prov = pd.DataFrame(provenance_rows)
    prov_path = args.output / f"sub-{args.subject:02d}_provenance.tsv"
    prov.to_csv(prov_path, sep="\t", index=False)
    print(f"saved {prov_path}")
    print("CONDITION_AVERAGE_CONTRASTS_COMPLETE")


if __name__ == "__main__":
    main()
