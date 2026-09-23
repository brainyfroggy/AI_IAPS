#!/usr/bin/env python3
"""Single-trial LORO decoding in the 17 frozen Kastner/Wang ROIs."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from nibabel.processing import resample_from_to
from scipy.ndimage import gaussian_filter
from sklearn import __version__ as sklearn_version
from sklearn.svm import SVC


ROI_ORDER = (
    "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "V3a", "V3b",
    "IPS", "LO1", "LO2", "hMT", "VO1", "VO2", "PHC1", "PHC2",
)
ROI_LABEL_NAMES = {
    "V1v": ("V1v",), "V1d": ("V1d",), "V2v": ("V2v",),
    "V2d": ("V2d",), "V3v": ("V3v",), "V3d": ("V3d",),
    "hV4": ("hV4",), "V3a": ("V3a",), "V3b": ("V3b",),
    "IPS": ("IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5"),
    "LO1": ("LO1",), "LO2": ("LO2",), "hMT": ("hMT",),
    "VO1": ("VO1",), "VO2": ("VO2",), "PHC1": ("PHC1",),
    "PHC2": ("PHC2",),
}
CONTRASTS = (
    ("within_natural_pleasant_vs_neutral", "within", "natural", "natural", "pleasant"),
    ("within_ai_pleasant_vs_neutral", "within", "ai", "ai", "pleasant"),
    ("within_natural_unpleasant_vs_neutral", "within", "natural", "natural", "unpleasant"),
    ("within_ai_unpleasant_vs_neutral", "within", "ai", "ai", "unpleasant"),
    ("train_natural_pleasant_vs_neutral_test_ai", "cross", "natural", "ai", "pleasant"),
    ("train_ai_pleasant_vs_neutral_test_natural", "cross", "ai", "natural", "pleasant"),
    ("train_natural_unpleasant_vs_neutral_test_ai", "cross", "natural", "ai", "unpleasant"),
    ("train_ai_unpleasant_vs_neutral_test_natural", "cross", "ai", "natural", "unpleasant"),
)


def same_grid(left: nib.spatialimages.SpatialImage, right: nib.spatialimages.SpatialImage) -> bool:
    return left.shape[:3] == right.shape[:3] and np.allclose(
        left.affine, right.affine, atol=1e-5, rtol=0
    )


def parse_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                value = int(float(parts[0]))
            except ValueError:
                continue
            if value > 0:
                labels[parts[1]] = value
    return labels


def roi_indices(
    atlas_path: Path,
    labels_path: Path,
    reference: nib.spatialimages.SpatialImage,
    analysis_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
    atlas = nib.load(atlas_path)
    was_resampled = not same_grid(atlas, reference)
    if was_resampled:
        atlas = resample_from_to(
            atlas, (reference.shape[:3], reference.affine), order=0, mode="constant", cval=0
        )
    values = np.rint(np.asarray(atlas.dataobj)).astype(np.int32)
    labels = parse_labels(labels_path)
    full: dict[str, np.ndarray] = {}
    counts = []
    for roi in ROI_ORDER:
        missing = [name for name in ROI_LABEL_NAMES[roi] if name not in labels]
        if missing:
            raise RuntimeError(f"Missing atlas labels for {roi}: {missing}")
        mask = np.isin(values, [labels[name] for name in ROI_LABEL_NAMES[roi]])
        before = int(mask.sum())
        indices = np.flatnonzero((mask & analysis_mask).ravel()).astype(np.int64)
        if len(indices) < 10:
            raise RuntimeError(f"{roi} has only {len(indices)} analysis voxels")
        full[roi] = indices
        counts.append(
            {
                "roi": roi,
                "atlas_voxels_on_target_grid": before,
                "analysis_voxels": int(len(indices)),
                "affine_only_resampling_in_decoder": was_resampled,
            }
        )
    union = np.unique(np.concatenate(list(full.values())))
    positions = {roi: np.searchsorted(union, full[roi]) for roi in ROI_ORDER}
    return union, positions, counts


def load_hdf5_matrix(path: Path, n_voxels: int) -> np.ndarray:
    matrix = np.empty((600, n_voxels), dtype=np.float32)
    with h5py.File(path, "r") as handle:
        dataset = handle["betasmd"]
        if dataset.shape != (n_voxels, 1, 1, 600):
            raise RuntimeError(f"Unexpected Type-D shape: {dataset.shape}")
        block = 8192
        for start in range(0, n_voxels, block):
            stop = min(start + block, n_voxels)
            matrix[:, start:stop] = np.asarray(dataset[start:stop, 0, 0, :]).T
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("GLMsingle betas contain nonfinite values")
    return matrix


def load_glmsingle(
    root: Path,
    atlas: Path,
    labels: Path,
    smoothing: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict], pd.DataFrame, nib.Nifti1Image]:
    mask_image = nib.load(root / "analysis_mask.nii.gz")
    mask = np.asarray(mask_image.dataobj) > 0
    flat = np.load(root / "flat_mask_indices.npy").astype(np.int64)
    if not np.array_equal(flat, np.flatnonzero(mask.ravel())):
        raise RuntimeError("GLMsingle flat-mask index set is inconsistent")
    union, positions, counts = roi_indices(atlas, labels, mask_image, mask)
    selected = np.searchsorted(flat, union)
    if np.any(selected >= len(flat)) or not np.array_equal(flat[selected], union):
        raise RuntimeError("ROI union is not contained in the GLMsingle mask")

    matrix = load_hdf5_matrix(
        root / "glmsingle" / "TYPED_FITHRF_GLMDENOISE_RR.hdf5", len(flat)
    )
    if smoothing == 0:
        data = matrix[:, selected].copy()
    else:
        sigma_mm = smoothing / math.sqrt(8 * math.log(2))
        sigma_vox = sigma_mm / nib.affines.voxel_sizes(mask_image.affine)
        denominator = gaussian_filter(
            mask.astype(np.float32), sigma=sigma_vox, mode="constant", cval=0.0
        )
        denominator = denominator.ravel()[union]
        if np.any(denominator <= np.finfo(np.float32).eps):
            raise RuntimeError("Zero mask-normalized smoothing denominator")
        data = np.empty((600, len(union)), dtype=np.float32)
        volume = np.zeros(mask.shape, dtype=np.float32)
        for trial in range(600):
            volume.fill(0)
            volume.ravel()[flat] = matrix[trial]
            smoothed = gaussian_filter(
                volume, sigma=sigma_vox, mode="constant", cval=0.0
            )
            data[trial] = smoothed.ravel()[union] / denominator
            if (trial + 1) % 100 == 0:
                print(f"smoothed {trial + 1}/600 betas at {smoothing:g} mm", flush=True)
    manifest = pd.read_csv(root / "trial_manifest.tsv", sep="\t").sort_values("beta_index")
    return data, positions, counts, manifest, mask_image


def load_spm(
    root: Path,
    manifest_path: Path,
    atlas: Path,
    labels: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict], pd.DataFrame, nib.Nifti1Image]:
    index = pd.read_csv(root / "trial_beta_index.tsv", sep="\t").sort_values("beta_index")
    if index["beta_index"].tolist() != list(range(600)):
        raise RuntimeError("SPM trial-beta index is not exactly 0..599")
    paths = [Path(value) for value in index["beta_file"]]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("At least one indexed SPM trial beta is missing")
    reference = nib.load(paths[0])
    analysis_mask = np.ones(reference.shape[:3], dtype=bool)
    union, positions, counts = roi_indices(atlas, labels, reference, analysis_mask)
    data = np.empty((600, len(union)), dtype=np.float32)
    for trial, path in enumerate(paths):
        image = nib.load(path)
        if not same_grid(image, reference):
            raise RuntimeError(f"SPM beta grid mismatch: {path}")
        data[trial] = image.get_fdata(dtype=np.float32).ravel()[union]
        if (trial + 1) % 100 == 0:
            print(f"loaded {trial + 1}/600 SPM trial betas", flush=True)
    manifest = pd.read_csv(manifest_path, sep="\t").sort_values("beta_index")
    return data, positions, counts, manifest, reference


def load_transformed_4d(
    root: Path,
    atlas: Path,
    labels: Path,
    smoothing: float,
    apply_post_beta_smoothing: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict], pd.DataFrame, nib.Nifti1Image]:
    """Load native-estimated betas after explicit normalization to MNI space."""
    beta_image = nib.load(root / "transformed_betas.nii.gz")
    mask_image = nib.load(root / "transformed_mask.nii.gz")
    if beta_image.shape[:3] != mask_image.shape[:3] or not np.allclose(
        beta_image.affine, mask_image.affine, atol=1e-5, rtol=0
    ):
        raise RuntimeError("Transformed beta/mask grids differ")
    if beta_image.shape[3:] != (600,):
        raise RuntimeError(f"Expected 600 transformed betas; got {beta_image.shape}")
    mask = np.asarray(mask_image.dataobj) > 0
    union, positions, counts = roi_indices(atlas, labels, mask_image, mask)
    data = np.empty((600, len(union)), dtype=np.float32)

    if apply_post_beta_smoothing and smoothing > 0:
        sigma_mm = smoothing / math.sqrt(8 * math.log(2))
        sigma_vox = sigma_mm / nib.affines.voxel_sizes(mask_image.affine)
        denominator = gaussian_filter(
            mask.astype(np.float32), sigma=sigma_vox, mode="constant", cval=0.0
        ).ravel()[union]
        if np.any(denominator <= np.finfo(np.float32).eps):
            raise RuntimeError("Zero mask-normalized smoothing denominator")
    else:
        denominator = None

    for trial in range(600):
        volume = np.asarray(beta_image.dataobj[..., trial], dtype=np.float32)
        if denominator is None:
            data[trial] = volume.ravel()[union]
        else:
            smoothed = gaussian_filter(
                volume * mask, sigma=sigma_vox, mode="constant", cval=0.0
            )
            data[trial] = smoothed.ravel()[union] / denominator
        if (trial + 1) % 100 == 0:
            print(f"loaded transformed beta {trial + 1}/600", flush=True)
    if not np.all(np.isfinite(data)):
        raise RuntimeError("Transformed beta features contain nonfinite values")
    manifest = pd.read_csv(root / "trial_manifest.tsv", sep="\t").sort_values("beta_index")
    return data, positions, counts, manifest, mask_image


def validate_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"beta_index", "run", "source", "valence", "image_id"}
    if not required.issubset(frame.columns) or len(frame) != 600:
        raise RuntimeError(f"Invalid trial manifest columns/length: {frame.columns.tolist()}")
    if frame["beta_index"].astype(int).tolist() != list(range(600)):
        raise RuntimeError("Manifest beta index is not exactly 0..599")
    if sorted(frame["run"].astype(int).unique()) != list(range(1, 11)):
        raise RuntimeError("Manifest does not contain runs 1..10")
    return frame.reset_index(drop=True)


def decode(
    data: np.ndarray,
    positions: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    common: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_values = manifest["run"].to_numpy(dtype=int)
    source_values = manifest["source"].astype(str).str.lower().to_numpy()
    valence_values = manifest["valence"].astype(str).str.lower().to_numpy()
    subject_rows = []
    fold_rows = []
    for roi in ROI_ORDER:
        roi_data = data[:, positions[roi]]
        finite_global = np.all(np.isfinite(roi_data), axis=0)
        roi_data = roi_data[:, finite_global]
        if roi_data.shape[1] < 10:
            raise RuntimeError(f"{roi}: fewer than ten finite features")
        for name, kind, train_source, test_source, positive in CONTRASTS:
            accuracies = []
            correct = 0
            observations = 0
            for held_run in range(1, 11):
                train_mask = (
                    (source_values == train_source)
                    & (run_values != held_run)
                    & np.isin(valence_values, [positive, "neutral"])
                )
                test_mask = (
                    (source_values == test_source)
                    & (run_values == held_run)
                    & np.isin(valence_values, [positive, "neutral"])
                )
                train_index = np.flatnonzero(train_mask)
                test_index = np.flatnonzero(test_mask)
                y_train = (valence_values[train_index] == positive).astype(np.int8)
                y_test = (valence_values[test_index] == positive).astype(np.int8)
                if len(train_index) != 180 or len(test_index) != 20:
                    raise RuntimeError(
                        f"{name} run {held_run}: train/test={len(train_index)}/{len(test_index)}"
                    )
                if not np.array_equal(np.bincount(y_train, minlength=2), [90, 90]):
                    raise RuntimeError("Training classes are imbalanced")
                if not np.array_equal(np.bincount(y_test, minlength=2), [10, 10]):
                    raise RuntimeError("Test classes are imbalanced")

                x_train = roi_data[train_index]
                x_test = roi_data[test_index]
                mean = x_train.mean(axis=0, dtype=np.float64)
                std = x_train.std(axis=0, ddof=0, dtype=np.float64)
                valid = np.isfinite(mean) & np.isfinite(std) & (std > 1e-7)
                if int(valid.sum()) < 10:
                    raise RuntimeError(f"{roi}/{name}/run-{held_run}: too few train-variable features")
                x_train = ((x_train[:, valid] - mean[valid]) / std[valid]).astype(np.float32)
                x_test = ((x_test[:, valid] - mean[valid]) / std[valid]).astype(np.float32)
                classifier = SVC(kernel="linear", C=1.0, cache_size=2048)
                classifier.fit(x_train, y_train)
                prediction = classifier.predict(x_test)
                fold_correct = int(np.sum(prediction == y_test))
                accuracy = fold_correct / len(y_test)
                accuracies.append(accuracy)
                correct += fold_correct
                observations += len(y_test)
                fold_rows.append(
                    {
                        **common,
                        "roi": roi,
                        "contrast": name,
                        "contrast_kind": kind,
                        "train_source": train_source,
                        "test_source": test_source,
                        "positive_valence": positive,
                        "held_out_run": held_run,
                        "n_train": len(train_index),
                        "n_test": len(test_index),
                        "n_features": int(valid.sum()),
                        "n_correct": fold_correct,
                        "accuracy": accuracy,
                    }
                )
            subject_rows.append(
                {
                    **common,
                    "roi": roi,
                    "contrast": name,
                    "contrast_kind": kind,
                    "train_source": train_source,
                    "test_source": test_source,
                    "positive_valence": positive,
                    "n_folds": 10,
                    "n_test_observations": observations,
                    "accuracy": correct / observations,
                    "mean_fold_accuracy": float(np.mean(accuracies)),
                }
            )
    return pd.DataFrame(subject_rows), pd.DataFrame(fold_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", choices=["glmsingle_typed", "spm_lsa_historical_model"], required=True)
    parser.add_argument(
        "--input-format",
        choices=["native_model", "transformed_4d"],
        default="native_model",
    )
    parser.add_argument("--branch", required=True)
    parser.add_argument("--smoothing-mm", type=float, choices=[0, 3, 5, 8], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--atlas-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    if args.input_format == "transformed_4d":
        data, positions, counts, manifest, reference = load_transformed_4d(
            args.input_root,
            args.atlas,
            args.atlas_labels,
            args.smoothing_mm,
            apply_post_beta_smoothing=args.estimator == "glmsingle_typed",
        )
        if args.estimator == "glmsingle_typed":
            smoothing_location = "post-normalization MNI single-trial beta"
        else:
            smoothing_location = "pre-GLM native BOLD; betas normalized to MNI after GLM"
    elif args.estimator == "glmsingle_typed":
        data, positions, counts, manifest, reference = load_glmsingle(
            args.input_root, args.atlas, args.atlas_labels, args.smoothing_mm
        )
        smoothing_location = "post-GLM beta"
    else:
        if args.manifest is None:
            raise ValueError("--manifest is required for SPM")
        data, positions, counts, manifest, reference = load_spm(
            args.input_root, args.manifest, args.atlas, args.atlas_labels
        )
        smoothing_location = "pre-GLM BOLD"

    manifest = validate_manifest(manifest)
    np.save(args.output / "roi_union_features.npy", data)
    common = {
        "subject": "Sub4",
        "estimator": args.estimator,
        "branch": args.branch,
        "smoothing_fwhm_mm": args.smoothing_mm,
        "smoothing_location": smoothing_location,
        "trial_unit": "single trial",
        "cv": "leave-one-run-out",
        "classifier": "linear SVC C=1",
        "normalization": "per-voxel mean/std fit on outer-training trials only",
        "input_format": args.input_format,
    }
    subject, folds = decode(data, positions, manifest, common)
    subject.to_csv(args.output / "subject_results.csv", index=False)
    folds.to_csv(args.output / "fold_results.csv", index=False)
    pd.DataFrame(counts).assign(**common).to_csv(args.output / "roi_counts.csv", index=False)
    provenance = {
        **common,
        "input_root": str(args.input_root),
        "manifest": str(args.manifest) if args.manifest else str(args.input_root / "trial_manifest.tsv"),
        "atlas": str(args.atlas),
        "atlas_labels": str(args.atlas_labels),
        "reference_shape": list(reference.shape[:3]),
        "reference_affine": np.asarray(reference.affine).tolist(),
        "n_union_features": int(data.shape[1]),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "nibabel": nib.__version__,
        "scikit_learn": sklearn_version,
        "caveat": "Runs are held out, but repeated stimulus identities are not held out across runs.",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"ROI_SINGLETRIAL_COMPLETE {args.estimator} {args.branch} {args.smoothing_mm:g}mm")


if __name__ == "__main__":
    main()
