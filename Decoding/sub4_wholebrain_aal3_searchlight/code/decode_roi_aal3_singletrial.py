#!/usr/bin/env python3
"""Single-trial LORO cross-source decoding in the 75 grouped AAL3 ROIs.

Fork of sub4_spatial_sensitivity_32/code/decode_roi_singletrial.py, adapted
from the 17 Kastner/Wang visual ROIs to the whole-brain AAL3 atlas (75
groups from roi_labels.csv), and restricted to the 4 cross-source contrasts
per the 2026-08-05 TASK_BRIEF.md revision (within-source contrasts dropped).
Classifier/CV/normalization/output conventions are unchanged from the
original script. Smoothing is fixed at 0mm (no CLI flag) per this task's
scope decision -- betas are used unsmoothed, and the underlying fMRIPrep
BOLD was never smoothed either.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from nibabel.processing import resample_from_to
from sklearn import __version__ as sklearn_version
from sklearn.svm import SVC


CROSS_CONTRASTS = (
    ("train_natural_pleasant_vs_neutral_test_ai", "cross", "natural", "ai", "pleasant"),
    ("train_ai_pleasant_vs_neutral_test_natural", "cross", "ai", "natural", "pleasant"),
    ("train_natural_unpleasant_vs_neutral_test_ai", "cross", "natural", "ai", "unpleasant"),
    ("train_ai_unpleasant_vs_neutral_test_natural", "cross", "ai", "natural", "unpleasant"),
)

MIN_ROI_VOXELS = 10


def same_grid(left: nib.spatialimages.SpatialImage, right: nib.spatialimages.SpatialImage) -> bool:
    return left.shape[:3] == right.shape[:3] and np.allclose(
        left.affine, right.affine, atol=1e-5, rtol=0
    )


def load_roi_groups(labels_csv: Path) -> dict[str, list[int]]:
    frame = pd.read_csv(labels_csv)
    required = {"id", "roi_name", "group_id", "group_roi_name"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"roi_labels.csv missing columns: {frame.columns.tolist()}")
    ordered = frame.sort_values(["group_id", "id"])
    first_names = ordered.groupby("group_id")["group_roi_name"].first()
    name_counts = first_names.value_counts()
    duplicated_names = set(name_counts[name_counts > 1].index)
    groups: dict[str, list[int]] = {}
    for group_id, rows in ordered.groupby("group_id", sort=True):
        names = rows["group_roi_name"].unique()
        if len(names) != 1:
            raise RuntimeError(f"group_id {group_id} has inconsistent group_roi_name: {names}")
        base_name = str(names[0])
        # group_roi_name is not always unique across group_id (e.g. AAL3's
        # Cerebellum crus/hemisphere group and its separate vermis-adjacent
        # lobule group share the display name "Cerebellum") -- disambiguate
        # by group_id so distinct groups never collapse into one dict key.
        key = f"{base_name}_{int(group_id)}" if base_name in duplicated_names else base_name
        groups[key] = [int(value) for value in rows["id"].tolist()]
    if len(groups) != frame["group_id"].nunique():
        raise RuntimeError(
            f"ROI group key collision: {len(groups)} keys for {frame['group_id'].nunique()} groups"
        )
    return groups


def roi_indices(
    atlas_path: Path,
    groups: dict[str, list[int]],
    reference: nib.spatialimages.SpatialImage,
    analysis_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict], list[str]]:
    atlas = nib.load(atlas_path)
    if not same_grid(atlas, reference):
        raise RuntimeError(
            f"AAL3 atlas grid does not match beta reference grid: {atlas_path} versus reference"
        )
    values = np.rint(np.asarray(atlas.dataobj)).astype(np.int32)
    full: dict[str, np.ndarray] = {}
    counts = []
    dropped: list[str] = []
    for roi, label_ids in groups.items():
        mask = np.isin(values, label_ids)
        before = int(mask.sum())
        indices = np.flatnonzero((mask & analysis_mask).ravel()).astype(np.int64)
        counts.append(
            {
                "roi": roi,
                "atlas_voxels_on_target_grid": before,
                "analysis_voxels": int(len(indices)),
            }
        )
        if len(indices) < MIN_ROI_VOXELS:
            # Known AAL3 quirk: the generic "Thalamus" group (label ids 81/82)
            # is a placeholder with zero image voxels -- AAL3 replaces it with
            # 16 named subdivisions (Thal_AV, Thal_VA, ...) which are present
            # and decoded normally. Skip any group under threshold rather than
            # hard-failing the whole run, since this is expected atlas
            # structure, not a data error (confirmed via prepare_aal3_atlas.py
            # provenance.json before this script was written).
            print(f"WARNING: dropping ROI '{roi}' with only {len(indices)} analysis voxels", flush=True)
            dropped.append(roi)
            continue
        full[roi] = indices
    union = np.unique(np.concatenate(list(full.values())))
    positions = {roi: np.searchsorted(union, full[roi]) for roi in full}
    return union, positions, counts, dropped


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
    groups: dict[str, list[int]],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict], list[str], pd.DataFrame, nib.Nifti1Image]:
    mask_image = nib.load(root / "analysis_mask.nii.gz")
    mask = np.asarray(mask_image.dataobj) > 0
    flat = np.load(root / "flat_mask_indices.npy").astype(np.int64)
    if not np.array_equal(flat, np.flatnonzero(mask.ravel())):
        raise RuntimeError("GLMsingle flat-mask index set is inconsistent")
    union, positions, counts, dropped = roi_indices(atlas, groups, mask_image, mask)
    selected = np.searchsorted(flat, union)
    if np.any(selected >= len(flat)) or not np.array_equal(flat[selected], union):
        raise RuntimeError("ROI union is not contained in the GLMsingle mask")

    matrix = load_hdf5_matrix(
        root / "glmsingle" / "TYPED_FITHRF_GLMDENOISE_RR.hdf5", len(flat)
    )
    data = matrix[:, selected].copy()
    manifest = pd.read_csv(root / "trial_manifest.tsv", sep="\t").sort_values("beta_index")
    return data, positions, counts, dropped, manifest, mask_image


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
    for roi in sorted(positions):
        roi_data = data[:, positions[roi]]
        finite_global = np.all(np.isfinite(roi_data), axis=0)
        roi_data = roi_data[:, finite_global]
        if roi_data.shape[1] < MIN_ROI_VOXELS:
            raise RuntimeError(f"{roi}: fewer than {MIN_ROI_VOXELS} finite features")
        for name, kind, train_source, test_source, positive in CROSS_CONTRASTS:
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
                if int(valid.sum()) < MIN_ROI_VOXELS:
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
    parser.add_argument("--input-root", type=Path, required=True, help="mni_res_native branch directory")
    parser.add_argument("--atlas", type=Path, required=True, help="aal3_native_labels.nii.gz from prepare_aal3_atlas.py")
    parser.add_argument("--atlas-labels", type=Path, required=True, help="roi_labels.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    groups = load_roi_groups(args.atlas_labels)
    data, positions, counts, dropped, manifest, reference = load_glmsingle(
        args.input_root, args.atlas, groups
    )
    manifest = validate_manifest(manifest)
    np.save(args.output / "roi_union_features.npy", data)
    common = {
        "subject": "Sub4",
        "estimator": "glmsingle_typed",
        "branch": "mni_res_native",
        "smoothing_fwhm_mm": 0,
        "smoothing_location": "post-GLM beta",
        "trial_unit": "single trial",
        "cv": "leave-one-run-out",
        "classifier": "linear SVC C=1",
        "normalization": "per-voxel mean/std fit on outer-training trials only",
        "atlas": "AAL3v1 2mm, grouped (75 ROIs, roi_labels.csv)",
    }
    subject, folds = decode(data, positions, manifest, common)
    subject.to_csv(args.output / "subject_results.csv", index=False)
    folds.to_csv(args.output / "fold_results.csv", index=False)
    pd.DataFrame(counts).assign(**common).to_csv(args.output / "roi_counts.csv", index=False)
    provenance = {
        **common,
        "input_root": str(args.input_root),
        "manifest": str(args.input_root / "trial_manifest.tsv"),
        "atlas": str(args.atlas),
        "atlas_labels": str(args.atlas_labels),
        "reference_shape": list(reference.shape[:3]),
        "reference_affine": np.asarray(reference.affine).tolist(),
        "n_groups_total": len(groups),
        "n_groups_decoded": len(positions),
        "n_groups_dropped": dropped,
        "n_union_features": int(data.shape[1]),
        "contrasts": [name for name, *_ in CROSS_CONTRASTS],
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
    print(f"ROI_AAL3_SINGLETRIAL_COMPLETE groups={len(positions)} dropped={dropped}")


if __name__ == "__main__":
    main()
