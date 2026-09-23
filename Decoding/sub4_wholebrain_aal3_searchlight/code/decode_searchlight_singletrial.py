#!/usr/bin/env python3
"""Whole-brain searchlight cross-source decoding via nilearn.decoding.SearchLight.

Reuses this project's cross-source contrast/fold logic from
decode_roi_singletrial.py (10-fold leave-one-run-out, per-fold train-only
normalization, linear SVC C=1), but swaps ROI-mask feature extraction for
nilearn's sphere-based SearchLight, which handles neighbor lookup and
per-sphere parallelization natively.

Design notes (see TASK_BRIEF.md section 4 and 6, and this task's plan file):
- nilearn.decoding.SearchLight's `radius` parameter is in **millimeters**,
  computed in world/affine space (correctly handles the anisotropic native
  voxel size, ~1.7966x1.7966x2.25mm, unlike a naive voxel-index radius).
  This script's --radius-voxels is a nominal voxel count converted to mm
  using the in-plane voxel size, recorded alongside the resulting mm value
  in provenance so results are interpretable either way.
- Per-fold train-only normalization is achieved by passing a
  sklearn Pipeline(StandardScaler, SVC) as the estimator: nilearn calls
  sklearn's cross_val_score() per sphere, which fits the whole pipeline
  (including the scaler) on the training fold only and transforms the test
  fold with those fitted train statistics -- equivalent in effect to the ROI
  script's manual per-fold mean/std normalization. One deliberate deviation:
  the ROI script explicitly drops training voxels with near-zero variance
  (std <= 1e-7) before fitting; StandardScaler instead leaves such features
  in place (internally clipping the tiny variance to avoid divide-by-zero).
  This is immaterial in practice -- single-trial GLM betas essentially never
  have a truly constant voxel within a small in-mask sphere -- but is
  recorded as a caveat below rather than silently assumed identical.
- Cross-source train/test splits are precomputed as a list of
  (train_idx, test_idx) tuples over the full 0..599 trial row range (same
  contrast logic as decode_roi_singletrial.py) and passed to nilearn as
  `cv=folds`; sklearn wraps a plain list of index pairs via
  _CVIterableWrapper, which is safely re-iterated once per sphere (not a
  one-shot generator).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import h5py
import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
import sklearn
from nilearn.decoding import SearchLight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CROSS_CONTRASTS = (
    ("train_natural_pleasant_vs_neutral_test_ai", "cross", "natural", "ai", "pleasant"),
    ("train_ai_pleasant_vs_neutral_test_natural", "cross", "ai", "natural", "pleasant"),
    ("train_natural_unpleasant_vs_neutral_test_ai", "cross", "natural", "ai", "unpleasant"),
    ("train_ai_unpleasant_vs_neutral_test_natural", "cross", "ai", "natural", "unpleasant"),
)
CONTRAST_LOOKUP = {name: (kind, train_source, test_source, positive) for name, kind, train_source, test_source, positive in CROSS_CONTRASTS}


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


def validate_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"beta_index", "run", "source", "valence", "image_id"}
    if not required.issubset(frame.columns) or len(frame) != 600:
        raise RuntimeError(f"Invalid trial manifest columns/length: {frame.columns.tolist()}")
    if frame["beta_index"].astype(int).tolist() != list(range(600)):
        raise RuntimeError("Manifest beta index is not exactly 0..599")
    if sorted(frame["run"].astype(int).unique()) != list(range(1, 11)):
        raise RuntimeError("Manifest does not contain runs 1..10")
    return frame.reset_index(drop=True)


def build_folds(
    manifest: pd.DataFrame, train_source: str, test_source: str, positive: str
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    run_values = manifest["run"].to_numpy(dtype=int)
    source_values = manifest["source"].astype(str).str.lower().to_numpy()
    valence_values = manifest["valence"].astype(str).str.lower().to_numpy()
    y = (valence_values == positive).astype(np.int8)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
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
        if len(train_index) != 180 or len(test_index) != 20:
            raise RuntimeError(f"run {held_run}: train/test={len(train_index)}/{len(test_index)}")
        if not np.array_equal(np.bincount(y[train_index], minlength=2), [90, 90]):
            raise RuntimeError("Training classes are imbalanced")
        if not np.array_equal(np.bincount(y[test_index], minlength=2), [10, 10]):
            raise RuntimeError("Test classes are imbalanced")
        folds.append((train_index, test_index))
    return folds, y


def build_4d_image(matrix: np.ndarray, flat_mask_indices: np.ndarray, mask_image: nib.Nifti1Image) -> nib.Nifti1Image:
    n_trials = matrix.shape[0]
    volume = np.zeros(mask_image.shape + (n_trials,), dtype=np.float32)
    flat_volume = volume.reshape(-1, n_trials)
    flat_volume[flat_mask_indices] = matrix.T
    header = mask_image.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(volume, mask_image.affine, header)


def append_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": pd.Timestamp.utcnow().isoformat(), **record}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True, help="mni_res_native branch directory")
    parser.add_argument(
        "--contrasts", nargs="+", default=[name for name, *_ in CROSS_CONTRASTS], choices=list(CONTRAST_LOOKUP)
    )
    parser.add_argument("--radius-voxels", type=float, required=True, help="nominal radius in in-plane voxels; converted to mm")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    mask_image = nib.load(args.input_root / "analysis_mask.nii.gz")
    mask = np.asarray(mask_image.dataobj) > 0
    flat = np.load(args.input_root / "flat_mask_indices.npy").astype(np.int64)
    if not np.array_equal(flat, np.flatnonzero(mask.ravel())):
        raise RuntimeError("GLMsingle flat-mask index set is inconsistent")

    matrix = load_hdf5_matrix(args.input_root / "glmsingle" / "TYPED_FITHRF_GLMDENOISE_RR.hdf5", len(flat))
    imgs = build_4d_image(matrix, flat, mask_image)

    manifest = pd.read_csv(args.input_root / "trial_manifest.tsv", sep="\t").sort_values("beta_index")
    manifest = validate_manifest(manifest)

    voxel_sizes_mm = [float(value) for value in nib.affines.voxel_sizes(mask_image.affine)]
    inplane_voxel_mm = voxel_sizes_mm[0]
    radius_mm = args.radius_voxels * inplane_voxel_mm

    common = {
        "subject": "Sub4",
        "estimator": "glmsingle_typed",
        "branch": "mni_res_native",
        "smoothing_fwhm_mm": 0,
        "cv": "leave-one-run-out",
        "classifier": "linear SVC C=1 (Pipeline: StandardScaler + SVC)",
        "normalization": "per-voxel mean/std fit on outer-training trials only (via Pipeline fit inside cross_val_score)",
        "radius_voxels_nominal": args.radius_voxels,
        "radius_mm": radius_mm,
        "voxel_sizes_mm": voxel_sizes_mm,
        "n_mask_voxels": int(len(flat)),
        "caveat": (
            "StandardScaler does not drop near-zero-variance training voxels "
            "the way decode_roi_singletrial.py's manual normalization does; "
            "considered immaterial for single-trial GLM betas within a small "
            "in-mask sphere."
        ),
    }

    append_ledger(
        args.ledger,
        {
            "event": "JOB_START",
            "job": "searchlight",
            "output": str(args.output),
            "radius_voxels_nominal": args.radius_voxels,
            "radius_mm": radius_mm,
            "contrasts": args.contrasts,
            "n_jobs": args.n_jobs,
        },
    )

    estimator_template = Pipeline(
        [("scale", StandardScaler()), ("svc", SVC(kernel="linear", C=1.0, cache_size=2048))]
    )

    per_contrast_timing = {}
    for name in args.contrasts:
        kind, train_source, test_source, positive = CONTRAST_LOOKUP[name]
        folds, y = build_folds(manifest, train_source, test_source, positive)
        searchlight = SearchLight(
            mask_img=mask_image,
            radius=radius_mm,
            estimator=estimator_template,
            n_jobs=args.n_jobs,
            scoring="accuracy",
            cv=folds,
            verbose=1,
        )
        t0 = time.time()
        searchlight.fit(imgs, y)
        elapsed = time.time() - t0
        per_contrast_timing[name] = elapsed

        scores = searchlight.scores_.astype(np.float32)
        scores[~mask] = np.nan
        out_path = args.output / f"{name}_accuracy.nii.gz"
        nib.save(nib.Nifti1Image(scores, mask_image.affine, mask_image.header), out_path)

        append_ledger(
            args.ledger,
            {
                "event": "JOB_END",
                "job": "searchlight",
                "contrast": name,
                "elapsed_seconds": elapsed,
                "output": str(out_path),
                "mean_accuracy": float(np.nanmean(scores)),
                "max_accuracy": float(np.nanmax(scores)),
            },
        )
        print(f"SEARCHLIGHT_CONTRAST_COMPLETE {name} elapsed={elapsed:.1f}s mean_acc={np.nanmean(scores):.4f}", flush=True)

    provenance = {
        **common,
        "input_root": str(args.input_root),
        "contrasts_run": args.contrasts,
        "per_contrast_elapsed_seconds": per_contrast_timing,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "nibabel": nib.__version__,
        "nilearn": nilearn.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    append_ledger(args.ledger, {"event": "STAGE_COMPLETE", "job": "searchlight", "output": str(args.output)})
    print("SEARCHLIGHT_COMPLETE")


if __name__ == "__main__":
    main()
