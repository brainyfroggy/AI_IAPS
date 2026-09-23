#!/usr/bin/env python3
"""Leakage-safe whole-brain searchlight comparison for the AI-IAPS pilot.

This program applies *the same* decoder to the legacy SPM LS-A beta array and
to masked GLMsingle Type-D betas.  Its primary cross-validation is
leave-one-run-out (LORO).  An optional identity-blocked sensitivity analysis
holds out complete matched natural/AI stimulus families.

The implementation is intentionally independent of nilearn's SearchLight.
For binary diagonal LDA (or standardized nearest centroid), the decision
function is a sum of voxelwise evidence.  A sparse center-by-voxel neighborhood
matrix therefore evaluates thousands of 5-mm spheres at once while preserving
the exact estimator that would be fit sphere by sphere.

Important interpretation details
--------------------------------
* Every scaling statistic and every classifier parameter is estimated from the
  training fold only.
* Smoothing is applied to beta images independently, with mask-normalized
  Gaussian smoothing.  It never mixes trials or folds.
* Accuracy outside evaluated searchlight centers is NaN, not zero.
* The output directory must not already exist.  This script has no overwrite
  option by design.
* Legacy and GLMsingle maps can have different MNI grids.  Scalar/spatial
  summaries are directly comparable; voxelwise subtraction is not performed.

Example
-------
python decode_compare.py \
  --subject 4 \
  --glmsingle-dir /path/to/sub-04/glmsingle_run \
  --output /path/to/a/new/output_directory \
  --new-smoothing-mm 0 3 \
  --cv leave-one-run-out identity-groupkfold

Run ``python decode_compare.py --self-test`` for a synthetic end-to-end test.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEGACY_ROOT = PROJECT_ROOT / "Decoding" / "data" / "30subs"
DEFAULT_LEGACY_MASK = PROJECT_ROOT / "Decoding" / "data" / "mask.nii"
DEFAULT_LEGACY_MANIFEST = PROJECT_ROOT / "Decoding" / "stimuli_600trials.csv"
TYPE_D_NAME = "TYPED_FITHRF_GLMDENOISE_RR.hdf5"
CHANCE = 0.5
HDF5_TARGET_READ_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class Contrast:
    """A binary train/test source contrast; class 1 is the emotional class."""

    name: str
    kind: str
    train_source: str
    test_source: str
    positive_valence: str
    negative_valence: str = "neutral"


CONTRASTS: tuple[Contrast, ...] = (
    Contrast(
        "within_natural_pleasant_vs_neutral",
        "within",
        "natural",
        "natural",
        "pleasant",
    ),
    Contrast(
        "within_ai_pleasant_vs_neutral",
        "within",
        "ai",
        "ai",
        "pleasant",
    ),
    Contrast(
        "within_natural_unpleasant_vs_neutral",
        "within",
        "natural",
        "natural",
        "unpleasant",
    ),
    Contrast(
        "within_ai_unpleasant_vs_neutral",
        "within",
        "ai",
        "ai",
        "unpleasant",
    ),
    Contrast(
        "train_natural_pleasant_vs_neutral_test_ai",
        "cross",
        "natural",
        "ai",
        "pleasant",
    ),
    Contrast(
        "train_ai_pleasant_vs_neutral_test_natural",
        "cross",
        "ai",
        "natural",
        "pleasant",
    ),
    Contrast(
        "train_natural_unpleasant_vs_neutral_test_ai",
        "cross",
        "natural",
        "ai",
        "unpleasant",
    ),
    Contrast(
        "train_ai_unpleasant_vs_neutral_test_natural",
        "cross",
        "ai",
        "natural",
        "unpleasant",
    ),
)
CONTRAST_BY_NAME = {item.name: item for item in CONTRASTS}


@dataclass(frozen=True)
class Fold:
    fold_id: str
    held_out: str
    train_index: np.ndarray
    test_index: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class BranchSpec:
    name: str
    kind: str
    beta_path: Path
    mask_path: Path
    manifest_path: Path
    smoothing_fwhm_mm: tuple[float, ...]
    flat_indices_path: Path | None = None
    input_smoothing_note: str = "unknown"


@dataclass
class BetaDataset:
    data: np.ndarray  # trials x masked voxels, float32
    mask: np.ndarray
    flat_indices: np.ndarray
    reference: nib.Nifti1Image


@dataclass(frozen=True)
class FeatureModel:
    """Raw-space form of a classifier fit after training-only z-scoring."""

    weight_raw: np.ndarray
    intercept_raw: np.ndarray
    valid_voxels: np.ndarray
    n_class_0: int
    n_class_1: int


@dataclass
class Neighborhoods:
    """Compressed searchlight neighborhoods without permanently stored data."""

    center_columns: np.ndarray
    center_flat_indices: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    n_voxels: int

    @property
    def n_centers(self) -> int:
        return int(len(self.center_columns))

    def sparse_block(self, start: int, stop: int) -> csr_matrix:
        first = int(self.indptr[start])
        last = int(self.indptr[stop])
        local_indptr = self.indptr[start : stop + 1] - first
        local_indices = self.indices[first:last]
        # float32 is intentional: it keeps sparse matrix products float32.
        data = np.ones(last - first, dtype=np.float32)
        return csr_matrix(
            (data, local_indices, local_indptr),
            shape=(stop - start, self.n_voxels),
            dtype=np.float32,
        )


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def _file_fingerprint(path: Path, block_size: int = 1024 * 1024) -> dict:
    """Return a cheap provenance fingerprint without hashing multi-GB inputs."""

    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(block_size))
        if stat.st_size > block_size:
            handle.seek(max(0, stat.st_size - block_size))
            digest.update(handle.read(block_size))
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256_first_and_last_1MiB": digest.hexdigest(),
    }


def _strip_extension(value: str) -> str:
    name = Path(str(value)).name
    # Stimulus identifiers may legitimately contain a period (for example,
    # ``2900.1`` and its matched AI image ``2900.1_1``).  Path.stem would
    # incorrectly collapse both to ``2900`` when the BIDS manifest has already
    # removed ``.jpg``.  Strip only known file extensions.
    lower = name.lower()
    for suffix in (
        ".nii.gz",
        ".nii",
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
    ):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _base_identity(image_id: str) -> str:
    """Map e.g. 2045_2 (AI) and 2045 (natural) to the family 2045."""

    return re.sub(r"_[0-9]+$", "", _strip_extension(str(image_id)))


def normalize_manifest(path: Path, strict_real_data: bool = True) -> pd.DataFrame:
    """Normalize either stimuli_600trials.csv or GLMsingle trial_manifest.tsv."""

    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=delimiter)
    # GLMsingle's wrapper writes the authoritative Type-D last-axis index.  Use
    # it explicitly rather than relying on incidental TSV row order.
    if "beta_index" in frame:
        beta_index = pd.to_numeric(frame["beta_index"], errors="raise").astype(int)
        if sorted(beta_index.tolist()) != list(range(len(frame))):
            raise ValueError(f"beta_index must be a permutation of 0..{len(frame) - 1}: {path}")
        frame = frame.assign(beta_index=beta_index).sort_values("beta_index").reset_index(drop=True)
    if "run" not in frame:
        raise ValueError(f"Manifest lacks 'run': {path}")

    if "source" in frame:
        source = frame["source"].astype(str).str.lower()
    elif "AI" in frame:
        ai = frame["AI"].astype(str).str.strip().str.lower()
        source = ai.map(
            {
                "yes": "ai",
                "y": "ai",
                "1": "ai",
                "true": "ai",
                "no": "natural",
                "n": "natural",
                "0": "natural",
                "false": "natural",
            }
        )
    elif "trial_type" in frame:
        source = np.where(
            frame["trial_type"].astype(str).str.lower().str.endswith("ai"),
            "ai",
            "natural",
        )
        source = pd.Series(source, index=frame.index)
    else:
        raise ValueError(f"Manifest lacks source/AI/trial_type: {path}")

    if "valence" in frame:
        valence = frame["valence"].astype(str).str.lower()
    elif "emotion_type" in frame:
        valence = frame["emotion_type"].astype(str).str.lower()
    elif "trial_type" in frame:
        valence = (
            frame["trial_type"]
            .astype(str)
            .str.lower()
            .str.replace(r"ai$", "", regex=True)
        )
    elif "group" in frame:
        valence = (
            frame["group"].astype(str).str.lower().str.replace(r"ai$", "", regex=True)
        )
    else:
        raise ValueError(f"Manifest lacks valence/emotion_type/trial_type: {path}")

    image_column = next(
        (name for name in ("image_id", "img", "stimulus", "stimulus_id") if name in frame),
        None,
    )
    if image_column is None:
        raise ValueError(f"Manifest lacks an image identity column: {path}")
    image_id = frame[image_column].astype(str).map(_strip_extension)

    normalized = pd.DataFrame(
        {
            "trial_index": np.arange(len(frame), dtype=np.int32),
            "run": pd.to_numeric(frame["run"], errors="raise").astype(np.int16),
            "source": source,
            "valence": valence,
            "image_id": image_id,
            "base_identity": image_id.map(_base_identity),
        }
    )
    if normalized[["source", "valence"]].isna().any().any():
        raise ValueError(f"Could not normalize all source/valence values in {path}")
    if not set(normalized["source"]).issubset({"natural", "ai"}):
        raise ValueError(f"Unexpected sources in {path}: {sorted(normalized['source'].unique())}")
    if not set(normalized["valence"]).issubset({"pleasant", "neutral", "unpleasant"}):
        raise ValueError(
            f"Unexpected valences in {path}: {sorted(normalized['valence'].unique())}"
        )

    # Beta order is run-major chronological.  We validate rather than silently sort.
    if np.any(np.diff(normalized["run"].to_numpy()) < 0):
        raise ValueError(f"Manifest is not in run-major beta order: {path}")
    if "OriginalOrder" in frame:
        observed_all = pd.to_numeric(frame["OriginalOrder"], errors="raise").to_numpy()
        global_order = np.array_equal(observed_all, np.arange(len(frame)))
        local_order = all(
            np.array_equal(
                pd.to_numeric(part["OriginalOrder"], errors="raise").to_numpy(),
                np.arange(len(part)),
            )
            for _, part in frame.groupby("run", sort=False)
        )
        if not (global_order or local_order):
            raise ValueError(f"Manifest is not in global or within-run OriginalOrder: {path}")

    if strict_real_data:
        if len(normalized) != 600:
            raise ValueError(f"Expected 600 trials, found {len(normalized)} in {path}")
        run_counts = normalized.groupby("run").size()
        if list(run_counts.index) != list(range(1, 11)) or not np.all(run_counts == 60):
            raise ValueError(f"Expected runs 1-10 with 60 trials each in {path}: {run_counts}")
        condition_counts = normalized.groupby(["source", "valence"]).size()
        if len(condition_counts) != 6 or not np.all(condition_counts == 100):
            raise ValueError(f"Expected 100 trials in each source x valence cell: {condition_counts}")
        identity_counts = normalized.groupby("image_id").size()
        if len(identity_counts) != 120 or not np.all(identity_counts == 5):
            raise ValueError("Expected 120 source-specific images repeated exactly five times")
        family_sources = normalized.drop_duplicates(["base_identity", "source"]).groupby(
            "base_identity"
        )["source"].nunique()
        if len(family_sources) != 60 or not np.all(family_sources == 2):
            raise ValueError("Expected 60 matched natural/AI base-identity families")
    return normalized


def assert_manifests_aligned(left: pd.DataFrame, right: pd.DataFrame) -> None:
    columns = ["run", "source", "valence", "image_id", "base_identity"]
    if len(left) != len(right):
        raise ValueError(f"Manifest lengths differ: {len(left)} versus {len(right)}")
    mismatch = np.any(left[columns].to_numpy() != right[columns].to_numpy(), axis=1)
    if np.any(mismatch):
        examples = np.flatnonzero(mismatch)[:10]
        detail = pd.concat(
            {
                "legacy": left.loc[examples, columns],
                "glmsingle": right.loc[examples, columns],
            },
            axis=1,
        )
        raise ValueError(f"Legacy and GLMsingle trial orders differ:\n{detail}")


def _validate_reference(reference: nib.Nifti1Image) -> None:
    if len(reference.shape) != 3:
        raise ValueError(f"Reference mask must be 3-D, got {reference.shape}")
    if not np.all(np.isfinite(reference.affine)) or abs(np.linalg.det(reference.affine[:3, :3])) < 1e-8:
        raise ValueError("Reference affine is invalid or singular")


def _validate_finite_by_trial(data: np.ndarray, block: int = 32) -> None:
    for start in range(0, data.shape[0], block):
        stop = min(data.shape[0], start + block)
        if not np.all(np.isfinite(data[start:stop])):
            bad = np.argwhere(~np.isfinite(data[start:stop]))[0]
            raise ValueError(
                f"Nonfinite beta inside the analysis mask at trial {start + int(bad[0])}, "
                f"masked voxel {int(bad[1])}. Fix the mask/input rather than selecting voxels "
                "using held-out data."
            )


def load_legacy_betas(spec: BranchSpec, n_trials: int, trial_block: int) -> BetaDataset:
    reference = nib.load(str(spec.mask_path))
    _validate_reference(reference)
    mask = np.asarray(reference.dataobj) > 0
    flat_indices = np.flatnonzero(mask.ravel()).astype(np.int64)
    raw = np.load(spec.beta_path, mmap_mode="r")
    if raw.ndim != 4 or raw.shape[0] != n_trials or tuple(raw.shape[1:]) != tuple(mask.shape):
        raise ValueError(
            f"Legacy beta shape {raw.shape} is incompatible with {n_trials} trials and "
            f"mask shape {mask.shape}"
        )
    flat_raw = raw.reshape(raw.shape[0], -1)
    data = np.empty((n_trials, len(flat_indices)), dtype=np.float32)
    for start in range(0, n_trials, trial_block):
        stop = min(n_trials, start + trial_block)
        data[start:stop] = flat_raw[start:stop, flat_indices]
    _validate_finite_by_trial(data)
    return BetaDataset(data, mask, flat_indices, reference)


def load_glmsingle_betas(spec: BranchSpec, n_trials: int, trial_block: int) -> BetaDataset:
    reference = nib.load(str(spec.mask_path))
    _validate_reference(reference)
    mask = np.asarray(reference.dataobj) > 0
    inferred = np.flatnonzero(mask.ravel()).astype(np.int64)
    if spec.flat_indices_path is not None:
        flat_indices = np.asarray(np.load(spec.flat_indices_path), dtype=np.int64).ravel()
        if not np.array_equal(flat_indices, inferred):
            raise ValueError(
                "flat_mask_indices.npy does not exactly match analysis_mask.nii.gz; "
                "the HDF5-to-NIfTI mapping is ambiguous"
            )
    else:
        flat_indices = inferred

    with h5py.File(spec.beta_path, "r") as handle:
        if "betasmd" not in handle:
            raise ValueError(f"GLMsingle HDF5 lacks 'betasmd': {spec.beta_path}")
        dataset = handle["betasmd"]
        if dataset.shape[-1] != n_trials or int(np.prod(dataset.shape[:-1])) != len(flat_indices):
            raise ValueError(
                f"GLMsingle betasmd shape {dataset.shape} is incompatible with "
                f"{len(flat_indices)} masked voxels x {n_trials} trials"
            )
        # HDF5 is voxel(s) x trial.  GLMsingle writes this dataset contiguously
        # in voxel-major order, so selecting successive trial blocks forces HDF5
        # to reread nearly the entire file for every block.  Read contiguous
        # slabs along the first voxel axis instead, then transpose each modest
        # slab into the contiguous trial x voxel destination.  This preserves
        # C-order flattening of all leading voxel dimensions while requiring
        # only one sequential pass over the HDF5 dataset.
        data = np.empty((n_trials, len(flat_indices)), dtype=np.float32)
        voxels_per_first_axis = int(np.prod(dataset.shape[1:-1]))
        # Bound each temporary HDF5 slab independently of the legacy NPY
        # trial-block option.  At 600 trials of float32, this is about 14,000
        # masked voxels per read (roughly 32 MiB).
        bytes_per_first_axis = (
            voxels_per_first_axis * n_trials * max(dataset.dtype.itemsize, 4)
        )
        first_axis_block = max(1, HDF5_TARGET_READ_BYTES // bytes_per_first_axis)
        destination_start = 0
        trailing = (slice(None),) * (dataset.ndim - 2)
        for first_start in range(0, dataset.shape[0], first_axis_block):
            first_stop = min(dataset.shape[0], first_start + first_axis_block)
            selection = (slice(first_start, first_stop),) + trailing + (slice(None),)
            block = np.asarray(dataset[selection], dtype=np.float32)
            block_voxels = int(np.prod(block.shape[:-1]))
            destination_stop = destination_start + block_voxels
            data[:, destination_start:destination_stop] = block.reshape(
                block_voxels, n_trials
            ).T
            destination_start = destination_stop
        if destination_start != len(flat_indices):
            raise RuntimeError(
                f"Loaded {destination_start} GLMsingle voxels but expected {len(flat_indices)}"
            )
    _validate_finite_by_trial(data)
    return BetaDataset(data, mask, flat_indices, reference)


def mask_normalized_smoothing(
    data: np.ndarray,
    mask: np.ndarray,
    flat_indices: np.ndarray,
    affine: np.ndarray,
    fwhm_mm: float,
    truncate: float = 4.0,
) -> np.ndarray:
    """Smooth each beta independently while correcting for mask-edge dilution."""

    if fwhm_mm < 0:
        raise ValueError("Smoothing FWHM cannot be negative")
    if fwhm_mm == 0:
        return data
    zooms = nib.affines.voxel_sizes(affine)
    sigma_mm = fwhm_mm / math.sqrt(8.0 * math.log(2.0))
    sigma_vox = sigma_mm / zooms
    denominator = gaussian_filter(
        mask.astype(np.float32), sigma=sigma_vox, mode="constant", cval=0.0, truncate=truncate
    )
    denom_masked = denominator.ravel()[flat_indices]
    if np.any(denom_masked <= np.finfo(np.float32).eps):
        raise RuntimeError("Mask-normalized smoothing denominator is zero inside the mask")

    output = np.empty_like(data, dtype=np.float32)
    volume = np.zeros(mask.shape, dtype=np.float32)
    for trial in range(data.shape[0]):
        volume.fill(0.0)
        volume.ravel()[flat_indices] = data[trial]
        numerator = gaussian_filter(
            volume, sigma=sigma_vox, mode="constant", cval=0.0, truncate=truncate
        )
        output[trial] = numerator.ravel()[flat_indices] / denom_masked
    return output


def build_neighborhoods(
    flat_indices: np.ndarray,
    shape: Sequence[int],
    affine: np.ndarray,
    radius_mm: float,
    query_block: int,
    workers: int,
    max_centers: int | None,
    center_step: int,
) -> Neighborhoods:
    if radius_mm <= 0:
        raise ValueError("Searchlight radius must be positive")
    if center_step < 1:
        raise ValueError("center_step must be >= 1")
    ijk = np.column_stack(np.unravel_index(flat_indices, tuple(shape))).astype(np.int16)
    xyz = nib.affines.apply_affine(affine, ijk).astype(np.float32)
    tree = cKDTree(xyz)
    center_columns = np.arange(len(flat_indices), dtype=np.int32)[::center_step]
    if max_centers is not None and len(center_columns) > max_centers:
        positions = np.linspace(0, len(center_columns) - 1, max_centers, dtype=np.int64)
        center_columns = center_columns[positions]
    center_flat = flat_indices[center_columns]

    indptr = np.zeros(len(center_columns) + 1, dtype=np.int64)
    index_chunks: list[np.ndarray] = []
    cursor = 0
    for start in range(0, len(center_columns), query_block):
        stop = min(len(center_columns), start + query_block)
        neighborhoods = tree.query_ball_point(
            xyz[center_columns[start:stop]],
            r=radius_mm + 1e-6,
            workers=workers,
            return_sorted=True,
        )
        lengths = np.fromiter((len(item) for item in neighborhoods), dtype=np.int64)
        local = np.empty(int(lengths.sum()), dtype=np.int32)
        offset = 0
        for item in neighborhoods:
            n_item = len(item)
            local[offset : offset + n_item] = item
            offset += n_item
        index_chunks.append(local)
        indptr[start + 1 : stop + 1] = cursor + np.cumsum(lengths)
        cursor += len(local)
    indices = np.concatenate(index_chunks) if index_chunks else np.empty(0, dtype=np.int32)
    return Neighborhoods(center_columns, center_flat, indptr, indices, len(flat_indices))


def _condition_mask(frame: pd.DataFrame, source: str, contrast: Contrast) -> np.ndarray:
    return (
        (frame["source"].to_numpy() == source)
        & frame["valence"].isin([contrast.negative_valence, contrast.positive_valence]).to_numpy()
    )


def _labels(frame: pd.DataFrame, index: np.ndarray, contrast: Contrast) -> np.ndarray:
    return (frame.iloc[index]["valence"].to_numpy() == contrast.positive_valence).astype(np.int8)


def _check_fold(fold: Fold, contrast: Contrast) -> None:
    if np.intersect1d(fold.train_index, fold.test_index).size:
        raise RuntimeError(f"Train/test trial overlap in {contrast.name}, {fold.fold_id}")
    for name, labels in (("train", fold.y_train), ("test", fold.y_test)):
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            raise RuntimeError(
                f"Missing class in {name} for {contrast.name}, {fold.fold_id}: {counts}"
            )


def validate_excluded_runs(
    frame: pd.DataFrame,
    excluded_runs: Sequence[int] = (),
    *,
    require_two_remaining: bool = False,
) -> tuple[int, ...]:
    """Validate and canonicalize acquisition runs omitted from all folds.

    The manifest and beta array deliberately remain in their authoritative
    full-trial order.  Exclusion is implemented only when fold indices are
    constructed, so no beta-axis remapping or large subset copy is required.
    """

    requested = tuple(int(value) for value in excluded_runs)
    if len(requested) != len(set(requested)):
        raise ValueError(f"Duplicate excluded runs are not allowed: {requested}")
    canonical = tuple(sorted(requested))
    observed = tuple(sorted(int(value) for value in frame["run"].unique()))
    missing = sorted(set(canonical) - set(observed))
    if missing:
        raise ValueError(
            f"Excluded runs are not present in the manifest: {missing}; observed runs: {list(observed)}"
        )
    remaining = sorted(set(observed) - set(canonical))
    if not remaining:
        raise ValueError("Run exclusion removed every acquisition run")
    if require_two_remaining and len(remaining) < 2:
        raise ValueError(
            "Leave-one-run-out decoding requires at least two acquisition runs after exclusion"
        )
    return canonical


def leave_one_run_out_folds(
    frame: pd.DataFrame,
    contrast: Contrast,
    excluded_runs: Sequence[int] = (),
) -> list[Fold]:
    train_condition = _condition_mask(frame, contrast.train_source, contrast)
    test_condition = _condition_mask(frame, contrast.test_source, contrast)
    excluded = validate_excluded_runs(
        frame, excluded_runs, require_two_remaining=True
    )
    runs = sorted(set(int(value) for value in frame["run"].unique()) - set(excluded))
    output: list[Fold] = []
    run_values = frame["run"].to_numpy()
    eligible = ~np.isin(run_values, excluded)
    for run in runs:
        train_index = np.flatnonzero(train_condition & eligible & (run_values != run))
        test_index = np.flatnonzero(test_condition & eligible & (run_values == run))
        fold = Fold(
            fold_id=f"run-{run:02d}",
            held_out=str(run),
            train_index=train_index,
            test_index=test_index,
            y_train=_labels(frame, train_index, contrast),
            y_test=_labels(frame, test_index, contrast),
        )
        _check_fold(fold, contrast)
        output.append(fold)
    return output


def identity_group_folds(
    frame: pd.DataFrame,
    contrast: Contrast,
    n_splits: int,
    seed: int,
    excluded_runs: Sequence[int] = (),
) -> list[Fold]:
    """Hold out matched natural/AI base identities as indivisible groups."""

    if n_splits < 2:
        raise ValueError("Identity GroupKFold requires at least two splits")
    excluded = validate_excluded_runs(frame, excluded_runs)
    eligible = ~frame["run"].isin(excluded).to_numpy()
    test_candidates = np.flatnonzero(
        _condition_mask(frame, contrast.test_source, contrast) & eligible
    )
    candidate_labels = _labels(frame, test_candidates, contrast)
    candidate_groups = frame.iloc[test_candidates]["base_identity"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_groups = frame["base_identity"].astype(str).to_numpy()
    train_condition = _condition_mask(frame, contrast.train_source, contrast)
    test_condition = _condition_mask(frame, contrast.test_source, contrast)
    output = []
    for fold_number, (_, local_test) in enumerate(
        splitter.split(np.zeros(len(test_candidates)), candidate_labels, candidate_groups), start=1
    ):
        held_groups = np.unique(candidate_groups[local_test])
        held = np.isin(all_groups, held_groups)
        train_index = np.flatnonzero(train_condition & eligible & ~held)
        test_index = np.flatnonzero(test_condition & eligible & held)
        fold = Fold(
            fold_id=f"identity-{fold_number:02d}",
            held_out=";".join(held_groups.tolist()),
            train_index=train_index,
            test_index=test_index,
            y_train=_labels(frame, train_index, contrast),
            y_test=_labels(frame, test_index, contrast),
        )
        _check_fold(fold, contrast)
        train_groups = set(all_groups[train_index])
        test_groups = set(all_groups[test_index])
        if train_groups.intersection(test_groups):
            raise RuntimeError("Identity family leaked between train and test")
        output.append(fold)
    return output


def fit_feature_model(
    data: np.ndarray,
    train_index: np.ndarray,
    y_train: np.ndarray,
    classifier: str,
    variance_shrinkage: float,
    scale_epsilon: float,
) -> FeatureModel:
    """Fit z-scored diagonal LDA and algebraically express it in raw units.

    The scaler is fit on the training rows only.  We compute standardized class
    statistics analytically, then fold the scaler into raw-space coefficients.
    This is exactly equivalent to transforming train/test matrices explicitly,
    but avoids materializing another multi-hundred-megabyte beta matrix.
    """

    if not 0.0 <= variance_shrinkage <= 1.0:
        raise ValueError("variance_shrinkage must be in [0, 1]")
    class_index = [train_index[y_train == label] for label in (0, 1)]
    counts = [len(index) for index in class_index]
    if min(counts) < 2:
        raise ValueError(f"At least two training observations per class are required: {counts}")

    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    for index in class_index:
        samples = np.take(data, index, axis=0)
        means.append(samples.mean(axis=0, dtype=np.float64).astype(np.float32))
        variances.append(samples.var(axis=0, ddof=1, dtype=np.float64).astype(np.float32))
        del samples

    n0, n1 = counts
    mean0, mean1 = means
    var0, var1 = variances
    train_mean = ((n0 * mean0 + n1 * mean1) / (n0 + n1)).astype(np.float32)
    total_m2 = (
        (n0 - 1) * var0
        + (n1 - 1) * var1
        + n0 * np.square(mean0 - train_mean)
        + n1 * np.square(mean1 - train_mean)
    )
    train_variance = total_m2 / (n0 + n1 - 1)
    valid = (
        np.isfinite(train_variance)
        & np.isfinite(mean0)
        & np.isfinite(mean1)
        & (train_variance > scale_epsilon**2)
    )
    scale = np.ones(data.shape[1], dtype=np.float32)
    scale[valid] = np.sqrt(train_variance[valid]).astype(np.float32)
    zmean0 = np.zeros_like(mean0)
    zmean1 = np.zeros_like(mean1)
    zmean0[valid] = (mean0[valid] - train_mean[valid]) / scale[valid]
    zmean1[valid] = (mean1[valid] - train_mean[valid]) / scale[valid]

    if classifier == "diaglda":
        pooled_raw = ((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2)
        pooled_z = np.ones_like(pooled_raw)
        pooled_z[valid] = pooled_raw[valid] / train_variance[valid]
        denominator = (1.0 - variance_shrinkage) * pooled_z + variance_shrinkage
        valid &= np.isfinite(denominator) & (denominator > scale_epsilon**2)
    elif classifier == "nearest-centroid":
        denominator = np.ones(data.shape[1], dtype=np.float32)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    weight_z = np.zeros(data.shape[1], dtype=np.float32)
    intercept_z = np.zeros(data.shape[1], dtype=np.float32)
    weight_z[valid] = (zmean1[valid] - zmean0[valid]) / denominator[valid]
    intercept_z[valid] = -0.5 * (
        np.square(zmean1[valid]) - np.square(zmean0[valid])
    ) / denominator[valid]
    weight_raw = np.zeros_like(weight_z)
    intercept_raw = np.zeros_like(intercept_z)
    weight_raw[valid] = weight_z[valid] / scale[valid]
    intercept_raw[valid] = intercept_z[valid] - train_mean[valid] * weight_raw[valid]
    return FeatureModel(weight_raw, intercept_raw, valid, n0, n1)


def _spatial_statistics(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "n_centers_valid": 0,
            "spatial_mean_accuracy": np.nan,
            "spatial_median_accuracy": np.nan,
            "spatial_std_accuracy": np.nan,
            "spatial_q05_accuracy": np.nan,
            "spatial_q95_accuracy": np.nan,
            "fraction_centers_above_chance": np.nan,
        }
    return {
        "n_centers_valid": int(len(finite)),
        "spatial_mean_accuracy": float(np.mean(finite)),
        "spatial_median_accuracy": float(np.median(finite)),
        "spatial_std_accuracy": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "spatial_q05_accuracy": float(np.quantile(finite, 0.05)),
        "spatial_q95_accuracy": float(np.quantile(finite, 0.95)),
        "fraction_centers_above_chance": float(np.mean(finite > CHANCE)),
    }


def evaluate_fold(
    data: np.ndarray,
    fold: Fold,
    neighborhoods: Neighborhoods,
    classifier: str,
    variance_shrinkage: float,
    scale_epsilon: float,
    min_voxels: int,
    center_block: int,
) -> tuple[np.ndarray, float, int]:
    model = fit_feature_model(
        data,
        fold.train_index,
        fold.y_train,
        classifier,
        variance_shrinkage,
        scale_epsilon,
    )
    test = np.take(data, fold.test_index, axis=0)
    # This is (z_test * weight_z + intercept_z), evaluated in raw units.
    evidence = test * model.weight_raw[None, :]
    evidence += model.intercept_raw[None, :]
    del test

    whole_scores = evidence[:, model.valid_voxels].sum(axis=1, dtype=np.float64)
    whole_predictions = (whole_scores > 0).astype(np.int8)
    whole_accuracy = float(np.mean(whole_predictions == fold.y_test))

    accuracy = np.full(neighborhoods.n_centers, np.nan, dtype=np.float32)
    valid_float = model.valid_voxels.astype(np.float32)
    for start in range(0, neighborhoods.n_centers, center_block):
        stop = min(neighborhoods.n_centers, start + center_block)
        adjacency = neighborhoods.sparse_block(start, stop)
        valid_counts = np.asarray(adjacency.dot(valid_float)).ravel()
        scores = np.asarray(adjacency.dot(evidence.T), dtype=np.float32)
        predictions = scores > 0
        block_accuracy = np.mean(predictions == fold.y_test[None, :], axis=1)
        block_accuracy[valid_counts < min_voxels] = np.nan
        accuracy[start:stop] = block_accuracy.astype(np.float32)
    return accuracy, whole_accuracy, int(model.valid_voxels.sum())


def _map_peak(values: np.ndarray, neighborhoods: Neighborhoods, affine: np.ndarray) -> dict:
    finite = np.isfinite(values)
    if not np.any(finite):
        return {
            "peak_accuracy": np.nan,
            "peak_i": np.nan,
            "peak_j": np.nan,
            "peak_k": np.nan,
            "peak_x_mm": np.nan,
            "peak_y_mm": np.nan,
            "peak_z_mm": np.nan,
        }
    center = int(np.nanargmax(values))
    # center_flat_indices index the full NIfTI volume.  The caller adds ijk.
    return {"peak_accuracy": float(values[center]), "peak_center_index": center}


def save_accuracy_map(
    center_values: np.ndarray,
    neighborhoods: Neighborhoods,
    reference: nib.Nifti1Image,
    path: Path,
) -> dict:
    volume = np.full(reference.shape, np.nan, dtype=np.float32)
    volume.ravel()[neighborhoods.center_flat_indices] = center_values.astype(np.float32)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header["cal_min"] = 0.0
    header["cal_max"] = 1.0
    image = nib.Nifti1Image(volume, reference.affine, header)
    nib.save(image, str(path))

    peak = _map_peak(center_values, neighborhoods, reference.affine)
    if "peak_center_index" in peak:
        center = peak.pop("peak_center_index")
        flat = int(neighborhoods.center_flat_indices[center])
        ijk = np.asarray(np.unravel_index(flat, reference.shape), dtype=int)
        xyz = nib.affines.apply_affine(reference.affine, ijk)
        peak.update(
            {
                "peak_i": int(ijk[0]),
                "peak_j": int(ijk[1]),
                "peak_k": int(ijk[2]),
                "peak_x_mm": float(xyz[0]),
                "peak_y_mm": float(xyz[1]),
                "peak_z_mm": float(xyz[2]),
            }
        )
    return peak


def _smoothing_slug(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"additional_smoothing-{text}mm"


def decode_contrast(
    data: np.ndarray,
    frame: pd.DataFrame,
    contrast: Contrast,
    cv_scheme: str,
    neighborhoods: Neighborhoods,
    reference: nib.Nifti1Image,
    output_dir: Path,
    common: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    if cv_scheme == "leave-one-run-out":
        folds = leave_one_run_out_folds(frame, contrast, args.exclude_runs)
    elif cv_scheme == "identity-groupkfold":
        folds = identity_group_folds(
            frame,
            contrast,
            args.identity_splits,
            args.random_seed,
            args.exclude_runs,
        )
    else:
        raise ValueError(f"Unknown CV scheme: {cv_scheme}")

    correct_sum = np.zeros(neighborhoods.n_centers, dtype=np.float64)
    observation_sum = np.zeros(neighborhoods.n_centers, dtype=np.int32)
    fold_rows: list[dict] = []
    whole_correct = 0.0
    whole_observations = 0
    whole_fold_accuracies: list[float] = []
    started = time.monotonic()
    for number, fold in enumerate(folds, start=1):
        fold_accuracy, whole_accuracy, n_valid_features = evaluate_fold(
            data,
            fold,
            neighborhoods,
            args.classifier,
            args.variance_shrinkage,
            args.scale_epsilon,
            args.min_voxels,
            args.center_block_size,
        )
        finite = np.isfinite(fold_accuracy)
        correct_sum[finite] += fold_accuracy[finite] * len(fold.test_index)
        observation_sum[finite] += len(fold.test_index)
        whole_correct += whole_accuracy * len(fold.test_index)
        whole_observations += len(fold.test_index)
        whole_fold_accuracies.append(whole_accuracy)
        train_counts = np.bincount(fold.y_train, minlength=2)
        test_counts = np.bincount(fold.y_test, minlength=2)
        row = {
            **common,
            "contrast": contrast.name,
            "contrast_kind": contrast.kind,
            "train_source": contrast.train_source,
            "test_source": contrast.test_source,
            "positive_valence": contrast.positive_valence,
            "negative_valence": contrast.negative_valence,
            "cv_scheme": cv_scheme,
            "fold_id": fold.fold_id,
            "held_out": fold.held_out,
            "n_train": int(len(fold.train_index)),
            "n_test": int(len(fold.test_index)),
            "n_train_class_0": int(train_counts[0]),
            "n_train_class_1": int(train_counts[1]),
            "n_test_class_0": int(test_counts[0]),
            "n_test_class_1": int(test_counts[1]),
            "n_training_valid_voxels": n_valid_features,
            "whole_mask_accuracy": whole_accuracy,
            **_spatial_statistics(fold_accuracy),
        }
        fold_rows.append(row)
        print(
            f"    {contrast.name} {cv_scheme} {number}/{len(folds)}: "
            f"whole={whole_accuracy:.3f}, spatial_mean={row['spatial_mean_accuracy']:.3f}",
            flush=True,
        )

    aggregate = np.full(neighborhoods.n_centers, np.nan, dtype=np.float32)
    valid = observation_sum > 0
    aggregate[valid] = (correct_sum[valid] / observation_sum[valid]).astype(np.float32)
    map_path = output_dir / f"{contrast.name}_accuracy.nii.gz"
    peak = save_accuracy_map(aggregate, neighborhoods, reference, map_path)
    map_row = {
        **common,
        "contrast": contrast.name,
        "contrast_kind": contrast.kind,
        "train_source": contrast.train_source,
        "test_source": contrast.test_source,
        "positive_valence": contrast.positive_valence,
        "negative_valence": contrast.negative_valence,
        "cv_scheme": cv_scheme,
        "n_folds": int(len(folds)),
        "mean_fold_whole_mask_accuracy": float(np.mean(whole_fold_accuracies)),
        "pooled_whole_mask_accuracy": float(whole_correct / whole_observations),
        "minimum_test_observations_per_center": int(observation_sum[valid].min()) if np.any(valid) else 0,
        "maximum_test_observations_per_center": int(observation_sum[valid].max()) if np.any(valid) else 0,
        "elapsed_seconds": float(time.monotonic() - started),
        "map_path": str(map_path.resolve()),
        **_spatial_statistics(aggregate),
        **peak,
    }
    return fold_rows, map_row


def decode_variant(
    dataset: BetaDataset,
    frame: pd.DataFrame,
    branch: BranchSpec,
    smoothing_mm: float,
    output_root: Path,
    contrasts: Sequence[Contrast],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    variant_dir = output_root / branch.name / _smoothing_slug(smoothing_mm)
    variant_dir.mkdir(parents=True, exist_ok=False)
    if smoothing_mm == 0:
        data = dataset.data
    else:
        print(f"  Mask-normalized beta smoothing: {smoothing_mm:g} mm FWHM", flush=True)
        data = mask_normalized_smoothing(
            dataset.data,
            dataset.mask,
            dataset.flat_indices,
            dataset.reference.affine,
            smoothing_mm,
            args.smoothing_truncate,
        )

    print(
        f"  Building {args.radius_mm:g}-mm neighborhoods for {len(dataset.flat_indices):,} voxels",
        flush=True,
    )
    neighborhoods = build_neighborhoods(
        dataset.flat_indices,
        dataset.mask.shape,
        dataset.reference.affine,
        args.radius_mm,
        args.neighbor_query_block_size,
        args.n_jobs,
        args.max_centers,
        args.center_step,
    )
    sizes = np.diff(neighborhoods.indptr)
    common = {
        "subject": args.subject,
        "branch": branch.name,
        "beta_kind": branch.kind,
        "input_smoothing_note": branch.input_smoothing_note,
        "additional_smoothing_fwhm_mm": float(smoothing_mm),
        "classifier": args.classifier,
        "variance_shrinkage_to_unit": float(args.variance_shrinkage),
        "searchlight_radius_mm": float(args.radius_mm),
        "chance_accuracy": CHANCE,
        "n_input_voxels": int(data.shape[1]),
        "n_searchlight_centers": int(neighborhoods.n_centers),
        "median_searchlight_voxels": float(np.median(sizes)),
        "min_required_training_valid_voxels": int(args.min_voxels),
        "analysis_role": (
            "diagnostic_run_exclusion_sensitivity"
            if args.exclude_runs
            else "primary_all_runs"
        ),
        "excluded_runs": ";".join(str(value) for value in args.exclude_runs),
        "n_analysis_trials": int(
            (~frame["run"].isin(args.exclude_runs)).sum()
        ),
    }

    all_folds: list[dict] = []
    all_maps: list[dict] = []
    for cv_scheme in args.cv:
        cv_dir = variant_dir / cv_scheme
        cv_dir.mkdir()
        for contrast in contrasts:
            fold_rows, map_row = decode_contrast(
                data,
                frame,
                contrast,
                cv_scheme,
                neighborhoods,
                dataset.reference,
                cv_dir,
                common,
                args,
            )
            all_folds.extend(fold_rows)
            all_maps.append(map_row)
        pd.DataFrame([row for row in all_folds if row["cv_scheme"] == cv_scheme]).to_csv(
            cv_dir / "fold_summary.csv", index=False
        )
        pd.DataFrame([row for row in all_maps if row["cv_scheme"] == cv_scheme]).to_csv(
            cv_dir / "map_summary.csv", index=False
        )
    if data is not dataset.data:
        del data
        gc.collect()
    return all_folds, all_maps


def _resolve_glmsingle(root: Path) -> tuple[Path, Path, Path | None, Path]:
    h5_candidates = [root / TYPE_D_NAME, root / "glmsingle" / TYPE_D_NAME]
    h5_path = next((path for path in h5_candidates if path.exists()), None)
    if h5_path is None:
        raise FileNotFoundError(f"Could not find {TYPE_D_NAME} under {root}")
    run_root = h5_path.parent.parent if h5_path.parent.name == "glmsingle" else h5_path.parent
    mask_candidates = [run_root / "analysis_mask.nii.gz", run_root / "analysis_mask.nii"]
    mask_path = next((path for path in mask_candidates if path.exists()), None)
    if mask_path is None:
        raise FileNotFoundError(f"Could not find analysis_mask.nii[.gz] under {run_root}")
    indices_path = run_root / "flat_mask_indices.npy"
    manifest_candidates = [run_root / "trial_manifest.tsv", run_root / "trial_manifest.csv"]
    manifest_path = next((path for path in manifest_candidates if path.exists()), None)
    if manifest_path is None:
        raise FileNotFoundError(f"Could not find trial_manifest.tsv/csv under {run_root}")
    return h5_path, mask_path, indices_path if indices_path.exists() else None, manifest_path


def make_branch_specs(args: argparse.Namespace) -> list[BranchSpec]:
    specs: list[BranchSpec] = []
    if args.only in {"both", "legacy"}:
        legacy_beta = args.legacy_beta or DEFAULT_LEGACY_ROOT / f"Sub{args.subject}_beta.npy"
        specs.append(
            BranchSpec(
                name="legacy_lsa_8mm",
                kind="legacy_npy",
                beta_path=legacy_beta,
                mask_path=args.legacy_mask,
                manifest_path=args.legacy_manifest,
                smoothing_fwhm_mm=tuple(args.legacy_smoothing_mm),
                input_smoothing_note="SPM LS-A betas estimated from images already smoothed 8 mm FWHM",
            )
        )
    if args.only in {"both", "glmsingle"}:
        if args.glmsingle_dir is None:
            raise ValueError("--glmsingle-dir is required unless --only legacy")
        h5_path, mask_path, indices_path, manifest_path = _resolve_glmsingle(args.glmsingle_dir)
        specs.append(
            BranchSpec(
                name="glmsingle_typed",
                kind="glmsingle_type_d_hdf5",
                beta_path=h5_path,
                mask_path=mask_path,
                manifest_path=manifest_path,
                flat_indices_path=indices_path,
                smoothing_fwhm_mm=tuple(args.new_smoothing_mm),
                input_smoothing_note="No spatial smoothing before beta estimation",
            )
        )
    for spec in specs:
        for path in (spec.beta_path, spec.mask_path, spec.manifest_path):
            if not path.exists():
                raise FileNotFoundError(path)
        if spec.flat_indices_path is not None and not spec.flat_indices_path.exists():
            raise FileNotFoundError(spec.flat_indices_path)
        if len(spec.smoothing_fwhm_mm) == 0 or any(value < 0 for value in spec.smoothing_fwhm_mm):
            raise ValueError(f"Invalid smoothing list for {spec.name}: {spec.smoothing_fwhm_mm}")
        if len(set(spec.smoothing_fwhm_mm)) != len(spec.smoothing_fwhm_mm):
            raise ValueError(f"Duplicate smoothing values for {spec.name}")
    return specs


def side_by_side_table(map_rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(map_rows)
    if frame.empty or not {"legacy_lsa_8mm", "glmsingle_typed"}.issubset(set(frame["branch"])):
        return pd.DataFrame()
    legacy = frame[frame["branch"] == "legacy_lsa_8mm"].copy()
    # If requested, multiple legacy extra-smoothing variants remain explicit.
    legacy = legacy.rename(
        columns={
            "additional_smoothing_fwhm_mm": "legacy_additional_smoothing_fwhm_mm",
            "spatial_mean_accuracy": "legacy_spatial_mean_accuracy",
            "spatial_median_accuracy": "legacy_spatial_median_accuracy",
            "pooled_whole_mask_accuracy": "legacy_pooled_whole_mask_accuracy",
            "map_path": "legacy_map_path",
        }
    )
    new = frame[frame["branch"] == "glmsingle_typed"].copy().rename(
        columns={
            "additional_smoothing_fwhm_mm": "new_additional_smoothing_fwhm_mm",
            "spatial_mean_accuracy": "new_spatial_mean_accuracy",
            "spatial_median_accuracy": "new_spatial_median_accuracy",
            "pooled_whole_mask_accuracy": "new_pooled_whole_mask_accuracy",
            "map_path": "new_map_path",
        }
    )
    keys = ["subject", "cv_scheme", "contrast", "classifier", "searchlight_radius_mm"]
    keep_legacy = keys + [
        "legacy_additional_smoothing_fwhm_mm",
        "legacy_spatial_mean_accuracy",
        "legacy_spatial_median_accuracy",
        "legacy_pooled_whole_mask_accuracy",
        "legacy_map_path",
    ]
    keep_new = keys + [
        "new_additional_smoothing_fwhm_mm",
        "new_spatial_mean_accuracy",
        "new_spatial_median_accuracy",
        "new_pooled_whole_mask_accuracy",
        "new_map_path",
    ]
    result = new[keep_new].merge(legacy[keep_legacy], on=keys, how="inner")
    result["delta_new_minus_legacy_spatial_mean"] = (
        result["new_spatial_mean_accuracy"] - result["legacy_spatial_mean_accuracy"]
    )
    result["delta_new_minus_legacy_spatial_median"] = (
        result["new_spatial_median_accuracy"] - result["legacy_spatial_median_accuracy"]
    )
    result["delta_new_minus_legacy_whole_mask"] = (
        result["new_pooled_whole_mask_accuracy"]
        - result["legacy_pooled_whole_mask_accuracy"]
    )
    ordered = keys + [
        "legacy_additional_smoothing_fwhm_mm",
        "new_additional_smoothing_fwhm_mm",
        "legacy_spatial_mean_accuracy",
        "new_spatial_mean_accuracy",
        "delta_new_minus_legacy_spatial_mean",
        "legacy_spatial_median_accuracy",
        "new_spatial_median_accuracy",
        "delta_new_minus_legacy_spatial_median",
        "legacy_pooled_whole_mask_accuracy",
        "new_pooled_whole_mask_accuracy",
        "delta_new_minus_legacy_whole_mask",
        "legacy_map_path",
        "new_map_path",
    ]
    return result[ordered].sort_values(
        ["cv_scheme", "contrast", "new_additional_smoothing_fwhm_mm"]
    )


def write_provenance(
    output: Path,
    args: argparse.Namespace,
    specs: Sequence[BranchSpec],
    contrasts: Sequence[Contrast],
) -> None:
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "arguments": vars(args),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "nibabel": nib.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
        },
        "algorithm": {
            "primary_cv": "leave one acquisition run out",
            "identity_sensitivity": (
                "StratifiedGroupKFold on base stimulus identity; matched natural/AI variants "
                "are withheld together"
            ),
            "scaling": "per-voxel mean and sample SD fit on training trials only",
            "classifier": args.classifier,
            "diaglda_variance_regularization": (
                "pooled within-class standardized variance shrunk toward 1"
            ),
            "class_priors": "equal",
            "searchlight": "Euclidean sphere in world coordinates; sparse exact evidence sum",
            "smoothing": "independent beta-volume Gaussian smoothing normalized by smoothed mask",
            "map_outside_centers": "NaN",
            "run_exclusion": (
                "All trials from the declared excluded acquisition runs are omitted from every "
                "training and test fold. The source beta array and manifest retain their "
                "authoritative full-trial order."
            ),
        },
        "contrasts": [asdict(item) for item in contrasts],
        "branches": [
            {
                **asdict(spec),
                "beta_fingerprint": _file_fingerprint(spec.beta_path),
                "mask_fingerprint": _file_fingerprint(spec.mask_path),
                "manifest_fingerprint": _file_fingerprint(spec.manifest_path),
            }
            for spec in specs
        ],
        "interpretation_caveats": [
            "Legacy input was already smoothed 8 mm; additional smoothing is cumulative.",
            "Legacy and GLMsingle maps may have different grids/masks. Compare tabulated scalar and "
            "spatial summaries; do not subtract map arrays without explicit resampling.",
            "Spatial mean accuracy is a descriptive average across overlapping searchlights, not an "
            "independent-observation inferential statistic.",
            "No permutation p-values or multiple-comparison correction are produced here.",
            *(
                [
                    "An excluded-run result is a secondary robustness analysis. Relative to the "
                    "all-run LORO result, it changes both the training composition of retained "
                    "folds and the set of pooled held-out observations.",
                    "Run exclusion occurs at decoding only. Upstream GLMsingle Type-D betas, HRF "
                    "selection, denoising, and ridge selection were estimated from the complete "
                    "run set and are not re-estimated by this option.",
                ]
                if args.exclude_runs
                else []
            ),
        ],
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=_json_ready) + "\n", encoding="utf-8"
    )


def run_comparison(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {args.output}. "
            "Choose a new versioned output directory."
        )
    if args.subject is None:
        raise ValueError("--subject is required")
    if args.radius_mm <= 0 or args.min_voxels < 1:
        raise ValueError("radius-mm must be positive and min-voxels must be >= 1")
    if len(set(args.cv)) != len(args.cv):
        raise ValueError(f"Duplicate CV schemes are not allowed: {args.cv}")
    if args.contrasts and len(set(args.contrasts)) != len(args.contrasts):
        raise ValueError(f"Duplicate contrasts are not allowed: {args.contrasts}")
    specs = make_branch_specs(args)
    contrasts = [CONTRAST_BY_NAME[name] for name in args.contrasts] if args.contrasts else list(CONTRASTS)

    manifests = {spec.name: normalize_manifest(spec.manifest_path) for spec in specs}
    if len(manifests) == 2:
        assert_manifests_aligned(manifests["legacy_lsa_8mm"], manifests["glmsingle_typed"])
    frame = next(iter(manifests.values()))
    args.exclude_runs = list(
        validate_excluded_runs(
            frame,
            args.exclude_runs,
            require_two_remaining="leave-one-run-out" in args.cv,
        )
    )
    included_in_decoding = ~frame["run"].isin(args.exclude_runs)

    plan = {
        "subject": args.subject,
        "branches": [spec.name for spec in specs],
        "variants": {spec.name: list(spec.smoothing_fwhm_mm) for spec in specs},
        "cv": args.cv,
        "contrasts": [item.name for item in contrasts],
        "n_trials": len(frame),
        "n_analysis_trials": int(included_in_decoding.sum()),
        "excluded_runs": args.exclude_runs,
        "output": str(args.output),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return

    args.output.mkdir(parents=True, exist_ok=False)
    write_provenance(args.output, args, specs, contrasts)
    frame.assign(included_in_decoding=included_in_decoding).to_csv(
        args.output / "trial_manifest_normalized.tsv", sep="\t", index=False
    )

    fold_rows: list[dict] = []
    map_rows: list[dict] = []
    for spec in specs:
        print(f"Loading {spec.name}: {spec.beta_path}", flush=True)
        if spec.kind == "legacy_npy":
            dataset = load_legacy_betas(spec, len(frame), args.trial_block_size)
        else:
            dataset = load_glmsingle_betas(spec, len(frame), args.trial_block_size)
        print(
            f"  Loaded {dataset.data.shape[0]} trials x {dataset.data.shape[1]:,} voxels; "
            f"grid={dataset.mask.shape}",
            flush=True,
        )
        for smoothing in spec.smoothing_fwhm_mm:
            new_folds, new_maps = decode_variant(
                dataset,
                manifests[spec.name],
                spec,
                smoothing,
                args.output,
                contrasts,
                args,
            )
            fold_rows.extend(new_folds)
            map_rows.extend(new_maps)
            pd.DataFrame(fold_rows).to_csv(args.output / "fold_summary.csv", index=False)
            pd.DataFrame(map_rows).to_csv(args.output / "map_summary.csv", index=False)
        del dataset
        gc.collect()

    comparison = side_by_side_table(map_rows)
    if not comparison.empty:
        comparison.to_csv(args.output / "side_by_side_summary.csv", index=False)
    print(f"Completed comparison: {args.output}", flush=True)


def _synthetic_manifest(n_runs: int = 5, identities_per_cell: int = 4) -> pd.DataFrame:
    rows = []
    trial = 0
    for run in range(1, n_runs + 1):
        for valence in ("pleasant", "neutral", "unpleasant"):
            for identity in range(identities_per_cell):
                base = f"{valence[:1]}{identity:02d}"
                for source in ("natural", "ai"):
                    image = base if source == "natural" else f"{base}_0"
                    rows.append(
                        {
                            "trial_index": trial,
                            "run": run,
                            "source": source,
                            "valence": valence,
                            "image_id": image,
                            "base_identity": base,
                        }
                    )
                    trial += 1
    return pd.DataFrame(rows)


def run_self_test() -> None:
    if _strip_extension("2900.1") != "2900.1":
        raise AssertionError("Decimal stimulus identity was treated as a file extension")
    if _base_identity("2900.1_1.jpg") != "2900.1":
        raise AssertionError("Matched decimal natural/AI identity parsing failed")
    rng = np.random.default_rng(20260728)
    shape = (7, 7, 7)
    mask = np.ones(shape, dtype=bool)
    flat_indices = np.flatnonzero(mask.ravel())
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    reference = nib.Nifti1Image(mask.astype(np.uint8), affine)
    frame = _synthetic_manifest()
    data = rng.normal(0, 1, (len(frame), len(flat_indices))).astype(np.float32)
    signal_volume = np.zeros(shape, dtype=bool)
    signal_volume[2:5, 2:5, 2:5] = True
    signal = signal_volume.ravel()[flat_indices]
    emotional = frame["valence"].isin(["pleasant", "unpleasant"]).to_numpy()
    data[np.ix_(emotional, signal)] += 1.25
    dataset = BetaDataset(data, mask, flat_indices, reference)
    neighborhoods = build_neighborhoods(
        flat_indices,
        shape,
        affine,
        radius_mm=5.0,
        query_block=64,
        workers=1,
        max_centers=None,
        center_step=1,
    )
    # Test both CV generators and both decomposable classifiers.
    contrast = CONTRAST_BY_NAME["train_natural_pleasant_vs_neutral_test_ai"]
    for folds in (
        leave_one_run_out_folds(frame, contrast),
        identity_group_folds(frame, contrast, n_splits=2, seed=20260728),
    ):
        for classifier in ("diaglda", "nearest-centroid"):
            accuracies = []
            for fold in folds:
                values, whole, valid = evaluate_fold(
                    dataset.data,
                    fold,
                    neighborhoods,
                    classifier=classifier,
                    variance_shrinkage=0.1,
                    scale_epsilon=1e-7,
                    min_voxels=5,
                    center_block=32,
                )
                if not np.all((values[np.isfinite(values)] >= 0) & (values[np.isfinite(values)] <= 1)):
                    raise AssertionError("Searchlight accuracy left [0, 1]")
                if not 0 <= whole <= 1 or valid <= 0:
                    raise AssertionError("Invalid whole-mask self-test result")
                accuracies.append(values)
            aggregate = np.nanmean(np.stack(accuracies), axis=0)
            center_flat = np.ravel_multi_index((3, 3, 3), shape)
            center_row = int(np.flatnonzero(neighborhoods.center_flat_indices == center_flat)[0])
            if aggregate[center_row] <= CHANCE:
                raise AssertionError(
                    f"Injected signal was not decoded for {classifier}: {aggregate[center_row]}"
                )

    smoothed = mask_normalized_smoothing(data[:2], mask, flat_indices, affine, 3.0)
    if smoothed.shape != data[:2].shape or not np.all(np.isfinite(smoothed)):
        raise AssertionError("Mask-normalized smoothing self-test failed")
    with tempfile.TemporaryDirectory(prefix="decode_compare_") as temp:
        temp_path = Path(temp)
        map_path = temp_path / "test.nii.gz"
        values = np.full(neighborhoods.n_centers, 0.5, dtype=np.float32)
        save_accuracy_map(values, neighborhoods, reference, map_path)
        loaded = nib.load(str(map_path))
        if loaded.get_data_dtype() != np.dtype(np.float32) or loaded.shape != shape:
            raise AssertionError("NIfTI output self-test failed")

        # Exercise both on-disk beta loaders and authoritative beta_index order.
        mask_path = temp_path / "analysis_mask.nii.gz"
        nib.save(reference, str(mask_path))
        indices_path = temp_path / "flat_mask_indices.npy"
        np.save(indices_path, flat_indices)
        h5_path = temp_path / TYPE_D_NAME
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset("betasmd", data=data.T[:, None, None, :])
        manifest_path = temp_path / "trial_manifest.tsv"
        disk_manifest = frame.copy()
        disk_manifest["beta_index"] = np.arange(len(disk_manifest))
        # Reverse physical TSV rows: normalize_manifest must restore beta order.
        disk_manifest.iloc[::-1].to_csv(manifest_path, sep="\t", index=False)
        normalized = normalize_manifest(manifest_path, strict_real_data=False)
        if not np.array_equal(normalized["run"].to_numpy(), frame["run"].to_numpy()):
            raise AssertionError("beta_index manifest ordering self-test failed")
        h5_spec = BranchSpec(
            "synthetic_glmsingle",
            "glmsingle_type_d_hdf5",
            h5_path,
            mask_path,
            manifest_path,
            (0.0,),
            indices_path,
        )
        loaded_h5 = load_glmsingle_betas(h5_spec, len(frame), trial_block=7)
        if not np.array_equal(loaded_h5.data, data):
            raise AssertionError("GLMsingle HDF5 beta-axis self-test failed")

        legacy_path = temp_path / "legacy.npy"
        legacy_volumes = data.reshape((len(frame),) + shape)
        np.save(legacy_path, legacy_volumes)
        legacy_spec = BranchSpec(
            "synthetic_legacy",
            "legacy_npy",
            legacy_path,
            mask_path,
            manifest_path,
            (0.0,),
        )
        loaded_legacy = load_legacy_betas(legacy_spec, len(frame), trial_block=9)
        if not np.array_equal(loaded_legacy.data, data):
            raise AssertionError("Legacy NPY beta-axis self-test failed")
    print("SELF-TEST PASSED: CV blocking, classifiers, smoothing, sparse searchlights, and NIfTI I/O")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--self-test", action="store_true", help="Run a synthetic test and exit")
    parser.add_argument("--subject", type=int, choices=[4, 5, 6])
    parser.add_argument("--output", type=Path, help="A new, non-existing output directory")
    parser.add_argument("--only", choices=["both", "legacy", "glmsingle"], default="both")
    parser.add_argument("--legacy-beta", type=Path)
    parser.add_argument("--legacy-mask", type=Path, default=DEFAULT_LEGACY_MASK)
    parser.add_argument("--legacy-manifest", type=Path, default=DEFAULT_LEGACY_MANIFEST)
    parser.add_argument(
        "--glmsingle-dir",
        type=Path,
        help="Run root containing analysis_mask, trial_manifest, and glmsingle/Type-D HDF5",
    )
    parser.add_argument("--legacy-smoothing-mm", type=float, nargs="+", default=[0.0])
    parser.add_argument("--new-smoothing-mm", type=float, nargs="+", default=[0.0, 3.0])
    parser.add_argument(
        "--cv",
        nargs="+",
        choices=["leave-one-run-out", "identity-groupkfold"],
        default=["leave-one-run-out"],
        help="Primary CV and optional identity-blocked sensitivity",
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        choices=list(CONTRAST_BY_NAME),
        help="Subset of the eight contrasts; all are run when omitted",
    )
    parser.add_argument(
        "--exclude-runs",
        type=int,
        nargs="+",
        default=[],
        metavar="RUN",
        help=(
            "Secondary sensitivity only: omit every trial from these acquisition runs from "
            "all training and test folds while retaining the full beta-axis input"
        ),
    )
    parser.add_argument("--classifier", choices=["diaglda", "nearest-centroid"], default="diaglda")
    parser.add_argument(
        "--variance-shrinkage",
        type=float,
        default=0.1,
        help="Diagonal-LDA pooled variance shrinkage toward standardized variance 1",
    )
    parser.add_argument("--scale-epsilon", type=float, default=1e-7)
    parser.add_argument("--radius-mm", type=float, default=5.0)
    parser.add_argument("--min-voxels", type=int, default=10)
    parser.add_argument("--identity-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260728)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Workers for KD-tree neighborhood queries")
    parser.add_argument("--center-block-size", type=int, default=4096)
    parser.add_argument("--neighbor-query-block-size", type=int, default=8192)
    parser.add_argument("--trial-block-size", type=int, default=16)
    parser.add_argument("--smoothing-truncate", type=float, default=4.0)
    parser.add_argument(
        "--center-step",
        type=int,
        default=1,
        help="Evaluate every Nth masked voxel (1 is required for a full map)",
    )
    parser.add_argument(
        "--max-centers",
        type=int,
        help="Evenly sample at most this many centers; diagnostic use only",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/manifests and print plan")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if args.output is None:
        parser.error("--output is required unless --self-test")
    if not 0 <= args.variance_shrinkage <= 1:
        parser.error("--variance-shrinkage must be in [0, 1]")
    for name in (
        "center_block_size",
        "neighbor_query_block_size",
        "trial_block_size",
        "identity_splits",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()
