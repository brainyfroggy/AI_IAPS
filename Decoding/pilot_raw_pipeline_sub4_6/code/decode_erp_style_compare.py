#!/usr/bin/env python3
"""Reproduce the historical AI-IAPS Avg(random) analysis on matched beta pipelines.

The user-facing name for this analysis is "ERP-style decoding".  More exactly,
the attached historical figure used the ``voxel_trial_source_zscore`` variant of
Avg(random): single-trial betas are normalized, randomly grouped and averaged
into pseudo-trials, and classified with a linear SVM.  This program applies that
exact decoder to:

1. the original SPM LS-A betas (already smoothed 8 mm FWHM);
2. GLMsingle Type-D betas without added smoothing; and
3. the same GLMsingle Type-D betas after independent 3 mm FWHM smoothing.

It also loads the exact 28-subject result used by the attached figure and checks
that recomputing Subjects 4, 5 and 6 from the original individual SPM beta
images reproduces their saved historical values.  The historical method is deliberately preserved for
like-for-like comparison, including its source-level normalization before CV.
That behavior uses held-out data distribution information and is therefore
descriptive rather than a leakage-controlled estimate of out-of-sample performance.

The output directory must not exist.  Results are first written to a sibling
staging directory and published by atomic rename only after all validation
checks pass.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from nibabel.processing import resample_from_to
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from decode_compare import (
    BranchSpec,
    TYPE_D_NAME,
    assert_manifests_aligned,
    load_glmsingle_betas,
    normalize_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PILOT_ROOT = Path(__file__).resolve().parents[1]
DECODING_ROOT = PROJECT_ROOT / "Decoding"
DEFAULT_LEGACY_ROOT = PROJECT_ROOT / "GLM_singletrial" / "betas"
DEFAULT_LEGACY_GROUPS = PROJECT_ROOT / "GLM_singletrial" / "beta_groups.csv"
DEFAULT_LEGACY_MANIFEST = DECODING_ROOT / "stimuli_600trials.csv"
DEFAULT_GLMSINGLE_ROOT = PILOT_ROOT / "glmsingle_sdc_defaultgrid"
DEFAULT_ATLAS = (
    Path(r"C:\MRIcroGL\Resources\atlas\kastner.nii.gz")
    if os.name == "nt"
    else Path("/mnt/c/MRIcroGL/Resources/atlas/kastner.nii.gz")
)
DEFAULT_ATLAS_LABELS = DEFAULT_ATLAS.with_name("kastner.nii.txt")
DEFAULT_HISTORICAL = (
    DECODING_ROOT
    / "results"
    / "decoding_multisub_avg_voxel_trial_source_zscore_aal3_all.pkl"
)
DEFAULT_HISTORICAL_NOTEBOOK = (
    DECODING_ROOT
    / "decoding_multisub_parallel_avg_voxel_trial_source_zscore_AAL3_all.ipynb"
)

ROI_ORDER = (
    "V1v",
    "V1d",
    "V2v",
    "V2d",
    "V3v",
    "V3d",
    "hV4",
    "V3a",
    "V3b",
    "IPS",
    "LO1",
    "LO2",
    "hMT",
    "VO1",
    "VO2",
    "PHC1",
    "PHC2",
)

ROI_LABEL_NAMES = {
    "V1v": ("V1v",),
    "V1d": ("V1d",),
    "V2v": ("V2v",),
    "V2d": ("V2d",),
    "V3v": ("V3v",),
    "V3d": ("V3d",),
    "hV4": ("hV4",),
    "V3a": ("V3a",),
    "V3b": ("V3b",),
    "IPS": ("IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5"),
    "LO1": ("LO1",),
    "LO2": ("LO2",),
    "hMT": ("hMT",),
    "VO1": ("VO1",),
    "VO2": ("VO2",),
    "PHC1": ("PHC1",),
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

HISTORICAL_KEYS = {
    "within_natural_pleasant_vs_neutral": "pleasant_vs_neutral",
    "within_ai_pleasant_vs_neutral": "pleasantAI_vs_neutralAI",
    "within_natural_unpleasant_vs_neutral": "unpleasant_vs_neutral",
    "within_ai_unpleasant_vs_neutral": "unpleasantAI_vs_neutralAI",
    "train_natural_pleasant_vs_neutral_test_ai": (
        "train_pleasantvneutral_test_pleasantAIvneutralAI"
    ),
    "train_ai_pleasant_vs_neutral_test_natural": (
        "train_pleasantAIvneutralAI_test_pleasantvneutral"
    ),
    "train_natural_unpleasant_vs_neutral_test_ai": (
        "train_unpleasantvneutral_test_unpleasantAIvneutralAI"
    ),
    "train_ai_unpleasant_vs_neutral_test_natural": (
        "train_unpleasantAIvneutralAI_test_unpleasantvneutral"
    ),
}

CONTRAST_LABELS = {
    "within_natural_pleasant_vs_neutral": "PL vs Nt, Natural",
    "within_ai_pleasant_vs_neutral": "PL vs Nt, AI",
    "within_natural_unpleasant_vs_neutral": "UP vs Nt, Natural",
    "within_ai_unpleasant_vs_neutral": "UP vs Nt, AI",
    "train_natural_pleasant_vs_neutral_test_ai": "PL vs Nt, Natural to AI",
    "train_ai_pleasant_vs_neutral_test_natural": "PL vs Nt, AI to Natural",
    "train_natural_unpleasant_vs_neutral_test_ai": "UP vs Nt, Natural to AI",
    "train_ai_unpleasant_vs_neutral_test_natural": "UP vs Nt, AI to Natural",
}

PIPELINE_LABELS = {
    "historical_28sub_original": "Original result (n=28)",
    "legacy_lsa_8mm": "Legacy LS-A 8 mm (n=3)",
    "glmsingle_typed_0mm": "GLMsingle Type-D 0 mm (n=3)",
    "glmsingle_typed_3mm": "GLMsingle Type-D +3 mm (n=3)",
}

PIPELINE_ORDER = (
    "historical_28sub_original",
    "legacy_lsa_8mm",
    "glmsingle_typed_0mm",
    "glmsingle_typed_3mm",
)

MODE = "voxel_trial_source_zscore"
N_REPEATS = 20
N_FOLDS = 4
N_AVG_GROUPS = 3
SEED = 42
# One accuracy quantum (1/160) is allowed for rare linear-SVC boundary ties
# across the historical Windows and current Linux/scikit-learn environments.
REPRODUCTION_TOLERANCE = 1.0 / 160.0 + 1e-12


@dataclass(frozen=True)
class GridRois:
    union_flat_indices: np.ndarray
    roi_union_positions: Mapping[str, np.ndarray]
    count_rows: list[dict]


_RESAMPLED_ROI_CACHE: dict[tuple, np.ndarray] = {}


def sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path, full_hash_limit: int = 64 * 1024 * 1024) -> dict:
    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if stat.st_size <= full_hash_limit:
        result["sha256"] = sha256(path)
    else:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
            handle.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
        result["sha256_first_and_last_1MiB"] = digest.hexdigest()
    return result


def sem(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return float("nan")
    return float(np.std(array, ddof=1) / math.sqrt(len(array)))


def safe_pattern_zscore(array: np.ndarray) -> np.ndarray:
    """Historical pattern normalization: z-score voxels within each trial."""

    normalized = zscore(array, axis=1)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def fit_trial_zscore(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    std[std == 0] = 1
    return mean, std


def apply_trial_zscore(
    array: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return (array - mean) / std


def compute_voxel_source_zscore(
    condition_data: Mapping[tuple[str, str], np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    """Replicate the historical pre-CV source normalization exactly.

    This deliberately uses all trials from one source, including observations
    later assigned to a test fold.  It is retained only to match the attached
    analysis and is explicitly flagged in provenance and the report.
    """

    output: dict[tuple[str, str], np.ndarray] = {}
    for source in ("natural", "ai"):
        keys = [(source, value) for value in ("pleasant", "neutral", "unpleasant")]
        sizes = [len(condition_data[key]) for key in keys]
        combined = np.vstack([condition_data[key] for key in keys])
        combined = safe_pattern_zscore(combined)
        mean = np.mean(combined, axis=0)
        std = np.std(combined, axis=0)
        std[std == 0] = 1
        combined = (combined - mean) / std
        cursor = 0
        for key, size in zip(keys, sizes):
            output[key] = combined[cursor : cursor + size]
            cursor += size
    return output


def preprocess_inside_fold(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Historical voxel-then-trial normalization inside each split."""

    train = safe_pattern_zscore(train.copy())
    test = safe_pattern_zscore(test.copy())
    mean, std = fit_trial_zscore(train)
    return apply_trial_zscore(train, mean, std), apply_trial_zscore(test, mean, std)


def average_chunks(array: np.ndarray, chunks: Sequence[np.ndarray]) -> np.ndarray:
    return np.asarray([np.mean(array[chunk], axis=0) for chunk in chunks])


def decode_within_avg(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, list[dict]]:
    accuracies: list[float] = []
    records: list[dict] = []
    indices = np.arange(first.shape[0])
    for repeat in range(N_REPEATS):
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + repeat)
        for fold_number, (train_index, test_index) in enumerate(splitter.split(indices), start=1):
            train = np.vstack([first[train_index], second[train_index]])
            test = np.vstack([first[test_index], second[test_index]])
            train, test = preprocess_inside_fold(train, test)
            n_train = len(train_index)
            n_test = len(test_index)
            first_train, second_train = train[:n_train], train[n_train:]
            first_test, second_test = test[:n_test], test[n_test:]

            chunks = np.array_split(np.arange(n_train), N_AVG_GROUPS)
            x_train = np.vstack(
                [average_chunks(first_train, chunks), average_chunks(second_train, chunks)]
            )
            y_train = np.asarray([1] * N_AVG_GROUPS + [0] * N_AVG_GROUPS)
            x_test = np.vstack(
                [
                    np.mean(first_test, axis=0, keepdims=True),
                    np.mean(second_test, axis=0, keepdims=True),
                ]
            )
            y_test = np.asarray([1, 0])
            classifier = SVC(kernel="linear", C=1.0)
            classifier.fit(x_train, y_train)
            accuracy = float(classifier.score(x_test, y_test))
            accuracies.append(accuracy)
            records.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold_number,
                    "n_train_trials_per_class": n_train,
                    "n_test_trials_per_class": n_test,
                    "n_train_averages_per_class": N_AVG_GROUPS,
                    "n_test_averages_per_class": 1,
                    "accuracy": accuracy,
                }
            )
    return float(np.mean(accuracies)), records


def decode_cross_avg(
    train_first: np.ndarray,
    train_second: np.ndarray,
    test_first: np.ndarray,
    test_second: np.ndarray,
) -> tuple[float, list[dict]]:
    accuracies: list[float] = []
    records: list[dict] = []
    for repeat in range(N_REPEATS):
        train = np.vstack([train_first, train_second])
        test = np.vstack([test_first, test_second])
        train, test = preprocess_inside_fold(train, test)
        n_train_first = len(train_first)
        n_test_first = len(test_first)
        permutation = np.random.RandomState(SEED + repeat).permutation(n_train_first)
        groups = np.array_split(permutation, N_FOLDS)
        x_train = np.vstack(
            [average_chunks(train[:n_train_first], groups), average_chunks(train[n_train_first:], groups)]
        )
        x_test = np.vstack(
            [average_chunks(test[:n_test_first], groups), average_chunks(test[n_test_first:], groups)]
        )
        y_train = np.asarray([1] * N_FOLDS + [0] * N_FOLDS)
        y_test = np.asarray([1] * N_FOLDS + [0] * N_FOLDS)
        classifier = SVC(kernel="linear", C=1.0)
        classifier.fit(x_train, y_train)
        accuracy = float(classifier.score(x_test, y_test))
        accuracies.append(accuracy)
        records.append(
            {
                "repeat": repeat + 1,
                "fold": 0,
                "n_train_trials_per_class": len(train_first),
                "n_test_trials_per_class": len(test_first),
                "n_train_averages_per_class": N_FOLDS,
                "n_test_averages_per_class": N_FOLDS,
                "accuracy": accuracy,
            }
        )
    return float(np.mean(accuracies)), records


def fit_source_normalization(
    data: np.ndarray,
    manifest: pd.DataFrame,
    train_source: str,
    training_runs: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the historical pattern/source transform using training runs only."""

    selected = np.flatnonzero(
        (manifest["source"].to_numpy() == train_source)
        & manifest["run"].isin(training_runs).to_numpy()
    )
    # All three valences are intentionally used, matching the historical
    # source-level normalization while excluding the outer test run.
    if len(selected) != len(training_runs) * 30:
        raise RuntimeError(
            f"Expected {len(training_runs) * 30} {train_source} source-normalization "
            f"trials, found {len(selected)}"
        )
    patterned = safe_pattern_zscore(data[selected])
    mean = np.mean(patterned, axis=0)
    std = np.std(patterned, axis=0)
    std[std == 0] = 1
    return mean, std


def apply_source_normalization(
    data: np.ndarray,
    index: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (safe_pattern_zscore(data[index]) - mean) / std


def decode_runwise_loro(
    data: np.ndarray,
    manifest: pd.DataFrame,
    train_source: str,
    test_source: str,
    positive: str,
) -> tuple[float, list[dict]]:
    """Run-wise leakage-controlled ERP decoding using condition means and LORO.

    Every fitted normalization parameter uses only the nine outer-training
    runs.  The classifier receives one condition mean per training run and one
    condition mean per class from the held-out run.
    """

    run_values = manifest["run"].to_numpy(dtype=int)
    source_values = manifest["source"].to_numpy()
    valence_values = manifest["valence"].to_numpy()
    runs = sorted(int(value) for value in np.unique(run_values))
    if runs != list(range(1, 11)):
        raise RuntimeError(f"Expected runs 1-10, found {runs}")
    fold_accuracies: list[float] = []
    records: list[dict] = []
    for held_run in runs:
        training_runs = [run for run in runs if run != held_run]
        source_mean, source_std = fit_source_normalization(
            data, manifest, train_source, training_runs
        )
        train_index = np.flatnonzero(
            (source_values == train_source)
            & (run_values != held_run)
            & np.isin(valence_values, [positive, "neutral"])
        )
        test_index = np.flatnonzero(
            (source_values == test_source)
            & (run_values == held_run)
            & np.isin(valence_values, [positive, "neutral"])
        )
        if len(train_index) != 180 or len(test_index) != 20:
            raise RuntimeError(
                f"Unexpected LORO sizes for run {held_run}: train={len(train_index)}, "
                f"test={len(test_index)}"
            )
        train = apply_source_normalization(
            data, train_index, source_mean, source_std
        )
        test = apply_source_normalization(data, test_index, source_mean, source_std)
        # Replicate the historical second pattern/voxel normalization, but fit
        # its voxel statistics strictly on the outer-training observations.
        train, test = preprocess_inside_fold(train, test)
        train_runs = run_values[train_index]
        train_valence = valence_values[train_index]
        test_valence = valence_values[test_index]

        train_patterns = []
        train_labels = []
        for label, valence in ((1, positive), (0, "neutral")):
            for run in training_runs:
                selected = (train_runs == run) & (train_valence == valence)
                if int(np.count_nonzero(selected)) != 10:
                    raise RuntimeError(
                        f"Expected 10 training trials for run {run}/{valence}"
                    )
                train_patterns.append(np.mean(train[selected], axis=0))
                train_labels.append(label)
        test_patterns = []
        for valence in (positive, "neutral"):
            selected = test_valence == valence
            if int(np.count_nonzero(selected)) != 10:
                raise RuntimeError(
                    f"Expected 10 test trials for run {held_run}/{valence}"
                )
            test_patterns.append(np.mean(test[selected], axis=0))

        classifier = SVC(kernel="linear", C=1.0)
        classifier.fit(np.asarray(train_patterns), np.asarray(train_labels))
        accuracy = float(
            classifier.score(np.asarray(test_patterns), np.asarray([1, 0]))
        )
        fold_accuracies.append(accuracy)
        records.append(
            {
                "held_out_run": held_run,
                "n_train_runs": 9,
                "n_test_runs": 1,
                "n_train_trials_per_class": 90,
                "n_test_trials_per_class": 10,
                "n_train_averages_per_class": 9,
                "n_test_averages_per_class": 1,
                "accuracy": accuracy,
            }
        )
    return float(np.mean(fold_accuracies)), records


def parse_mricrogl_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                label_id = int(float(parts[0]))
            except ValueError:
                continue
            if label_id > 0 and parts[1] != "*.*.*.*.*":
                labels[parts[1]] = label_id
    return labels


def load_atlas(
    path: Path, label_path: Path
) -> dict[str, nib.spatialimages.SpatialImage]:
    """Load the official 25-label Kastner/Wang atlas used by the attachment."""

    atlas_image = nib.load(str(path))
    atlas_data = np.rint(atlas_image.get_fdata()).astype(np.int32)
    label_ids = parse_mricrogl_labels(label_path)
    atlas: dict[str, nib.spatialimages.SpatialImage] = {}
    for roi in ROI_ORDER:
        missing = [name for name in ROI_LABEL_NAMES[roi] if name not in label_ids]
        if missing:
            raise ValueError(f"Missing atlas labels for {roi}: {missing}")
        member_ids = [label_ids[name] for name in ROI_LABEL_NAMES[roi]]
        mask = np.isin(atlas_data, member_ids).astype(np.uint8)
        if not np.any(mask):
            raise ValueError(f"Empty atlas ROI: {roi}")
        atlas[roi] = nib.Nifti1Image(mask, atlas_image.affine)
    return atlas


def same_grid(left: nib.spatialimages.SpatialImage, right: nib.spatialimages.SpatialImage) -> bool:
    return left.shape == right.shape and np.allclose(left.affine, right.affine, atol=1e-6, rtol=0)


def prepare_rois(
    atlas: Mapping[str, nib.spatialimages.SpatialImage],
    reference: nib.spatialimages.SpatialImage,
    analysis_mask: np.ndarray,
    branch: str,
    subject: int,
) -> GridRois:
    roi_full_indices: dict[str, np.ndarray] = {}
    count_rows: list[dict] = []
    for name in ROI_ORDER:
        image = atlas[name]
        atlas_count = int(np.count_nonzero(np.asarray(image.dataobj) > 0))
        was_resampled = not same_grid(image, reference)
        cache_key = (
            name,
            tuple(reference.shape),
            np.round(np.asarray(reference.affine), 6).tobytes(),
        )
        grid_mask = _RESAMPLED_ROI_CACHE.get(cache_key)
        if grid_mask is None:
            if was_resampled:
                transformed = resample_from_to(
                    image,
                    (reference.shape, reference.affine),
                    order=0,
                    mode="constant",
                    cval=0,
                )
            else:
                transformed = image
            grid_mask = np.asarray(transformed.dataobj) > 0.5
            _RESAMPLED_ROI_CACHE[cache_key] = grid_mask
        in_analysis = grid_mask & analysis_mask
        full_indices = np.flatnonzero(in_analysis.ravel()).astype(np.int64)
        if len(full_indices) < 10:
            raise ValueError(
                f"{branch} Sub{subject} {name} has only {len(full_indices)} in-mask voxels"
            )
        roi_full_indices[name] = full_indices
        count_rows.append(
            {
                "subject": f"Sub{subject}",
                "pipeline": branch,
                "roi": name,
                "atlas_native_voxels": atlas_count,
                "target_grid_voxels_before_analysis_mask": int(np.count_nonzero(grid_mask)),
                "analysis_voxels": int(len(full_indices)),
                "roi_resampled_nearest_neighbor": bool(was_resampled),
                "target_shape": "x".join(str(value) for value in reference.shape),
                "target_affine": json.dumps(np.asarray(reference.affine).tolist()),
            }
        )
    union = np.unique(np.concatenate(list(roi_full_indices.values()))).astype(np.int64)
    positions = {name: np.searchsorted(union, values) for name, values in roi_full_indices.items()}
    for name, values in roi_full_indices.items():
        if not np.array_equal(union[positions[name]], values):
            raise RuntimeError(f"Could not map ROI {name} into union")
    return GridRois(union, positions, count_rows)


def restrict_rois_to_valid_columns(
    rois: GridRois, valid: np.ndarray
) -> GridRois:
    if valid.ndim != 1 or len(valid) != len(rois.union_flat_indices):
        raise ValueError("ROI validity vector has the wrong shape")
    old_to_new = np.full(len(valid), -1, dtype=np.int64)
    old_to_new[valid] = np.arange(int(np.count_nonzero(valid)), dtype=np.int64)
    positions: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for roi, row in zip(ROI_ORDER, rois.count_rows):
        kept = rois.roi_union_positions[roi][valid[rois.roi_union_positions[roi]]]
        positions[roi] = old_to_new[kept]
        if len(positions[roi]) < 10:
            raise ValueError(f"Only {len(positions[roi])} valid voxels remain in {roi}")
        updated = dict(row)
        updated["analysis_voxels"] = int(len(positions[roi]))
        updated["valid_voxel_rule"] = "finite across all 600 trial betas"
        rows.append(updated)
    return GridRois(rois.union_flat_indices[valid], positions, rows)


def load_legacy_roi_data(
    subject: int,
    beta_directory: Path,
    beta_files: Sequence[str],
    atlas: Mapping[str, nib.spatialimages.SpatialImage],
) -> tuple[np.ndarray, GridRois]:
    """Load the exact individual SPM LS-A betas used by the attached result."""

    if len(beta_files) != 600:
        raise ValueError(f"Expected 600 legacy beta files, found {len(beta_files)}")
    paths = [beta_directory / name for name in beta_files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing legacy betas, first: {missing[:3]}")
    reference = nib.load(str(paths[0]))
    analysis_mask = np.ones(reference.shape, dtype=bool)
    rois = prepare_rois(atlas, reference, analysis_mask, "legacy_lsa_8mm", subject)
    data = np.empty((600, len(rois.union_flat_indices)), dtype=np.float32)
    for index, path in enumerate(paths):
        image = nib.load(str(path))
        if not same_grid(image, reference):
            raise ValueError(f"Legacy grid mismatch: {path}")
        data[index] = image.get_fdata(dtype=np.float32).ravel()[rois.union_flat_indices]
        if (index + 1) % 100 == 0:
            print(f"    loaded {index + 1}/600 legacy betas", flush=True)
    valid = np.all(np.isfinite(data), axis=0)
    rois = restrict_rois_to_valid_columns(rois, valid)
    data = data[:, valid]
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Nonfinite values remain in legacy ROI data for Sub{subject}")
    return data, rois


def smooth_and_extract(
    data: np.ndarray,
    mask: np.ndarray,
    flat_indices: np.ndarray,
    selected_columns: np.ndarray,
    affine: np.ndarray,
    fwhm_mm: float,
    truncate: float = 4.0,
) -> np.ndarray:
    zooms = nib.affines.voxel_sizes(affine)
    sigma_mm = fwhm_mm / math.sqrt(8.0 * math.log(2.0))
    sigma_vox = sigma_mm / zooms
    denominator = gaussian_filter(
        mask.astype(np.float32), sigma=sigma_vox, mode="constant", cval=0.0, truncate=truncate
    )
    selected_full = flat_indices[selected_columns]
    selected_denominator = denominator.ravel()[selected_full]
    if np.any(selected_denominator <= np.finfo(np.float32).eps):
        raise RuntimeError("Zero smoothing denominator in selected ROI")
    output = np.empty((data.shape[0], len(selected_columns)), dtype=np.float32)
    volume = np.zeros(mask.shape, dtype=np.float32)
    for trial in range(data.shape[0]):
        volume.fill(0.0)
        volume.ravel()[flat_indices] = data[trial]
        smoothed = gaussian_filter(
            volume, sigma=sigma_vox, mode="constant", cval=0.0, truncate=truncate
        )
        output[trial] = smoothed.ravel()[selected_full] / selected_denominator
        if (trial + 1) % 100 == 0:
            print(f"    smoothed {trial + 1}/{data.shape[0]} trials", flush=True)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Nonfinite values after selected-ROI smoothing")
    return output


def load_glmsingle_roi_variants(
    subject: int,
    subject_root: Path,
    atlas: Mapping[str, nib.spatialimages.SpatialImage],
) -> tuple[np.ndarray, np.ndarray, GridRois]:
    spec = BranchSpec(
        name="glmsingle_typed",
        kind="glmsingle_type_d_hdf5",
        beta_path=subject_root / "glmsingle" / TYPE_D_NAME,
        mask_path=subject_root / "analysis_mask.nii.gz",
        manifest_path=subject_root / "trial_manifest.tsv",
        smoothing_fwhm_mm=(0.0, 3.0),
        flat_indices_path=subject_root / "flat_mask_indices.npy",
    )
    beta = load_glmsingle_betas(spec, n_trials=600, trial_block=16)
    rois = prepare_rois(atlas, beta.reference, beta.mask, "glmsingle_typed", subject)
    selected_columns = np.searchsorted(beta.flat_indices, rois.union_flat_indices)
    if np.any(selected_columns >= len(beta.flat_indices)) or not np.array_equal(
        beta.flat_indices[selected_columns], rois.union_flat_indices
    ):
        raise RuntimeError(f"ROI/GLMsingle mask mapping failed for Sub{subject}")
    unsmoothed = beta.data[:, selected_columns].copy()
    print(f"  Sub{subject}: independently smoothing Type-D betas 3 mm", flush=True)
    smoothed = smooth_and_extract(
        beta.data,
        beta.mask,
        beta.flat_indices,
        selected_columns,
        beta.reference.affine,
        3.0,
    )
    del beta
    gc.collect()
    return unsmoothed, smoothed, rois


def condition_arrays(
    data: np.ndarray, manifest: pd.DataFrame
) -> dict[tuple[str, str], np.ndarray]:
    output: dict[tuple[str, str], np.ndarray] = {}
    for source in ("natural", "ai"):
        for valence in ("pleasant", "neutral", "unpleasant"):
            index = np.flatnonzero(
                (manifest["source"].to_numpy() == source)
                & (manifest["valence"].to_numpy() == valence)
            )
            if len(index) != 100:
                raise ValueError(f"Expected 100 {source}/{valence} trials, found {len(index)}")
            output[(source, valence)] = data[index]
    return output


def decode_branch(
    subject: int,
    pipeline: str,
    data: np.ndarray,
    rois: GridRois,
    manifest: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    subject_rows: list[dict] = []
    fold_rows: list[dict] = []
    for roi in ROI_ORDER:
        roi_data = data[:, rois.roi_union_positions[roi]]
        if roi_data.shape[1] < 10 or not np.all(np.isfinite(roi_data)):
            raise ValueError(f"Invalid {pipeline} Sub{subject} {roi} data: {roi_data.shape}")
        raw_conditions = condition_arrays(roi_data, manifest)
        normalized = compute_voxel_source_zscore(raw_conditions)
        for name, kind, train_source, test_source, positive in CONTRASTS:
            if kind == "within":
                accuracy, details = decode_within_avg(
                    normalized[(train_source, positive)],
                    normalized[(train_source, "neutral")],
                )
            else:
                accuracy, details = decode_cross_avg(
                    normalized[(train_source, positive)],
                    normalized[(train_source, "neutral")],
                    normalized[(test_source, positive)],
                    normalized[(test_source, "neutral")],
                )
            subject_rows.append(
                {
                    "subject": f"Sub{subject}",
                    "pipeline": pipeline,
                    "analysis_kind": kind,
                    "contrast": name,
                    "contrast_label": CONTRAST_LABELS[name],
                    "roi": roi,
                    "n_voxels": int(roi_data.shape[1]),
                    "accuracy": accuracy,
                }
            )
            for detail in details:
                fold_rows.append(
                    {
                        "subject": f"Sub{subject}",
                        "pipeline": pipeline,
                        "analysis_kind": kind,
                        "contrast": name,
                        "roi": roi,
                        **detail,
                    }
                )
    return subject_rows, fold_rows


def decode_branch_runwise_safe(
    subject: int,
    pipeline: str,
    data: np.ndarray,
    rois: GridRois,
    manifest: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    subject_rows: list[dict] = []
    fold_rows: list[dict] = []
    for roi in ROI_ORDER:
        roi_data = data[:, rois.roi_union_positions[roi]]
        if roi_data.shape[1] < 10 or not np.all(np.isfinite(roi_data)):
            raise ValueError(f"Invalid {pipeline} Sub{subject} {roi} data: {roi_data.shape}")
        for name, kind, train_source, test_source, positive in CONTRASTS:
            accuracy, details = decode_runwise_loro(
                roi_data,
                manifest,
                train_source,
                test_source,
                positive,
            )
            subject_rows.append(
                {
                    "subject": f"Sub{subject}",
                    "pipeline": pipeline,
                    "analysis_kind": kind,
                    "contrast": name,
                    "contrast_label": CONTRAST_LABELS[name],
                    "roi": roi,
                    "n_voxels": int(roi_data.shape[1]),
                    "accuracy": accuracy,
                }
            )
            for detail in details:
                fold_rows.append(
                    {
                        "subject": f"Sub{subject}",
                        "pipeline": pipeline,
                        "analysis_kind": kind,
                        "contrast": name,
                        "roi": roi,
                        **detail,
                    }
                )
    return subject_rows, fold_rows


def load_historical(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    historical_mode = saved.get("avg_z_mode")
    if historical_mode != MODE:
        raise ValueError(f"Historical result mode is {historical_mode!r}, expected {MODE!r}")
    available_kastner = tuple(saved.get("kastner_roi_names", []))
    missing_rois = [roi for roi in ROI_ORDER if roi not in available_kastner]
    if missing_rois:
        raise ValueError(f"Historical result lacks attachment ROIs: {missing_rois}")
    all_subjects = list(saved["subs"])
    subjects = [name for name in all_subjects if name not in {"Sub3", "Sub8"}]
    expected = [f"Sub{value}" for value in range(1, 32) if value not in {3, 8, 10}]
    if subjects != expected or len(subjects) != 28:
        raise ValueError(f"Historical cohort is not the expected 28 subjects: {subjects}")
    rows: list[dict] = []
    for subject in subjects:
        for name, kind, *_ in CONTRASTS:
            key = HISTORICAL_KEYS[name]
            container = saved["within_avg"] if kind == "within" else saved["cross_avg"]
            for roi in ROI_ORDER:
                accuracy = float(container[subject][key][MODE][roi])
                rows.append(
                    {
                        "subject": subject,
                        "pipeline": "historical_28sub_original",
                        "analysis_kind": kind,
                        "contrast": name,
                        "contrast_label": CONTRAST_LABELS[name],
                        "roi": roi,
                        "accuracy": accuracy,
                    }
                )
    frame = pd.DataFrame(rows)
    if len(frame) != 28 * len(CONTRASTS) * len(ROI_ORDER) or not frame[
        "accuracy"
    ].between(0, 1).all():
        raise RuntimeError("Historical result extraction failed")
    return frame


def summarize_subject_results(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["pipeline", "analysis_kind", "contrast", "contrast_label", "roi"], sort=False)
        .agg(n_subjects=("subject", "nunique"), mean_accuracy=("accuracy", "mean"), sem_accuracy=("accuracy", sem))
        .reset_index()
    )


def collapsed_subject_results(frame: pd.DataFrame) -> pd.DataFrame:
    by_kind = (
        frame.groupby(["pipeline", "subject", "analysis_kind"], sort=False)["accuracy"]
        .mean()
        .reset_index()
    )
    overall = (
        frame.groupby(["pipeline", "subject"], sort=False)["accuracy"]
        .mean()
        .reset_index()
        .assign(analysis_kind="overall")
    )
    return pd.concat([by_kind, overall], ignore_index=True)


def collapse_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["pipeline", "analysis_kind"], sort=False)
        .agg(n_subjects=("subject", "nunique"), mean_accuracy=("accuracy", "mean"), sem_accuracy=("accuracy", sem))
        .reset_index()
    )


def reproduction_check(pilot: pd.DataFrame, historical: pd.DataFrame) -> dict:
    legacy = pilot[pilot["pipeline"] == "legacy_lsa_8mm"]
    reference = historical[historical["subject"].isin(["Sub4", "Sub5", "Sub6"])]
    keys = ["subject", "analysis_kind", "contrast", "roi"]
    merged = legacy.merge(reference, on=keys, suffixes=("_recomputed", "_historical"), validate="one_to_one")
    difference = np.abs(merged["accuracy_recomputed"] - merged["accuracy_historical"])
    result = {
        "n_compared": int(len(merged)),
        "n_exact_within_1e-10": int(np.count_nonzero(difference <= 1e-10)),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "tolerance": REPRODUCTION_TOLERANCE,
        "passed": bool(np.all(difference <= REPRODUCTION_TOLERANCE)),
    }
    if len(merged) != 3 * len(CONTRASTS) * len(ROI_ORDER) or not result["passed"]:
        raise RuntimeError(f"Legacy historical reproduction failed: {result}")
    return result


def save_attached_style_plot(summary: pd.DataFrame, pipeline: str, path: Path) -> None:
    colors_within = ["#1f4e79", "#8ecae6", "#b45f06", "#f6b26b"]
    colors_cross = ["#1b7837", "#a6dba0", "#8b0000", "#f4a3a3"]
    orders = [
        [item[0] for item in CONTRASTS if item[1] == "within"],
        [item[0] for item in CONTRASTS if item[1] == "cross"],
    ]
    figure, axes = plt.subplots(2, 1, figsize=(18, 12), constrained_layout=True)
    for axis, kind, order, colors in zip(axes, ("within", "cross"), orders, (colors_within, colors_cross)):
        subset = summary[(summary["pipeline"] == pipeline) & (summary["analysis_kind"] == kind)]
        x = np.arange(len(ROI_ORDER))
        width = 0.18
        for index, (contrast, color) in enumerate(zip(order, colors)):
            rows = subset[subset["contrast"] == contrast].set_index("roi").loc[list(ROI_ORDER)]
            offset = (index - 1.5) * width
            axis.bar(
                x + offset,
                rows["mean_accuracy"],
                width,
                yerr=rows["sem_accuracy"],
                label=CONTRAST_LABELS[contrast],
                color=color,
                edgecolor="black",
                linewidth=0.35,
                capsize=2,
                alpha=0.9,
            )
        axis.axhline(0.5, color="0.35", linestyle="--", linewidth=1)
        axis.set_ylim(0.40, 0.90)
        axis.set_ylabel("Accuracy")
        axis.set_title(f"{kind.title()} source — {PIPELINE_LABELS[pipeline]}", fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels(ROI_ORDER, rotation=45, ha="right")
        axis.legend(loc="lower right", fontsize=9, ncol=2)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_kastner_pipeline_grid(
    summary: pd.DataFrame,
    pipelines: Sequence[str],
    path: Path,
    title: str,
) -> None:
    """Plot ROI-resolved decoding in the same two-panel layout as the source figure."""

    colors_within = ["#1f4e79", "#8ecae6", "#b45f06", "#f6b26b"]
    colors_cross = ["#1b7837", "#a6dba0", "#8b0000", "#f4a3a3"]
    orders = {
        "within": [item[0] for item in CONTRASTS if item[1] == "within"],
        "cross": [item[0] for item in CONTRASTS if item[1] == "cross"],
    }
    figure, axes = plt.subplots(
        len(pipelines),
        2,
        figsize=(24, 3.6 * len(pipelines) + 1.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    legend_handles: dict[str, list] = {}
    for row_index, pipeline in enumerate(pipelines):
        for column_index, kind in enumerate(("within", "cross")):
            axis = axes[row_index, column_index]
            subset = summary[
                (summary["pipeline"] == pipeline)
                & (summary["analysis_kind"] == kind)
            ]
            x = np.arange(len(ROI_ORDER))
            width = 0.18
            bars = []
            colors = colors_within if kind == "within" else colors_cross
            for contrast_index, (contrast, color) in enumerate(
                zip(orders[kind], colors)
            ):
                rows = (
                    subset[subset["contrast"] == contrast]
                    .set_index("roi")
                    .loc[list(ROI_ORDER)]
                )
                offset = (contrast_index - 1.5) * width
                container = axis.bar(
                    x + offset,
                    rows["mean_accuracy"],
                    width,
                    yerr=rows["sem_accuracy"],
                    label=CONTRAST_LABELS[contrast],
                    color=color,
                    edgecolor="black",
                    linewidth=0.3,
                    capsize=1.5,
                    alpha=0.9,
                )
                bars.append(container)
            if row_index == 0:
                legend_handles[kind] = bars
                axis.set_title(f"{kind.title()} source", fontweight="bold")
            axis.axhline(0.5, color="0.35", linestyle="--", linewidth=1)
            axis.set_ylim(0.40, 0.90)
            axis.grid(axis="y", alpha=0.18)
            if column_index == 0:
                axis.set_ylabel(f"{PIPELINE_LABELS[pipeline]}\nAccuracy")
            if row_index == len(pipelines) - 1:
                axis.set_xticks(x)
                axis.set_xticklabels(ROI_ORDER, rotation=45, ha="right")
                axis.tick_params(axis="x", labelbottom=True)

    figure.suptitle(title, fontsize=18, fontweight="bold", y=0.995)
    figure.legend(
        legend_handles["within"],
        [CONTRAST_LABELS[value] for value in orders["within"]],
        loc="upper center",
        bbox_to_anchor=(0.28, 0.975),
        ncol=2,
        fontsize=9,
    )
    figure.legend(
        legend_handles["cross"],
        [CONTRAST_LABELS[value] for value in orders["cross"]],
        loc="upper center",
        bbox_to_anchor=(0.75, 0.975),
        ncol=2,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_collapsed_plot(
    summary: pd.DataFrame,
    path: Path,
    pipelines: Sequence[str] = PIPELINE_ORDER,
    title: str = "ERP-style Avg(random): historical result and matched pilot pipelines",
) -> None:
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    kinds = ("within", "cross", "overall")
    x = np.arange(len(kinds))
    width = min(0.24, 0.76 / len(pipelines))
    colors = ["#595959", "#4c78a8", "#59a14f", "#e15759"]
    for index, (pipeline, color) in enumerate(zip(pipelines, colors[-len(pipelines) :])):
        rows = summary[summary["pipeline"] == pipeline].set_index("analysis_kind").loc[list(kinds)]
        offset = (index - (len(pipelines) - 1) / 2) * width
        axis.bar(
            x + offset,
            rows["mean_accuracy"],
            width,
            yerr=rows["sem_accuracy"],
            label=PIPELINE_LABELS[pipeline],
            color=color,
            edgecolor="black",
            linewidth=0.4,
            capsize=3,
        )
    axis.axhline(0.5, color="0.35", linestyle="--", linewidth=1)
    axis.set_ylim(0.45, 0.72)
    axis.set_ylabel("Mean ROI accuracy")
    axis.set_xticks(x)
    axis.set_xticklabels(["Within source", "Cross source", "Overall"])
    axis.set_title(title, fontweight="bold")
    axis.legend(fontsize=9)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_matched_delta_plot(pilot: pd.DataFrame, path: Path) -> None:
    reduced = (
        pilot.groupby(["pipeline", "subject", "analysis_kind", "roi"], sort=False)["accuracy"]
        .mean()
        .reset_index()
    )
    wide = reduced.pivot(index=["subject", "analysis_kind", "roi"], columns="pipeline", values="accuracy")
    figure, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    for axis, kind in zip(axes, ("within", "cross")):
        subset = wide.xs(kind, level="analysis_kind")
        x = np.arange(len(ROI_ORDER))
        for pipeline, color, marker in (
            ("glmsingle_typed_0mm", "#59a14f", "o"),
            ("glmsingle_typed_3mm", "#e15759", "s"),
        ):
            delta = (subset[pipeline] - subset["legacy_lsa_8mm"]).groupby("roi").mean().loc[list(ROI_ORDER)]
            axis.plot(x, 100 * delta, marker=marker, linewidth=1.8, color=color, label=PIPELINE_LABELS[pipeline])
        axis.axhline(0, color="0.35", linewidth=1)
        axis.set_ylabel("Matched change (percentage points)")
        axis.set_title(f"{kind.title()} source: GLMsingle minus pilot legacy", fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels(ROI_ORDER, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=9)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_report(
    output: Path,
    collapsed: pd.DataFrame,
    pilot_collapsed: pd.DataFrame,
    safe_collapsed: pd.DataFrame,
    safe_subject_collapsed: pd.DataFrame,
    reproduction: dict,
    roi_delta: pd.DataFrame,
) -> None:
    lookup = collapsed.set_index(["pipeline", "analysis_kind"])["mean_accuracy"]
    lines = [
        "# ERP-style decoding comparison: Subjects 4, 5 and 6",
        "",
        "## Scope",
        "",
        "This report applies the exact `voxel_trial_source_zscore` Avg(random) decoder behind the attached figure to the three matched beta pipelines. The attached historical reference contains the predetermined 28-subject cohort: Subjects 1–31 excluding Subjects 3, 8 and 10.",
        "",
        "## ROI-collapsed accuracy",
        "",
        "| pipeline | within | cross | overall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for pipeline in PIPELINE_ORDER:
        lines.append(
            f"| {PIPELINE_LABELS[pipeline]} | {lookup[pipeline, 'within']:.4f} | "
            f"{lookup[pipeline, 'cross']:.4f} | {lookup[pipeline, 'overall']:.4f} |"
        )

    pilot_lookup = pilot_collapsed.pivot(
        index=["subject", "analysis_kind"], columns="pipeline", values="accuracy"
    )
    lines.extend(
        [
            "",
            "## Matched pipeline changes",
            "",
            "These differences use the same three subjects and are the valid descriptive pipeline comparison. Values are percentage-point changes relative to the pilot legacy LS-A branch.",
            "",
            "| pipeline | within | cross | overall |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for pipeline in ("glmsingle_typed_0mm", "glmsingle_typed_3mm"):
        values = {}
        for kind in ("within", "cross", "overall"):
            subset = pilot_lookup.xs(kind, level="analysis_kind")
            values[kind] = 100 * float(np.mean(subset[pipeline] - subset["legacy_lsa_8mm"]))
        lines.append(
            f"| {PIPELINE_LABELS[pipeline]} | {values['within']:+.2f} | "
            f"{values['cross']:+.2f} | {values['overall']:+.2f} |"
        )

    lines.extend(["", "### Exact attached-style ROI extrema"])
    for kind in ("within", "cross"):
        subset = roi_delta[
            (roi_delta["pipeline"] == "glmsingle_typed_3mm")
            & (roi_delta["analysis_kind"] == kind)
        ]
        best = subset.loc[subset["mean_delta"].idxmax()]
        worst = subset.loc[subset["mean_delta"].idxmin()]
        lines.extend(
            [
                "",
                f"For {kind}-source decoding, the largest ROI-averaged Type-D +3 mm change was {best['roi']} ({100 * best['mean_delta']:+.2f} points); the smallest was {worst['roi']} ({100 * worst['mean_delta']:+.2f} points).",
            ]
        )

    safe_lookup = safe_collapsed.set_index(["pipeline", "analysis_kind"])["mean_accuracy"]
    safe_wide = safe_subject_collapsed.pivot(
        index=["subject", "analysis_kind"], columns="pipeline", values="accuracy"
    )
    lines.extend(
        [
            "",
            "## Run-wise leakage-controlled ERP sensitivity",
            "",
            "Here each source × valence × run is averaged into one pattern. The outer fold leaves one complete acquisition run out, and every normalization parameter is fit on the other nine runs only.",
            "",
            "| pipeline | within | cross | overall | overall change vs legacy |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for pipeline in ("legacy_lsa_8mm", "glmsingle_typed_0mm", "glmsingle_typed_3mm"):
        if pipeline == "legacy_lsa_8mm":
            delta = 0.0
        else:
            subset = safe_wide.xs("overall", level="analysis_kind")
            delta = 100 * float(np.mean(subset[pipeline] - subset["legacy_lsa_8mm"]))
        lines.append(
            f"| {PIPELINE_LABELS[pipeline]} | {safe_lookup[pipeline, 'within']:.4f} | "
            f"{safe_lookup[pipeline, 'cross']:.4f} | {safe_lookup[pipeline, 'overall']:.4f} | "
            f"{delta:+.2f} points |"
        )
    lines.extend(
        [
            "",
            "This LORO analysis blocks acquisition runs and prevents train/test normalization leakage, but it is not stimulus-identity-held-out: the 600 trials contain 120 image identities repeated across runs.",
        ]
    )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"Recomputed pilot legacy values were checked against the saved historical subject-level values for {reproduction['n_compared']} Subject × contrast × ROI cells. Maximum absolute difference: `{reproduction['max_abs_difference']:.3g}`; validation: **{'PASS' if reproduction['passed'] else 'FAIL'}**.",
            "",
            "## Methodological guardrail",
            "",
            "The attached-style comparison is an exact historical-method replication, not leakage-controlled performance estimation. Source-level mean and variance were estimated from all trials before the random folds, and random folds do not block acquisition runs or stimulus identities. Averaging also yields highly correlated pseudo-trials. Consequently, these accuracies can be substantially higher than leave-one-run-out results and should not be interpreted as evidence that the underlying fMRI signal is better. The run-wise LORO sensitivity above and the previously completed LORO single-trial comparison are more appropriate run-generalization checks; an identity-held-out analysis would be needed for novel-stimulus generalization.",
            "",
            "The historical 28-subject bar and each three-subject pilot bar also have different sample sizes; only matched Sub4/5/6 branch differences isolate pipeline effects.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_self_test() -> None:
    rng = np.random.RandomState(7)
    features = 24
    common = rng.normal(size=(100, features)).astype(np.float32)
    first = common + rng.normal(scale=0.5, size=common.shape)
    second = common + rng.normal(scale=0.5, size=common.shape)
    first[:, :5] += 0.8
    second[:, :5] -= 0.8
    within_a, rows_a = decode_within_avg(first, second)
    within_b, rows_b = decode_within_avg(first, second)
    cross, cross_rows = decode_cross_avg(first, second, first + 0.1, second + 0.1)
    if within_a != within_b or rows_a != rows_b:
        raise AssertionError("Within decoder is not deterministic")
    if len(rows_a) != N_REPEATS * N_FOLDS or len(cross_rows) != N_REPEATS:
        raise AssertionError("Unexpected fold count")
    if not (0 <= within_a <= 1 and 0 <= cross <= 1):
        raise AssertionError("Accuracy is outside [0, 1]")
    print("SELF-TEST PASSED: deterministic historical Avg(random) decoder")


def run(args: argparse.Namespace) -> None:
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        historical = load_historical(args.historical_result)
        historical.to_csv(staging / "historical_28sub_subject_results.csv", index=False)
        historical_summary = summarize_subject_results(historical)
        historical_summary.to_csv(staging / "historical_28sub_summary.csv", index=False)
        atlas = load_atlas(args.atlas, args.atlas_labels)
        legacy_manifest = normalize_manifest(args.legacy_manifest)
        beta_groups = pd.read_csv(
            args.legacy_groups,
            header=None,
            names=["beta_file", "category"],
        )
        if len(beta_groups) != 600 or beta_groups["beta_file"].duplicated().any():
            raise ValueError("beta_groups.csv must contain 600 unique beta files")
        expected_category = legacy_manifest["valence"].astype(str) + np.where(
            legacy_manifest["source"].to_numpy() == "ai", "AI", ""
        )
        if not np.array_equal(
            beta_groups["category"].astype(str).to_numpy(),
            np.asarray(expected_category, dtype=str),
        ):
            raise ValueError("beta_groups.csv categories do not align with the trial manifest")
        beta_files = beta_groups["beta_file"].astype(str).tolist()

        subject_rows: list[dict] = []
        fold_rows: list[dict] = []
        safe_subject_rows: list[dict] = []
        safe_fold_rows: list[dict] = []
        count_rows: list[dict] = []
        input_records: list[dict] = []

        for subject in args.subjects:
            print(f"Sub{subject}: loading and decoding legacy LS-A", flush=True)
            legacy_directory = args.legacy_root / f"Sub{subject}"
            legacy_data, legacy_rois = load_legacy_roi_data(
                subject, legacy_directory, beta_files, atlas
            )
            rows, folds = decode_branch(
                subject,
                "legacy_lsa_8mm",
                legacy_data,
                legacy_rois,
                legacy_manifest,
            )
            subject_rows.extend(rows)
            fold_rows.extend(folds)
            safe_rows, safe_folds = decode_branch_runwise_safe(
                subject,
                "legacy_lsa_8mm",
                legacy_data,
                legacy_rois,
                legacy_manifest,
            )
            safe_subject_rows.extend(safe_rows)
            safe_fold_rows.extend(safe_folds)
            count_rows.extend(legacy_rois.count_rows)
            legacy_paths = [legacy_directory / name for name in beta_files]
            input_records.append(
                {
                    "path": str(legacy_directory.resolve()),
                    "kind": "600 individual SPM LS-A beta NIfTIs",
                    "n_files": len(legacy_paths),
                    "total_size_bytes": int(sum(path.stat().st_size for path in legacy_paths)),
                    "first_file": fingerprint(legacy_paths[0]),
                    "last_file": fingerprint(legacy_paths[-1]),
                }
            )
            del legacy_data
            gc.collect()

            print(f"Sub{subject}: loading GLMsingle Type-D", flush=True)
            glmsingle_root = args.glmsingle_root / f"Sub{subject}"
            glmsingle_manifest = normalize_manifest(glmsingle_root / "trial_manifest.tsv")
            assert_manifests_aligned(legacy_manifest, glmsingle_manifest)
            unsmoothed, smoothed, glmsingle_rois = load_glmsingle_roi_variants(
                subject, glmsingle_root, atlas
            )
            for pipeline, values in (
                ("glmsingle_typed_0mm", unsmoothed),
                ("glmsingle_typed_3mm", smoothed),
            ):
                print(f"Sub{subject}: decoding {pipeline}", flush=True)
                rows, folds = decode_branch(
                    subject,
                    pipeline,
                    values,
                    glmsingle_rois,
                    glmsingle_manifest,
                )
                subject_rows.extend(rows)
                fold_rows.extend(folds)
                safe_rows, safe_folds = decode_branch_runwise_safe(
                    subject,
                    pipeline,
                    values,
                    glmsingle_rois,
                    glmsingle_manifest,
                )
                safe_subject_rows.extend(safe_rows)
                safe_fold_rows.extend(safe_folds)
                for item in glmsingle_rois.count_rows:
                    copy = dict(item)
                    copy["pipeline"] = pipeline
                    count_rows.append(copy)
            input_records.extend(
                [
                    fingerprint(glmsingle_root / "glmsingle" / TYPE_D_NAME),
                    fingerprint(glmsingle_root / "analysis_mask.nii.gz"),
                    fingerprint(glmsingle_root / "trial_manifest.tsv"),
                ]
            )
            del unsmoothed, smoothed
            gc.collect()

        pilot = pd.DataFrame(subject_rows)
        folds = pd.DataFrame(fold_rows)
        safe_pilot = pd.DataFrame(safe_subject_rows)
        safe_folds = pd.DataFrame(safe_fold_rows)
        counts = pd.DataFrame(count_rows)
        expected_subject_rows = len(args.subjects) * 3 * len(CONTRASTS) * len(ROI_ORDER)
        expected_fold_rows = len(args.subjects) * 3 * len(ROI_ORDER) * (
            4 * N_REPEATS * N_FOLDS + 4 * N_REPEATS
        )
        expected_safe_fold_rows = (
            len(args.subjects) * 3 * len(ROI_ORDER) * len(CONTRASTS) * 10
        )
        if (
            len(pilot) != expected_subject_rows
            or len(folds) != expected_fold_rows
            or len(safe_pilot) != expected_subject_rows
            or len(safe_folds) != expected_safe_fold_rows
        ):
            raise RuntimeError(
                f"Unexpected output sizes: subject={len(pilot)}/{expected_subject_rows}, "
                f"fold={len(folds)}/{expected_fold_rows}, "
                f"safe_subject={len(safe_pilot)}/{expected_subject_rows}, "
                f"safe_fold={len(safe_folds)}/{expected_safe_fold_rows}"
            )
        if not all(
            frame["accuracy"].between(0, 1).all()
            for frame in (pilot, folds, safe_pilot, safe_folds)
        ):
            raise RuntimeError("Accuracy outside [0, 1]")

        reproduction = reproduction_check(pilot, historical)
        pilot.to_csv(staging / "pilot_subject_results.csv", index=False)
        folds.to_csv(staging / "pilot_fold_results.csv", index=False)
        safe_pilot.to_csv(staging / "runwise_loro_subject_results.csv", index=False)
        safe_folds.to_csv(staging / "runwise_loro_fold_results.csv", index=False)
        counts.to_csv(staging / "roi_voxel_counts.csv", index=False)

        pilot_summary = summarize_subject_results(pilot)
        pilot_summary.to_csv(staging / "pilot_summary.csv", index=False)
        safe_summary = summarize_subject_results(safe_pilot)
        safe_summary.to_csv(staging / "runwise_loro_summary.csv", index=False)
        combined = pd.concat([historical, pilot], ignore_index=True, sort=False)
        combined_summary = pd.concat([historical_summary, pilot_summary], ignore_index=True)
        combined_summary.to_csv(staging / "all_pipeline_roi_summary.csv", index=False)

        collapsed_subject = collapsed_subject_results(combined)
        collapsed = collapse_summary(collapsed_subject)
        collapsed_subject.to_csv(staging / "collapsed_subject_results.csv", index=False)
        collapsed.to_csv(staging / "collapsed_pipeline_summary.csv", index=False)

        pilot_collapsed = collapsed_subject[
            collapsed_subject["pipeline"] != "historical_28sub_original"
        ].copy()
        pilot_collapsed.to_csv(staging / "pilot_collapsed_subject_results.csv", index=False)
        pivot = pilot_collapsed.pivot(
            index=["subject", "analysis_kind"], columns="pipeline", values="accuracy"
        ).reset_index()
        for pipeline in ("glmsingle_typed_0mm", "glmsingle_typed_3mm"):
            pivot[f"{pipeline}_minus_legacy"] = pivot[pipeline] - pivot["legacy_lsa_8mm"]
        pivot.to_csv(staging / "matched_pipeline_differences.csv", index=False)

        safe_subject_collapsed = collapsed_subject_results(safe_pilot)
        safe_collapsed = collapse_summary(safe_subject_collapsed)
        safe_subject_collapsed.to_csv(
            staging / "runwise_loro_collapsed_subject_results.csv", index=False
        )
        safe_collapsed.to_csv(
            staging / "runwise_loro_collapsed_pipeline_summary.csv", index=False
        )
        safe_pivot = safe_subject_collapsed.pivot(
            index=["subject", "analysis_kind"], columns="pipeline", values="accuracy"
        ).reset_index()
        for pipeline in ("glmsingle_typed_0mm", "glmsingle_typed_3mm"):
            safe_pivot[f"{pipeline}_minus_legacy"] = (
                safe_pivot[pipeline] - safe_pivot["legacy_lsa_8mm"]
            )
        safe_pivot.to_csv(
            staging / "runwise_loro_matched_pipeline_differences.csv", index=False
        )

        roi_subject = (
            pilot.groupby(["pipeline", "subject", "analysis_kind", "roi"], sort=False)["accuracy"]
            .mean()
            .reset_index()
        )
        roi_wide = roi_subject.pivot(
            index=["subject", "analysis_kind", "roi"], columns="pipeline", values="accuracy"
        ).reset_index()
        roi_delta_rows = []
        for pipeline in ("glmsingle_typed_0mm", "glmsingle_typed_3mm"):
            working = roi_wide.assign(delta=roi_wide[pipeline] - roi_wide["legacy_lsa_8mm"])
            summary = (
                working.groupby(["analysis_kind", "roi"], sort=False)["delta"]
                .agg(mean_delta="mean", sem_delta=sem)
                .reset_index()
            )
            summary.insert(0, "pipeline", pipeline)
            roi_delta_rows.append(summary)
        roi_delta = pd.concat(roi_delta_rows, ignore_index=True)
        roi_delta.to_csv(staging / "matched_roi_differences.csv", index=False)

        for pipeline in PIPELINE_ORDER:
            save_attached_style_plot(
                combined_summary,
                pipeline,
                staging / f"attached_style_{pipeline}.png",
            )
        save_kastner_pipeline_grid(
            combined_summary,
            PIPELINE_ORDER,
            staging / "kastner_roi_historical_avg_random.png",
            "Kastner/Wang ROI decoding — historical Avg(random)",
        )
        save_kastner_pipeline_grid(
            safe_summary,
            ("legacy_lsa_8mm", "glmsingle_typed_0mm", "glmsingle_typed_3mm"),
            staging / "kastner_roi_runwise_loro.png",
            "Kastner/Wang ROI decoding — run-wise LORO means",
        )
        save_collapsed_plot(collapsed, staging / "erp_style_collapsed_comparison.png")
        save_matched_delta_plot(pilot, staging / "erp_style_matched_roi_differences.png")
        save_collapsed_plot(
            safe_collapsed,
            staging / "runwise_loro_collapsed_comparison.png",
            pipelines=(
                "legacy_lsa_8mm",
                "glmsingle_typed_0mm",
                "glmsingle_typed_3mm",
            ),
            title="Run-wise ERP means with leakage-controlled LORO",
        )
        save_matched_delta_plot(
            safe_pilot, staging / "runwise_loro_matched_roi_differences.png"
        )
        write_report(
            staging,
            collapsed,
            pilot_collapsed,
            safe_collapsed,
            safe_subject_collapsed,
            reproduction,
            roi_delta,
        )

        provenance = {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "subjects": [f"Sub{value}" for value in args.subjects],
            "historical_cohort": [f"Sub{value}" for value in range(1, 32) if value not in {3, 8, 10}],
            "analysis_name": "historical ERP-style Avg(random)",
            "mode": MODE,
            "n_repeats": N_REPEATS,
            "n_folds": N_FOLDS,
            "n_training_averages_per_class": N_AVG_GROUPS,
            "classifier": "linear SVC C=1",
            "historical_compatibility_behavior": {
                "source_normalization_before_cv": True,
                "random_folds_block_runs": False,
                "random_folds_block_identity": False,
                "interpretation": "exact attached-figure replication; descriptive and not leakage-controlled",
            },
            "runwise_sensitivity": {
                "outer_validation": "leave one complete acquisition run out",
                "training_patterns_per_class": 9,
                "test_patterns_per_class": 1,
                "normalization_fit": "outer-training runs only",
                "source_transfer_normalization": "training-source parameters applied to test source",
                "stimulus_identity_held_out": False,
                "identity_caveat": "120 image identities repeat across the 10 acquisition runs",
            },
            "roi_resampling": "nearest-neighbor atlas-to-native-beta-grid, then intersect analysis mask",
            "smoothing": "3 mm mask-normalized Gaussian applied independently to each Type-D beta",
            "reproduction_check": reproduction,
            "row_counts": {
                "historical_subject_results": int(len(historical)),
                "pilot_subject_results": int(len(pilot)),
                "pilot_fold_results": int(len(folds)),
                "runwise_loro_subject_results": int(len(safe_pilot)),
                "runwise_loro_fold_results": int(len(safe_folds)),
                "roi_voxel_count_rows": int(len(counts)),
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "nibabel": nib.__version__,
                "scipy": scipy.__version__,
                "h5py": h5py.__version__,
                "scikit_learn": sklearn_version,
                "matplotlib": matplotlib.__version__,
            },
            "inputs": {
                "historical_result": fingerprint(args.historical_result),
                "historical_notebook": fingerprint(DEFAULT_HISTORICAL_NOTEBOOK),
                "atlas": fingerprint(args.atlas),
                "atlas_labels": fingerprint(args.atlas_labels),
                "legacy_groups": fingerprint(args.legacy_groups),
                "legacy_manifest": fingerprint(args.legacy_manifest),
                "branch_inputs": input_records,
            },
            "script": fingerprint(Path(__file__).resolve()),
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )

        required = {
            "REPORT.md",
            "provenance.json",
            "pilot_subject_results.csv",
            "pilot_fold_results.csv",
            "runwise_loro_subject_results.csv",
            "runwise_loro_fold_results.csv",
            "runwise_loro_summary.csv",
            "runwise_loro_collapsed_subject_results.csv",
            "runwise_loro_collapsed_pipeline_summary.csv",
            "runwise_loro_matched_pipeline_differences.csv",
            "historical_28sub_subject_results.csv",
            "historical_28sub_summary.csv",
            "pilot_summary.csv",
            "all_pipeline_roi_summary.csv",
            "collapsed_subject_results.csv",
            "collapsed_pipeline_summary.csv",
            "pilot_collapsed_subject_results.csv",
            "matched_pipeline_differences.csv",
            "matched_roi_differences.csv",
            "roi_voxel_counts.csv",
            "erp_style_collapsed_comparison.png",
            "erp_style_matched_roi_differences.png",
            "kastner_roi_historical_avg_random.png",
            "kastner_roi_runwise_loro.png",
            "runwise_loro_collapsed_comparison.png",
            "runwise_loro_matched_roi_differences.png",
        } | {f"attached_style_{pipeline}.png" for pipeline in PIPELINE_ORDER}
        observed = {path.name for path in staging.iterdir() if path.is_file()}
        missing = required - observed
        if missing:
            raise RuntimeError(f"Missing final outputs: {sorted(missing)}")
        os.replace(staging, target)
        print(f"COMPLETE: {target}", flush=True)
    except Exception:
        print(f"FAILED; preserved staging directory: {staging}", file=sys.stderr, flush=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--legacy-groups", type=Path, default=DEFAULT_LEGACY_GROUPS)
    parser.add_argument("--legacy-manifest", type=Path, default=DEFAULT_LEGACY_MANIFEST)
    parser.add_argument("--glmsingle-root", type=Path, default=DEFAULT_GLMSINGLE_ROOT)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--atlas-labels", type=Path, default=DEFAULT_ATLAS_LABELS)
    parser.add_argument("--historical-result", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if args.output is None:
        parser.error("--output is required unless --self-test is used")
    if args.subjects != [4, 5, 6]:
        raise ValueError("This matched pilot is locked to Subjects 4, 5 and 6")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        run_self_test()
    else:
        run(arguments)
