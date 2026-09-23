#!/usr/bin/env python3
"""Aggregate the Sub4/5/6 whole-brain decoding pilot into one safe report.

The report deliberately keeps three kinds of result distinct:

1. historical ERP-style maps made with the earlier random validation schemes;
2. the corrected legacy LS-A branch evaluated with leakage-safe validation; and
3. later production GLMsingle Type-D ``map_summary.csv`` results found under
   ``decoding_new_sdc``.

Outputs
-------
``side_by_side_long.csv``
    Tidy metric table (one row per metric).
``side_by_side_wide.csv``
    The same values pivoted so pipeline/validation variants are columns.
``branch_summary.csv``
    One row per subject/aggregate/variant/contrast with both requested metrics.
``historical_30sub_reference.csv``
    Exact spatial means of the original 30-subject group maps, checked against
    the values recorded when this pilot was designed.
``group_maps/...``
    Float32 pilot-subject mean and valid-count NIfTI images.  A group map is
    created only after all input subject maps in that variant are verified to
    have the same native grid.
``REPORT.md``
    A compact, human-readable comparison and interpretation guardrails.

Historical maps contain zero outside evaluated centers, whereas corrected and
new maps contain NaN there.  That distinction is respected when calculating
spatial means and valid-count maps.  Historical pooled whole-mask accuracy is
unavailable: it cannot be reconstructed from a searchlight accuracy map.

By default an existing output directory is never overwritten.  ``--refresh``
may be used for a derived report directory; only the known files/directories
created by this script are replaced atomically or removed.

``--require-complete-pilot`` is the fail-closed production gate.  It requires
the exact Sub4/5/6 legacy and 20-value default-grid SDC result, validates the
underlying fold tables and float32 accuracy maps, and rejects diagnostic or
failed source trees.  Without that flag, the original permissive behavior is
preserved so earlier partial/baseline reports remain reproducible.

Voxelwise new-minus-old maps are *not* made by default.  They require explicit
branch/smoothing choices plus ``--allow-approximate-resampling``.  The new
group mean is then linearly resampled to the old native grid, support is
nearest-neighbor resampled, and a machine-readable approximation manifest is
written beside the difference maps.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
PILOT_ROOT = CODE_DIR.parent
PROJECT_ROOT = PILOT_ROOT.parents[1]
DECODING_ROOT = PROJECT_ROOT / "Decoding"
PILOT_SUBJECTS = (4, 5, 6)

REQUIRED_SUMMARY_COLUMNS = {
    "subject",
    "branch",
    "additional_smoothing_fwhm_mm",
    "contrast",
    "cv_scheme",
    "pooled_whole_mask_accuracy",
    "spatial_mean_accuracy",
    "map_path",
}

# These are the exact positive-support spatial means of the original 30-subject
# group maps recorded before the corrected pilot was run.  The script also
# recalculates each value from the NIfTI and fails if it no longer matches.
HISTORICAL_30SUB_REFERENCE = {
    "within_natural_pleasant_vs_neutral": 0.5605681065339833,
    "within_ai_pleasant_vs_neutral": 0.44810243870321864,
    "within_natural_unpleasant_vs_neutral": 0.48676633098587,
    "within_ai_unpleasant_vs_neutral": 0.4457454536244119,
    "train_natural_pleasant_vs_neutral_test_ai": 0.5276692596863944,
    "train_ai_pleasant_vs_neutral_test_natural": 0.5334320700621565,
    "train_natural_unpleasant_vs_neutral_test_ai": 0.5326669434013671,
    "train_ai_unpleasant_vs_neutral_test_natural": 0.5341634134121164,
}

EXPECTED_CONTRASTS = tuple(HISTORICAL_30SUB_REFERENCE)
EXPECTED_CV_FOLDS = {
    "leave-one-run-out": 10,
    "identity-groupkfold": 5,
}
EXPECTED_BRANCH_SMOOTHING = {
    "legacy_lsa_8mm": (0.0,),
    "glmsingle_typed": (0.0, 3.0),
}
DEFAULT_GLMSINGLE_FRACS = tuple(round(1.0 - 0.05 * index, 2) for index in range(20))
FORBIDDEN_PRODUCTION_PATH_MARKERS = ("smoke", "coarsegrid", "failed")


@dataclass(frozen=True)
class HistoricalSpec:
    contrast: str
    root_name: str
    stem: str
    cv_scheme: str
    input_note: str


HISTORICAL_SPECS = (
    HistoricalSpec(
        "within_natural_pleasant_vs_neutral",
        "searchlight_within_source_erp",
        "pleasant_vs_neutral",
        "historical-random-stratified-5fold-repeated10",
        "Historical ERP-style image-identity averaging; SPM LS-A betas already smoothed 8 mm",
    ),
    HistoricalSpec(
        "within_ai_pleasant_vs_neutral",
        "searchlight_within_source_erp",
        "pleasantAI_vs_neutralAI",
        "historical-random-stratified-5fold-repeated10",
        "Historical ERP-style image-identity averaging; SPM LS-A betas already smoothed 8 mm",
    ),
    HistoricalSpec(
        "within_natural_unpleasant_vs_neutral",
        "searchlight_within_source_erp",
        "unpleasant_vs_neutral",
        "historical-random-stratified-5fold-repeated10",
        "Historical ERP-style image-identity averaging; SPM LS-A betas already smoothed 8 mm",
    ),
    HistoricalSpec(
        "within_ai_unpleasant_vs_neutral",
        "searchlight_within_source_erp",
        "unpleasantAI_vs_neutralAI",
        "historical-random-stratified-5fold-repeated10",
        "Historical ERP-style image-identity averaging; SPM LS-A betas already smoothed 8 mm",
    ),
    HistoricalSpec(
        "train_natural_pleasant_vs_neutral_test_ai",
        "searchlight_cross_source_revised",
        "train_pleasantvneutral_test_pleasantAIvneutralAI",
        "historical-random-chunk-split-repeated20",
        "Historical random ERP-style trial chunking; SPM LS-A betas already smoothed 8 mm",
    ),
    HistoricalSpec(
        "train_ai_pleasant_vs_neutral_test_natural",
        "searchlight_cross_source_revised",
        "train_pleasantAIvneutralAI_test_pleasantvneutral",
        "historical-random-chunk-split-repeated20",
        "Historical random ERP-style trial chunking; SPM LS-A betas already smoothed 8 mm",
    ),
    HistoricalSpec(
        "train_natural_unpleasant_vs_neutral_test_ai",
        "searchlight_cross_source_revised",
        "train_unpleasantvneutral_test_unpleasantAIvneutralAI",
        "historical-random-chunk-split-repeated20",
        "Historical random ERP-style trial chunking; SPM LS-A betas already smoothed 8 mm",
    ),
    HistoricalSpec(
        "train_ai_unpleasant_vs_neutral_test_natural",
        "searchlight_cross_source_revised",
        "train_unpleasantAIvneutralAI_test_unpleasantvneutral",
        "historical-random-chunk-split-repeated20",
        "Historical random ERP-style trial chunking; SPM LS-A betas already smoothed 8 mm",
    ),
)

KNOWN_DERIVED_NAMES = {
    "side_by_side_long.csv",
    "side_by_side_wide.csv",
    "branch_summary.csv",
    "historical_30sub_reference.csv",
    "group_map_manifest.csv",
    "difference_manifest.json",
    "REPORT.md",
    "provenance.json",
    "group_maps",
    "difference_maps",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_subject(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"Cannot parse subject from {value!r}")
    return int(match.group(1))


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def smoothing_token(value: object) -> str:
    number = safe_float(value)
    if not math.isfinite(number):
        return "unknown"
    return f"{number:g}mm"


def slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return text.strip("-.") or "unnamed"


def validation_status(branch: str, cv_scheme: str) -> str:
    if branch == "historical_erp_style_8mm":
        return "historical_random_validation_not_matched_to_corrected"
    if cv_scheme == "leave-one-run-out":
        return "corrected_leakage_safe_run_held_out"
    if cv_scheme == "identity-groupkfold":
        return "corrected_leakage_safe_identity_held_out"
    return "validation_as_reported_in_input_summary"


def variant_id(row: pd.Series | dict) -> str:
    return "__".join(
        (
            slug(row["branch"]),
            f"smooth-{smoothing_token(row['additional_smoothing_fwhm_mm'])}",
            slug(row["cv_scheme"]),
        )
    )


def windows_from_wsl(path_text: str) -> Path | None:
    match = re.match(r"^/mnt/([A-Za-z])/(.*)$", path_text)
    if match is None:
        return None
    tail = match.group(2).replace("/", os.sep)
    return Path(f"{match.group(1).upper()}:\\{tail}")


def resolve_map_path(path_text: object, source_summary: Path) -> Path:
    raw = str(path_text).strip()
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw))
        converted = windows_from_wsl(raw)
        if converted is not None:
            candidates.append(converted)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # A moved output tree can leave an obsolete absolute path in map_summary.
    # Searching only below the source summary's ancestors is safer than a
    # project-wide basename search and resolves the nested canonical output.
    basename = Path(raw).name
    if basename:
        for ancestor in (source_summary.parent, *source_summary.parents[:4]):
            matches = list(ancestor.rglob(basename))
            if len(matches) == 1:
                return matches[0].resolve()
            if len(matches) > 1:
                same_contrast = [item for item in matches if basename.startswith(str(item.name).split("_accuracy")[0])]
                if len(same_contrast) == 1:
                    return same_contrast[0].resolve()
    raise FileNotFoundError(f"Map from {source_summary} does not exist: {raw}")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f".{path.name}.", dir=path.parent,
        delete=False, encoding="utf-8", newline=""
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    temporary.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tmp", prefix=f".{path.name}.", dir=path.parent,
        delete=False, encoding="utf-8", newline="\n"
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def atomic_nifti(data: np.ndarray, reference: nib.spatialimages.SpatialImage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), reference.affine, header)
    # NIfTI compression is inferred from the final suffix, so the temporary
    # filename must also end with .nii.gz.
    with tempfile.NamedTemporaryFile(
        suffix=".nii.gz", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        nib.save(image, str(temporary))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_output(output: Path, refresh: bool) -> None:
    if not output.exists():
        output.mkdir(parents=True)
        return
    existing = list(output.iterdir())
    if existing and not refresh:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new path or pass --refresh."
        )
    if refresh:
        unknown = [item.name for item in existing if item.name not in KNOWN_DERIVED_NAMES]
        if unknown:
            raise RuntimeError(
                "Refusing --refresh because the output contains files not owned by this script: "
                + ", ".join(sorted(unknown))
            )
        for item in existing:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def map_valid_data(path: Path, historical: bool) -> tuple[nib.spatialimages.SpatialImage, np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    # Preserve the proxy's scaled precision for scalar summaries.  Conversion
    # to float32 happens only when writing the requested derived NIfTI maps.
    data = np.asanyarray(image.dataobj)
    valid = np.isfinite(data)
    if historical:
        # Historical maps used literal zero outside evaluated searchlights.
        valid &= data > 0
    return image, data, valid


def historical_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    reference_rows: list[dict] = []
    for spec in HISTORICAL_SPECS:
        root = DECODING_ROOT / spec.root_name
        group_path = root / f"{spec.stem}_mean.nii.gz"
        if not group_path.exists():
            raise FileNotFoundError(group_path)
        _, group_data, group_valid = map_valid_data(group_path, historical=True)
        actual = float(np.mean(group_data[group_valid], dtype=np.float64))
        expected = HISTORICAL_30SUB_REFERENCE[spec.contrast]
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-10):
            raise RuntimeError(
                f"Historical 30-sub reference changed for {spec.contrast}: "
                f"expected {expected:.15g}, found {actual:.15g} in {group_path}"
            )
        reference_rows.append(
            {
                "aggregation_level": "historical_30sub_group_map_reference",
                "subject": "historical_group_30",
                "n_subjects": 30,
                "branch": "historical_erp_style_8mm",
                "beta_kind": "historical_precomputed_accuracy_map",
                "input_smoothing_note": spec.input_note,
                "additional_smoothing_fwhm_mm": 0.0,
                "classifier": "historical_linear_svm",
                "searchlight_radius_mm": 5.0,
                "contrast": spec.contrast,
                "cv_scheme": spec.cv_scheme,
                "validation_status": validation_status("historical_erp_style_8mm", spec.cv_scheme),
                "pooled_whole_mask_accuracy": math.nan,
                "spatial_mean_accuracy": expected,
                "map_path": str(group_path.resolve()),
                "metric_availability_note": "Pooled whole-mask accuracy is not reconstructible from this map",
            }
        )
        for subject in PILOT_SUBJECTS:
            path = root / "per_subject" / f"{spec.stem}_Sub{subject}.nii.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            _, data, valid = map_valid_data(path, historical=True)
            if not np.any(valid):
                raise RuntimeError(f"No positive/evaluated searchlight values in {path}")
            rows.append(
                {
                    "aggregation_level": "subject",
                    "subject": f"Sub{subject}",
                    "subject_number": subject,
                    "n_subjects": 1,
                    "branch": "historical_erp_style_8mm",
                    "beta_kind": "historical_precomputed_accuracy_map",
                    "input_smoothing_note": spec.input_note,
                    "additional_smoothing_fwhm_mm": 0.0,
                    "classifier": "historical_linear_svm",
                    "searchlight_radius_mm": 5.0,
                    "contrast": spec.contrast,
                    "cv_scheme": spec.cv_scheme,
                    "validation_status": validation_status("historical_erp_style_8mm", spec.cv_scheme),
                    "pooled_whole_mask_accuracy": math.nan,
                    "spatial_mean_accuracy": float(np.mean(data[valid], dtype=np.float64)),
                    "map_path": str(path.resolve()),
                    "invalid_map_value_policy": "zero_is_outside_evaluated_centers",
                    "metric_availability_note": "Pooled whole-mask accuracy is not reconstructible from this map",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(reference_rows)


def discover_summary_files(roots: Sequence[Path]) -> tuple[list[Path], list[str]]:
    summaries: list[Path] = []
    warnings: list[str] = []
    for root in roots:
        if not root.exists():
            warnings.append(f"Summary root does not exist and was skipped: {root}")
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = list(root.rglob("map_summary.csv"))
        for path in candidates:
            try:
                columns = set(pd.read_csv(path, nrows=0).columns)
            except Exception as error:  # pragma: no cover - defensive I/O report
                warnings.append(f"Unreadable map summary skipped ({path}): {error}")
                continue
            if REQUIRED_SUMMARY_COLUMNS.issubset(columns):
                summaries.append(path.resolve())
            else:
                warnings.append(f"Non-pilot map summary skipped (missing columns): {path}")
    return sorted(set(summaries)), warnings


def corrected_and_new_rows(roots: Sequence[Path]) -> tuple[pd.DataFrame, list[Path], list[str]]:
    summaries, warnings = discover_summary_files(roots)
    frames: list[pd.DataFrame] = []
    for path in summaries:
        frame = pd.read_csv(path)
        frame = frame.copy()
        frame["_source_summary"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), summaries, warnings

    result = pd.concat(frames, ignore_index=True, sort=False)
    result["subject_number"] = result["subject"].map(parse_subject)
    result = result[result["subject_number"].isin(PILOT_SUBJECTS)].copy()
    result["subject"] = result["subject_number"].map(lambda item: f"Sub{item}")
    result["additional_smoothing_fwhm_mm"] = pd.to_numeric(
        result["additional_smoothing_fwhm_mm"], errors="coerce"
    )

    resolved_paths: list[str] = []
    for _, row in result.iterrows():
        source = Path(row["_source_summary"])
        resolved_paths.append(str(resolve_map_path(row["map_path"], source)))
    result["map_path"] = resolved_paths

    dedup_keys = [
        "subject_number", "branch", "additional_smoothing_fwhm_mm",
        "contrast", "cv_scheme", "map_path",
    ]
    result = result.sort_values("_source_summary").drop_duplicates(dedup_keys, keep="first")
    core_keys = [
        "subject_number", "branch", "additional_smoothing_fwhm_mm",
        "contrast", "cv_scheme",
    ]
    core_counts = result.groupby(core_keys, dropna=False).size()
    ambiguous = core_counts[core_counts > 1]
    if len(ambiguous):
        # It is common for a later ``--only both`` decoder run to contain a
        # redundant legacy result.  Prefer the dedicated corrected-legacy tree
        # for that branch, but never guess between multiple GLMsingle runs.
        glmsingle_ambiguous = [key for key in ambiguous.index if key[1] != "legacy_lsa_8mm"]
        if glmsingle_ambiguous:
            examples = "; ".join(str(item) for item in glmsingle_ambiguous[:5])
            raise RuntimeError(
                "Ambiguous duplicate non-legacy results were discovered. Pass explicit "
                f"--summary-root path(s) to select one run. Examples: {examples}"
            )
        result["_legacy_preference"] = result["_source_summary"].map(
            lambda item: 0 if "decoding_corrected_legacy" in str(item) else 1
        )
        before = len(result)
        result = result.sort_values(["_legacy_preference", "_source_summary"]).drop_duplicates(
            core_keys, keep="first"
        )
        removed = before - len(result)
        warnings.append(
            f"Skipped {removed} redundant legacy summary row(s); preferred decoding_corrected_legacy"
        )
        result = result.drop(columns="_legacy_preference")
    result["aggregation_level"] = "subject"
    result["n_subjects"] = 1
    result["validation_status"] = [
        validation_status(branch, cv)
        for branch, cv in zip(result["branch"], result["cv_scheme"])
    ]
    result["invalid_map_value_policy"] = "nan_is_outside_evaluated_centers"
    result["metric_availability_note"] = "Both requested metrics reported by leakage-safe decoder"
    return result, summaries, warnings


def _production_key(row: pd.Series | dict, include_source: bool = False) -> tuple:
    key = (
        int(row["subject_number"]),
        str(row["branch"]),
        float(row["additional_smoothing_fwhm_mm"]),
        str(row["contrast"]),
        str(row["cv_scheme"]),
    )
    if include_source:
        return key + (str(row["_source_summary"]),)
    return key


def _expected_production_keys() -> Counter:
    return Counter(
        (
            subject,
            branch,
            smoothing,
            contrast,
            cv_scheme,
        )
        for subject in PILOT_SUBJECTS
        for branch, smoothings in EXPECTED_BRANCH_SMOOTHING.items()
        for smoothing in smoothings
        for contrast in EXPECTED_CONTRASTS
        for cv_scheme in EXPECTED_CV_FOLDS
    )


def _counter_examples(counter: Counter, limit: int = 5) -> str:
    return "; ".join(f"{key} x{count}" for key, count in list(counter.items())[:limit])


def _reject_forbidden_paths(paths: Iterable[object]) -> None:
    rejected: list[str] = []
    for value in paths:
        text = str(value).strip()
        lowered = text.replace("\\", "/").lower()
        if text and any(marker in lowered for marker in FORBIDDEN_PRODUCTION_PATH_MARKERS):
            rejected.append(text)
    if rejected:
        raise RuntimeError(
            "Production report source paths contain a forbidden diagnostic/failed marker "
            f"({', '.join(FORBIDDEN_PRODUCTION_PATH_MARKERS)}): " + "; ".join(rejected[:5])
        )


def _load_decoder_provenance(source_summary: Path) -> tuple[Path, dict]:
    # ``decode_compare.py`` writes one provenance file at the subject output
    # root, while also writing per-variant summaries below
    # ``branch/smoothing/cv``.  Discovery can legitimately retain either the
    # root or a nested summary, so resolve provenance upward through the known
    # decoder layout instead of assuming it is a sibling of every summary.
    candidates: list[Path] = []
    path: Path | None = None
    for depth, directory in enumerate(source_summary.parents):
        if depth >= 5:
            break
        candidate = directory / "provenance.json"
        candidates.append(candidate)
        if candidate.exists():
            path = candidate
            break
    if path is None:
        path = candidates[-1] if candidates else source_summary.with_name("provenance.json")

    if not path.exists():
        searched = "; ".join(str(item) for item in candidates)
        raise FileNotFoundError(
            "Decoder provenance is required by --require-complete-pilot; "
            f"searched from the retained summary toward its subject root: {searched}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"Cannot read decoder provenance {path}: {error}") from error
    if payload.get("algorithm", {}).get("map_outside_centers") != "NaN":
        raise RuntimeError(f"Decoder provenance does not declare NaN outside centers: {path}")
    return path, payload


def _branch_provenance(payload: dict, branch: str, provenance_path: Path) -> dict:
    matches = [item for item in payload.get("branches", []) if item.get("name") == branch]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {branch} branch in decoder provenance {provenance_path}; found {len(matches)}"
        )
    return matches[0]


def _validate_default_grid_glmsingle(
    source_summary: Path,
    branch_metadata: dict,
) -> None:
    provenance_paths = [
        branch_metadata.get("beta_path", ""),
        branch_metadata.get("mask_path", ""),
        branch_metadata.get("manifest_path", ""),
    ]
    _reject_forbidden_paths(provenance_paths)
    beta_path = resolve_map_path(branch_metadata.get("beta_path", ""), source_summary)
    if "glmsingle_sdc_defaultgrid" not in str(beta_path).replace("\\", "/").lower():
        raise RuntimeError(
            "The production glmsingle_typed beta must come from the designated "
            f"glmsingle_sdc_defaultgrid tree, not {beta_path}"
        )
    upstream_path = beta_path.parent.parent / "provenance.json"
    if not upstream_path.exists():
        raise FileNotFoundError(f"GLMsingle provenance is required: {upstream_path}")
    try:
        upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"Cannot read GLMsingle provenance {upstream_path}: {error}") from error
    fracs = np.asarray(upstream.get("fracs", []), dtype=np.float64)
    expected = np.asarray(DEFAULT_GLMSINGLE_FRACS, dtype=np.float64)
    if fracs.shape != expected.shape or not np.allclose(fracs, expected, rtol=0.0, atol=1e-6):
        raise RuntimeError(
            f"GLMsingle result is not the 20-value default ridge grid in {upstream_path}: "
            f"found {fracs.tolist()}"
        )
    _reject_forbidden_paths(
        [
            upstream_path,
            upstream.get("fmriprep_root", ""),
            upstream.get("bids_root", ""),
            *upstream.get("bold_files", []),
        ]
    )


def _validate_accuracy_map(
    row: pd.Series,
    source_summary: Path,
    branch_metadata: dict,
    mask_cache: dict[Path, tuple[nib.spatialimages.SpatialImage, np.ndarray]],
    mask_resolution_cache: dict[str, Path] | None = None,
) -> None:
    map_path = Path(row["map_path"])
    image = nib.load(str(map_path))
    if np.dtype(image.get_data_dtype()) != np.dtype(np.float32):
        raise RuntimeError(f"Production accuracy map is not float32: {map_path}")

    raw_mask_path = str(branch_metadata.get("mask_path", ""))
    if mask_resolution_cache is not None and raw_mask_path in mask_resolution_cache:
        mask_path = mask_resolution_cache[raw_mask_path]
    else:
        mask_path = resolve_map_path(raw_mask_path, source_summary)
        if mask_resolution_cache is not None:
            mask_resolution_cache[raw_mask_path] = mask_path
    if mask_path not in mask_cache:
        mask_image = nib.load(str(mask_path))
        mask_data = np.asanyarray(mask_image.dataobj)
        mask_cache[mask_path] = (mask_image, np.isfinite(mask_data) & (mask_data != 0))
    mask_image, mask = mask_cache[mask_path]
    if not same_grid(image, mask_image):
        raise RuntimeError(f"Accuracy map and analysis mask have different grids: {map_path}, {mask_path}")

    data = np.asanyarray(image.dataobj)
    finite = np.isfinite(data)
    if np.any(np.isinf(data)):
        raise RuntimeError(f"Production accuracy map contains infinity: {map_path}")
    if np.any(finite & ~mask):
        raise RuntimeError(f"Production accuracy map has finite values outside its analysis mask: {map_path}")
    if np.any(~np.isnan(data[~mask])):
        raise RuntimeError(f"Production accuracy map is not NaN outside evaluated mask centers: {map_path}")
    values = data[finite]
    if values.size == 0 or np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError(f"Finite accuracy values must be nonempty and within [0, 1]: {map_path}")

    finite_count = int(finite.sum())
    expected_finite = int(row["n_centers_valid"])
    mask_count = int(mask.sum())
    if finite_count != expected_finite:
        raise RuntimeError(
            f"Finite voxel count differs from n_centers_valid for {map_path}: "
            f"{finite_count} versus {expected_finite}"
        )
    if int(row["n_input_voxels"]) != mask_count:
        raise RuntimeError(
            f"Analysis-mask voxel count differs from n_input_voxels for {map_path}: "
            f"{mask_count} versus {row['n_input_voxels']}"
        )
    if int(row["n_searchlight_centers"]) != mask_count:
        raise RuntimeError(
            f"Expected every in-mask voxel to be a searchlight center for {map_path}; "
            f"found {row['n_searchlight_centers']} centers for {mask_count} mask voxels"
        )


def validate_complete_pilot(
    decoded: pd.DataFrame,
    summaries: Sequence[Path],
    summary_roots: Sequence[Path],
) -> dict:
    """Fail closed unless the complete matched Sub4/5/6 production result is present."""

    if decoded.empty:
        raise RuntimeError("--require-complete-pilot requires corrected legacy and GLMsingle rows")
    expected = _expected_production_keys()
    actual = Counter(_production_key(row) for _, row in decoded.iterrows())
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise RuntimeError(
            "Incomplete or unexpected production map_summary rows. "
            f"Expected {sum(expected.values())}; found {sum(actual.values())}. "
            f"Missing: {_counter_examples(missing) or 'none'}. "
            f"Extra/duplicate: {_counter_examples(extra) or 'none'}."
        )

    _reject_forbidden_paths(
        [
            *summary_roots,
            *summaries,
            *decoded["_source_summary"].tolist(),
            *decoded["map_path"].tolist(),
        ]
    )

    metric_columns = [
        "mean_fold_whole_mask_accuracy",
        "pooled_whole_mask_accuracy",
        "spatial_mean_accuracy",
    ]
    for column in metric_columns:
        if column not in decoded:
            raise RuntimeError(f"Complete pilot map summaries require column: {column}")
        values = pd.to_numeric(decoded[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values) & (values >= 0.0) & (values <= 1.0)):
            raise RuntimeError(f"Complete pilot map-summary values are not all finite/[0,1]: {column}")
    for _, row in decoded.iterrows():
        expected_folds = EXPECTED_CV_FOLDS[str(row["cv_scheme"])]
        if int(row["n_folds"]) != expected_folds:
            raise RuntimeError(
                f"Wrong n_folds for {_production_key(row)}: {row['n_folds']} versus {expected_folds}"
            )

    source_paths = sorted({Path(item) for item in decoded["_source_summary"]})
    fold_frames: list[pd.DataFrame] = []
    for source in source_paths:
        fold_path = source.with_name("fold_summary.csv")
        if not fold_path.exists():
            raise FileNotFoundError(f"Fold summary is required by --require-complete-pilot: {fold_path}")
        frame = pd.read_csv(fold_path)
        required = {
            "subject", "branch", "additional_smoothing_fwhm_mm", "contrast",
            "cv_scheme", "fold_id", "whole_mask_accuracy", "spatial_mean_accuracy",
        }
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise RuntimeError(f"Fold summary {fold_path} lacks columns: {sorted(missing_columns)}")
        frame = frame.copy()
        frame["subject_number"] = frame["subject"].map(parse_subject)
        frame["additional_smoothing_fwhm_mm"] = pd.to_numeric(
            frame["additional_smoothing_fwhm_mm"], errors="coerce"
        )
        frame["_source_summary"] = str(source)
        fold_frames.append(frame)

    folds = pd.concat(fold_frames, ignore_index=True, sort=False)
    expected_fold_counts = Counter()
    for _, row in decoded.iterrows():
        expected_fold_counts[_production_key(row, include_source=True)] = EXPECTED_CV_FOLDS[
            str(row["cv_scheme"])
        ]
    actual_fold_counts = Counter(_production_key(row, include_source=True) for _, row in folds.iterrows())
    if actual_fold_counts != expected_fold_counts:
        missing = expected_fold_counts - actual_fold_counts
        extra = actual_fold_counts - expected_fold_counts
        raise RuntimeError(
            "Fold-summary counts do not match the 144 expected map rows. "
            f"Expected {sum(expected_fold_counts.values())}; found {sum(actual_fold_counts.values())}. "
            f"Missing: {_counter_examples(missing) or 'none'}. "
            f"Extra/duplicate: {_counter_examples(extra) or 'none'}."
        )
    for key_values, group in folds.groupby(
        [
            "subject_number", "branch", "additional_smoothing_fwhm_mm", "contrast",
            "cv_scheme", "_source_summary",
        ],
        dropna=False,
    ):
        if group["fold_id"].nunique(dropna=False) != len(group):
            raise RuntimeError(f"Duplicate fold_id values for {key_values}")
    for column in ("whole_mask_accuracy", "spatial_mean_accuracy"):
        values = pd.to_numeric(folds[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values) & (values >= 0.0) & (values <= 1.0)):
            raise RuntimeError(f"Fold-summary values are not all finite/[0,1]: {column}")

    decoder_cache: dict[Path, tuple[Path, dict]] = {}
    branch_cache: dict[tuple[Path, str], dict] = {}
    mask_cache: dict[Path, tuple[nib.spatialimages.SpatialImage, np.ndarray]] = {}
    mask_resolution_cache: dict[str, Path] = {}
    validated_glmsingle_sources: set[Path] = set()
    for source in source_paths:
        decoder_cache[source] = _load_decoder_provenance(source)
    for _, row in decoded.iterrows():
        source = Path(row["_source_summary"])
        provenance_path, payload = decoder_cache[source]
        branch_key = (source, str(row["branch"]))
        if branch_key not in branch_cache:
            metadata = _branch_provenance(payload, branch_key[1], provenance_path)
            _reject_forbidden_paths(
                [
                    metadata.get("beta_path", ""),
                    metadata.get("mask_path", ""),
                    metadata.get("manifest_path", ""),
                ]
            )
            branch_cache[branch_key] = metadata
        metadata = branch_cache[branch_key]
        if row["branch"] == "glmsingle_typed" and source not in validated_glmsingle_sources:
            _validate_default_grid_glmsingle(source, metadata)
            validated_glmsingle_sources.add(source)
        _validate_accuracy_map(row, source, metadata, mask_cache, mask_resolution_cache)

    for source in source_paths:
        listed = {
            os.path.normcase(str(Path(item).resolve()))
            for item in decoded.loc[decoded["_source_summary"] == str(source), "map_path"]
        }
        on_disk = {
            os.path.normcase(str(item.resolve()))
            for item in source.parent.rglob("*_accuracy.nii.gz")
        }
        if on_disk != listed:
            raise RuntimeError(
                f"Accuracy-map file count/content differs from map_summary in {source.parent}: "
                f"listed {len(listed)}, on disk {len(on_disk)}"
            )

    return {
        "status": "passed",
        "subjects": list(PILOT_SUBJECTS),
        "map_summary_rows": int(len(decoded)),
        "fold_summary_rows": int(len(folds)),
        "accuracy_maps": int(decoded["map_path"].nunique()),
        "legacy_rows_per_subject": 16,
        "glmsingle_rows_per_subject": 32,
        "legacy_folds_per_subject": 120,
        "glmsingle_folds_per_subject": 240,
        "glmsingle_grid": list(DEFAULT_GLMSINGLE_FRACS),
    }


def metadata_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "branch", "beta_kind", "input_smoothing_note",
        "additional_smoothing_fwhm_mm", "classifier", "searchlight_radius_mm",
        "contrast", "cv_scheme", "validation_status",
    ]
    return [column for column in preferred if column in frame.columns]


def add_pilot_means(subject_rows: pd.DataFrame) -> pd.DataFrame:
    if subject_rows.empty:
        return subject_rows.copy()
    keys = metadata_columns(subject_rows)
    aggregate_rows: list[dict] = []
    for key_values, group in subject_rows.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(keys, key_values))
        subjects = sorted(set(group["subject_number"].dropna().astype(int)))
        row.update(
            {
                "aggregation_level": "pilot_mean",
                "subject": "pilot_mean_Sub4-6",
                "subject_number": math.nan,
                "n_subjects": len(subjects),
                "expected_subjects": ";".join(str(item) for item in PILOT_SUBJECTS),
                "included_subjects": ";".join(str(item) for item in subjects),
                "complete_three_subject_pilot": subjects == list(PILOT_SUBJECTS),
                "map_path": "",
                "invalid_map_value_policy": "not_applicable_scalar_aggregate",
                "metric_availability_note": group["metric_availability_note"].iloc[0]
                if "metric_availability_note" in group else "",
            }
        )
        for metric in ("pooled_whole_mask_accuracy", "spatial_mean_accuracy"):
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = values[np.isfinite(values)]
            row[metric] = float(valid.mean()) if len(valid) else math.nan
            row[f"n_subjects_with_{metric}"] = int(len(valid))
        aggregate_rows.append(row)
    return pd.DataFrame(aggregate_rows)


def to_long(branch_summary: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "aggregation_level", "subject", "subject_number", "n_subjects",
        "branch", "beta_kind", "input_smoothing_note",
        "additional_smoothing_fwhm_mm", "classifier", "searchlight_radius_mm",
        "contrast", "cv_scheme", "validation_status", "variant_id",
        "complete_three_subject_pilot", "map_path", "metric_availability_note",
    ]
    for column in id_columns:
        if column not in branch_summary:
            branch_summary[column] = math.nan
    long = branch_summary.melt(
        id_vars=id_columns,
        value_vars=["pooled_whole_mask_accuracy", "spatial_mean_accuracy"],
        var_name="metric",
        value_name="accuracy",
    )
    long["metric_available"] = np.isfinite(pd.to_numeric(long["accuracy"], errors="coerce"))
    return long.sort_values(
        ["aggregation_level", "subject", "contrast", "metric", "variant_id"],
        na_position="last",
    ).reset_index(drop=True)


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    index = ["aggregation_level", "subject", "contrast", "metric"]
    # ``pivot_table(dropna=False)`` constructs a Cartesian product of the index
    # levels.  Plain pivot retains only combinations that actually occur,
    # including observed historical pooled-metric rows whose value is blank.
    wide = long.pivot(index=index, columns="variant_id", values="accuracy").reset_index()
    wide.columns.name = None
    return wide


def same_grid(first: nib.spatialimages.SpatialImage, other: nib.spatialimages.SpatialImage) -> bool:
    return first.shape == other.shape and np.allclose(first.affine, other.affine, rtol=0.0, atol=1e-5)


def create_group_maps(subject_rows: pd.DataFrame, output: Path) -> pd.DataFrame:
    keys = [
        "branch", "additional_smoothing_fwhm_mm", "cv_scheme", "contrast",
    ]
    records: list[dict] = []
    for key_values, group in subject_rows.groupby(keys, dropna=False, sort=True):
        branch, smoothing, cv_scheme, contrast = key_values
        group = group.sort_values("subject_number")
        images: list[nib.spatialimages.SpatialImage] = []
        arrays: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        source_paths: list[str] = []
        for _, row in group.iterrows():
            path = Path(row["map_path"])
            historical = row["branch"] == "historical_erp_style_8mm"
            image, data, valid = map_valid_data(path, historical=historical)
            if images and not same_grid(images[0], image):
                raise RuntimeError(
                    "Cannot calculate a native-grid group mean because subject maps differ: "
                    f"{source_paths[0]} versus {path}"
                )
            images.append(image)
            arrays.append(data)
            valids.append(valid)
            source_paths.append(str(path))
        if not images:
            continue
        values = np.stack(arrays, axis=0)
        support = np.stack(valids, axis=0)
        valid_count = support.sum(axis=0, dtype=np.int16)
        numerator = np.where(support, values, 0.0).sum(axis=0, dtype=np.float64)
        mean = np.full(valid_count.shape, np.nan, dtype=np.float32)
        np.divide(numerator, valid_count, out=mean, where=valid_count > 0)

        relative = (
            Path("group_maps") / slug(branch) /
            f"additional_smoothing-{smoothing_token(smoothing)}" /
            slug(cv_scheme)
        )
        mean_path = output / relative / f"{slug(contrast)}_pilot-Sub4-6_mean_accuracy.nii.gz"
        count_path = output / relative / f"{slug(contrast)}_pilot-Sub4-6_valid_count.nii.gz"
        atomic_nifti(mean, images[0], mean_path)
        atomic_nifti(valid_count.astype(np.float32), images[0], count_path)
        subjects = sorted(set(group["subject_number"].astype(int)))
        records.append(
            {
                "branch": branch,
                "additional_smoothing_fwhm_mm": smoothing,
                "cv_scheme": cv_scheme,
                "contrast": contrast,
                "n_subjects": len(subjects),
                "included_subjects": ";".join(str(item) for item in subjects),
                "complete_three_subject_pilot": subjects == list(PILOT_SUBJECTS),
                "native_shape": "x".join(str(item) for item in images[0].shape),
                "native_affine_json": json.dumps(images[0].affine.tolist(), separators=(",", ":")),
                "group_mean_map": str(mean_path.resolve()),
                "valid_count_map": str(count_path.resolve()),
                "source_maps_json": json.dumps(source_paths),
            }
        )
    return pd.DataFrame(records)


def select_group_map(
    manifest: pd.DataFrame,
    branch: str,
    smoothing: float,
    cv_scheme: str,
    contrast: str,
) -> pd.Series:
    selected = manifest[
        (manifest["branch"] == branch)
        & np.isclose(
            pd.to_numeric(manifest["additional_smoothing_fwhm_mm"], errors="coerce"),
            smoothing,
            rtol=0.0,
            atol=1e-9,
        )
        & (manifest["cv_scheme"] == cv_scheme)
        & (manifest["contrast"] == contrast)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one group map for branch={branch}, smoothing={smoothing}, "
            f"cv={cv_scheme}, contrast={contrast}; found {len(selected)}"
        )
    return selected.iloc[0]


def create_differences(args: argparse.Namespace, manifest: pd.DataFrame, output: Path) -> list[dict]:
    requested = args.difference_new_branch is not None or args.difference_old_branch is not None
    if not requested:
        return []
    if args.difference_new_branch is None or args.difference_old_branch is None:
        raise ValueError("Both --difference-new-branch and --difference-old-branch are required")
    if not args.allow_approximate_resampling:
        raise ValueError(
            "Difference maps require --allow-approximate-resampling to acknowledge linear "
            "resampling of accuracy maps"
        )
    contrasts = sorted(
        set(manifest.loc[manifest["branch"] == args.difference_new_branch, "contrast"])
        & set(manifest.loc[manifest["branch"] == args.difference_old_branch, "contrast"])
    )
    if not contrasts:
        raise RuntimeError("No common contrasts found for the requested difference branches")
    records: list[dict] = []
    for contrast in contrasts:
        new = select_group_map(
            manifest, args.difference_new_branch, args.difference_new_smoothing,
            args.difference_cv, contrast,
        )
        old = select_group_map(
            manifest, args.difference_old_branch, args.difference_old_smoothing,
            args.difference_cv, contrast,
        )
        if not bool(new["complete_three_subject_pilot"]) or not bool(old["complete_three_subject_pilot"]):
            raise RuntimeError(f"Difference requested for incomplete three-subject maps: {contrast}")
        new_image = nib.load(new["group_mean_map"])
        old_image = nib.load(old["group_mean_map"])
        new_count_image = nib.load(new["valid_count_map"])
        old_count_image = nib.load(old["valid_count_map"])
        target = (old_image.shape, old_image.affine)
        # This call is intentional even when two grids happen to match.  The
        # output provenance therefore has one explicit, invariant operation.
        resampled_new = resample_from_to(new_image, target, order=1, mode="constant", cval=np.nan)
        resampled_new_count = resample_from_to(
            new_count_image, target, order=0, mode="constant", cval=0.0
        )
        old_data = np.asanyarray(old_image.dataobj).astype(np.float32, copy=False)
        new_data = np.asanyarray(resampled_new.dataobj).astype(np.float32, copy=False)
        old_count = np.asanyarray(old_count_image.dataobj)
        new_count = np.asanyarray(resampled_new_count.dataobj)
        overlap = np.isfinite(old_data) & np.isfinite(new_data) & (old_count > 0) & (new_count > 0)
        difference = np.full(old_image.shape, np.nan, dtype=np.float32)
        difference[overlap] = new_data[overlap] - old_data[overlap]
        overlap_count = np.where(overlap, np.minimum(old_count, new_count), 0).astype(np.float32)

        relative = Path("difference_maps") / (
            f"{slug(args.difference_new_branch)}_minus_{slug(args.difference_old_branch)}"
        ) / slug(args.difference_cv)
        base = (
            f"{slug(contrast)}_new-smooth-{smoothing_token(args.difference_new_smoothing)}"
            f"_old-smooth-{smoothing_token(args.difference_old_smoothing)}"
        )
        difference_path = output / relative / f"{base}_difference_accuracy.nii.gz"
        overlap_path = output / relative / f"{base}_overlap_valid_count.nii.gz"
        atomic_nifti(difference, old_image, difference_path)
        atomic_nifti(overlap_count, old_image, overlap_path)
        records.append(
            {
                "contrast": contrast,
                "operation": "new_group_mean_minus_old_group_mean",
                "new_branch": args.difference_new_branch,
                "old_branch": args.difference_old_branch,
                "new_smoothing_fwhm_mm": args.difference_new_smoothing,
                "old_smoothing_fwhm_mm": args.difference_old_smoothing,
                "cv_scheme": args.difference_cv,
                "target_grid": "old_branch_native_grid",
                "accuracy_interpolation": "continuous_linear_order_1",
                "support_interpolation": "nearest_neighbor_order_0",
                "interpretation": (
                    "Descriptive approximation only; interpolation changes searchlight-map samples "
                    "and is not a statistical comparison"
                ),
                "new_group_mean_map": new["group_mean_map"],
                "old_group_mean_map": old["group_mean_map"],
                "difference_map": str(difference_path.resolve()),
                "overlap_valid_count_map": str(overlap_path.resolve()),
            }
        )
    return records


def markdown_table(frame: pd.DataFrame, float_digits: int = 3) -> str:
    if frame.empty:
        return "_No rows available._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}"
            )
        else:
            display[column] = display[column].fillna("").astype(str)
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in display.iterrows():
        values = [str(row[column]).replace("|", "\\|").replace("\n", " ") for column in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report_markdown(
    branch_summary: pd.DataFrame,
    historical_reference: pd.DataFrame,
    group_manifest: pd.DataFrame,
    warnings: Sequence[str],
    difference_records: Sequence[dict],
    generated_at: str,
    require_complete_pilot: bool = False,
) -> str:
    pilot = branch_summary[branch_summary["aggregation_level"] == "pilot_mean"].copy()
    pilot_long = to_long(pilot)
    pilot_spatial = pilot_long[pilot_long["metric"] == "spatial_mean_accuracy"].pivot_table(
        index="contrast", columns="variant_id", values="accuracy", aggfunc="first"
    ).reset_index()
    pilot_pooled = pilot_long[pilot_long["metric"] == "pooled_whole_mask_accuracy"].pivot_table(
        index="contrast", columns="variant_id", values="accuracy", aggfunc="first"
    ).reset_index()
    reference_display = historical_reference[["contrast", "spatial_mean_accuracy"]].rename(
        columns={"spatial_mean_accuracy": "historical_30sub_spatial_mean_accuracy"}
    )
    complete_maps = int(group_manifest["complete_three_subject_pilot"].sum()) if not group_manifest.empty else 0
    all_maps = len(group_manifest)

    lines = [
        "# Whole-brain decoding pilot: Sub4, Sub5, and Sub6",
        "",
        f"Generated: `{generated_at}`",
        "",
        "This report puts the old and new outputs in one table while preserving a crucial distinction: "
        "the historical ERP-style results used random validation, whereas the corrected legacy and "
        "GLMsingle branches use held-out runs or held-out stimulus identities. Historical values are "
        "context, not a validation-matched estimate of improvement.",
    ]
    if require_complete_pilot:
        lines.extend(
            [
                "",
                "In this strict production report, `glmsingle_typed` explicitly denotes the "
                "susceptibility-distortion-corrected (SDC) pipeline using GLMsingle Type-D "
                "single-trial betas and the 20-value default fractional-ridge grid (`1.00` to "
                "`0.05` in `0.05` steps). The completeness gate verified all expected Sub4/5/6 "
                "rows, folds, source maps, and upstream ridge-grid provenance before reporting.",
            ]
        )
    lines.extend(
        [
        "",
        "`pooled_whole_mask_accuracy` is the accuracy of one classifier using the full analysis mask. "
        "`spatial_mean_accuracy` is the mean of local searchlight accuracies. They answer different "
        "questions and should not be compared to each other as if they were the same statistic.",
        "",
        "## Three-subject mean: pooled whole-mask accuracy",
        "",
        markdown_table(pilot_pooled),
        "",
        "Historical entries are blank here because pooled whole-mask accuracy cannot be reconstructed "
        "from an already-averaged searchlight accuracy map.",
        "",
        "## Three-subject mean: spatial mean accuracy",
        "",
        markdown_table(pilot_spatial),
        "",
        "## Original 30-subject historical group-map references",
        "",
        markdown_table(reference_display, float_digits=6),
        "",
        "These eight values are positive-support means of the original group NIfTI maps and are "
        "verified against fixed exact reference values on every report run. The original cohort and "
        "random validation differ from the current three-subject pilot.",
        "",
        "## Brain maps",
        "",
        f"Created {all_maps} native-grid group mean/count pairs; {complete_maps} include all three pilot subjects. "
        "Mean and valid-count images are float32. No cross-grid averaging occurs.",
        ]
    )
    if difference_records:
        lines.extend(
            [
                "",
                "## Approximate voxelwise differences",
                "",
                f"Created {len(difference_records)} explicitly requested new-minus-old maps. The new mean was "
                "linearly resampled to the old grid and support was nearest-neighbor resampled. These are "
                "descriptive approximations, not statistical tests; see `difference_manifest.json`.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "No voxelwise new-minus-old map was created. Such subtraction is disabled unless explicit "
                "branch/smoothing choices and `--allow-approximate-resampling` are supplied.",
            ]
        )
    lines.extend(
        [
            "",
            "With only three subjects, these results are a pipeline screen rather than group-level inference.",
        ]
    )
    if warnings:
        lines.extend(["", "## Input warnings", ""] + [f"- {item}" for item in warnings])
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        action="append",
        help=(
            "Root/file to scan for decoder map_summary.csv files; repeatable. "
            "By default only decoding_corrected_legacy and decoding_new_sdc are scanned, "
            "so diagnostic smoke outputs cannot enter the scientific report."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Derived report directory (default: pilot/reports/report-<UTC timestamp>)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace only known script-owned artifacts in an existing output directory",
    )
    parser.add_argument(
        "--require-complete-pilot",
        action="store_true",
        help=(
            "Fail unless the exact Sub4/5/6 legacy and 20-value default-grid SDC result is present, "
            "with expected rows/folds/maps and strict NIfTI integrity"
        ),
    )
    parser.add_argument("--difference-new-branch", help="Explicit new branch for optional difference maps")
    parser.add_argument("--difference-old-branch", help="Explicit old branch for optional difference maps")
    parser.add_argument("--difference-new-smoothing", type=float, default=0.0)
    parser.add_argument("--difference-old-smoothing", type=float, default=0.0)
    parser.add_argument("--difference-cv", default="leave-one-run-out")
    parser.add_argument(
        "--allow-approximate-resampling",
        action="store_true",
        help="Required acknowledgement before any cross-grid voxelwise subtraction",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or PILOT_ROOT / "reports" / f"report-{timestamp}"
    output = output.resolve()
    # Preserve the original preparation order when the opt-in strict gate is
    # absent.  In strict mode, validate every source before touching output.
    if not args.require_complete_pilot:
        prepare_output(output, refresh=args.refresh)

    historical, historical_reference = historical_rows()
    summary_roots = (
        [item.resolve() for item in args.summary_root]
        if args.summary_root
        else [
            PILOT_ROOT / "decoding_corrected_legacy",
            PILOT_ROOT / "decoding_new_sdc",
        ]
    )
    decoded, summaries, warnings = corrected_and_new_rows(summary_roots)
    complete_validation: dict | None = None
    if args.require_complete_pilot:
        complete_validation = validate_complete_pilot(decoded, summaries, summary_roots)
        prepare_output(output, refresh=args.refresh)
    if decoded.empty:
        warnings.append("No corrected/new map_summary.csv rows were found; report contains historical maps only")
    subject_rows = pd.concat([historical, decoded], ignore_index=True, sort=False)
    subject_rows["variant_id"] = subject_rows.apply(variant_id, axis=1)
    pilot_means = add_pilot_means(subject_rows)
    if not pilot_means.empty:
        pilot_means["variant_id"] = pilot_means.apply(variant_id, axis=1)
    historical_reference["variant_id"] = historical_reference.apply(variant_id, axis=1)
    branch_summary = pd.concat(
        [subject_rows, pilot_means, historical_reference], ignore_index=True, sort=False
    )
    sort_columns = ["aggregation_level", "subject", "contrast", "branch", "cv_scheme", "additional_smoothing_fwhm_mm"]
    branch_summary = branch_summary.sort_values(sort_columns, na_position="last").reset_index(drop=True)

    long = to_long(branch_summary.copy())
    wide = to_wide(long)
    group_manifest = create_group_maps(subject_rows, output)
    difference_records = create_differences(args, group_manifest, output)
    generated_at = utc_now()

    atomic_csv(branch_summary, output / "branch_summary.csv")
    atomic_csv(long, output / "side_by_side_long.csv")
    atomic_csv(wide, output / "side_by_side_wide.csv")
    atomic_csv(historical_reference, output / "historical_30sub_reference.csv")
    atomic_csv(group_manifest, output / "group_map_manifest.csv")
    if difference_records:
        atomic_text(
            json.dumps(
                {
                    "generated_at": generated_at,
                    "approximation_acknowledged": True,
                    "records": difference_records,
                },
                indent=2,
            ) + "\n",
            output / "difference_manifest.json",
        )
    provenance = {
        "generated_at": generated_at,
        "script": str(Path(__file__).resolve()),
        "pilot_subjects": list(PILOT_SUBJECTS),
        "summary_scan_roots": [str(item) for item in summary_roots],
        "source_map_summaries": [str(item) for item in summaries],
        "historical_within_root": str((DECODING_ROOT / "searchlight_within_source_erp").resolve()),
        "historical_cross_root": str((DECODING_ROOT / "searchlight_cross_source_revised").resolve()),
        "historical_invalid_value_policy": "zero is outside evaluated centers",
        "corrected_new_invalid_value_policy": "NaN is outside evaluated centers",
        "group_map_grid_policy": "same native shape and affine required within every variant",
        "warnings": warnings,
        "require_complete_pilot": bool(args.require_complete_pilot),
        "complete_pilot_validation": complete_validation,
        "command": sys.argv,
    }
    atomic_text(json.dumps(provenance, indent=2) + "\n", output / "provenance.json")
    atomic_text(
        report_markdown(
            branch_summary, historical_reference, group_manifest, warnings,
            difference_records, generated_at, args.require_complete_pilot,
        ),
        output / "REPORT.md",
    )
    print(f"Report written to {output}")
    print(f"Subject result rows: {len(subject_rows)}; group map pairs: {len(group_manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
