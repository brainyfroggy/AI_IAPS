#!/usr/bin/env python3
"""Quantify raw temporal SNR and coverage in fMRIPrep BOLD derivatives.

This is descriptive QC only.  It loads one compressed run once at a time and
does not detrend, filter, smooth, regress nuisance variables, censor volumes,
or alter any input.
Per-run tSNR is temporal mean / sample standard deviation (ddof=1).  Aggregate
tSNR is the equal-run mean of valid per-run tSNR values, so runs keep their
native (and potentially different) temporal lengths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np


DEFAULT_SPACE = "MNI152NLin6Asym"
DEFAULT_RESOLUTION = 2
DEFAULT_RUNS = tuple(range(1, 11))


def _subject_directory(root: Path, subject: int) -> Path:
    candidates = [root / f"sub-{subject:02d}", root / f"sub-{subject}"]
    hits = [path for path in candidates if path.is_dir()]
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one subject directory for Sub{subject} under {root}; found {hits}"
        )
    return hits[0]


def _run_number(path: Path) -> int:
    match = re.search(r"_run-(\d+)_", path.name)
    if not match:
        raise RuntimeError(f"Cannot parse a run number from {path}")
    return int(match.group(1))


def _session_label(path: Path) -> str:
    match = re.search(r"_ses-([A-Za-z0-9]+)_", path.name)
    return match.group(1) if match else ""


def discover_runs(
    fmriprep_root: Path,
    subject: int,
    space: str,
    resolution: int,
    expected_runs: Sequence[int],
) -> list[dict[str, object]]:
    """Discover exactly one target-space BOLD and companions per expected run."""
    expected = tuple(int(run) for run in expected_runs)
    if not expected or len(set(expected)) != len(expected) or any(run < 1 for run in expected):
        raise ValueError(f"Expected runs must be unique positive integers; received {expected}")
    expected = tuple(sorted(expected))

    subject_dir = _subject_directory(fmriprep_root, subject)
    bold_suffix = f"_space-{space}_res-{resolution}_desc-preproc_bold.nii.gz"
    candidates = sorted(subject_dir.glob(f"**/func/*{bold_suffix}"))
    by_run: dict[int, list[Path]] = {}
    for path in candidates:
        by_run.setdefault(_run_number(path), []).append(path)
    observed = tuple(sorted(by_run))
    if observed != expected:
        raise RuntimeError(
            f"Expected target-space runs {list(expected)} for Sub{subject}; found {list(observed)}"
        )

    records: list[dict[str, object]] = []
    for run in expected:
        hits = by_run[run]
        if len(hits) != 1:
            raise RuntimeError(f"Expected one target-space BOLD for run {run:02d}; found {hits}")
        bold = hits[0]
        prefix = bold.name[: -len(bold_suffix)]
        mask = bold.with_name(
            prefix + f"_space-{space}_res-{resolution}_desc-brain_mask.nii.gz"
        )
        sidecar = bold.with_name(
            prefix + f"_space-{space}_res-{resolution}_desc-preproc_bold.json"
        )
        confounds = bold.with_name(prefix + "_desc-confounds_timeseries.tsv")
        for label, path in (("mask", mask), ("JSON sidecar", sidecar), ("confounds", confounds)):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label} companion for {bold}: {path}")
        records.append(
            {
                "run": run,
                "session": _session_label(bold),
                "prefix": prefix,
                "bold": bold,
                "mask": mask,
                "json": sidecar,
                "confounds": confounds,
            }
        )
    return records


def _count_tsv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"Empty confounds file: {path}") from exc
        if not header or not any(value.strip() for value in header):
            raise RuntimeError(f"Confounds file has no header: {path}")
        return sum(1 for row in reader if row and any(value.strip() for value in row))


def _require_exact_grid(
    image: nib.spatialimages.SpatialImage,
    spatial_shape: tuple[int, int, int],
    affine: np.ndarray,
    label: str,
) -> None:
    if tuple(image.shape[:3]) != spatial_shape:
        raise RuntimeError(
            f"Grid shape mismatch for {label}: {tuple(image.shape[:3])} != {spatial_shape}"
        )
    if not np.array_equal(np.asarray(image.affine), affine):
        raise RuntimeError(f"Grid affine mismatch for {label}; exact agreement is required")


def _load_binary_mask(
    path: Path,
    spatial_shape: tuple[int, int, int],
    affine: np.ndarray,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise RuntimeError(f"Brain mask must be 3-D: {path} has shape {image.shape}")
    _require_exact_grid(image, spatial_shape, affine, str(path))
    values = np.asarray(image.dataobj)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Brain mask contains non-finite values: {path}")
    if not np.all((values == 0) | (values == 1)):
        raise RuntimeError(f"Brain mask is not binary 0/1: {path}")
    mask = values.astype(bool, copy=False)
    if not np.any(mask):
        raise RuntimeError(f"Brain mask is empty: {path}")
    return mask, image


def _save_float32_map(data: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"QC map must be 3-D; received {array.shape}")
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(array, reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    if qform is not None:
        image.set_qform(qform, int(qcode))
    if sform is not None:
        image.set_sform(sform, int(scode))
    image.set_data_dtype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))

    check = nib.load(str(path))
    if np.dtype(check.get_data_dtype()) != np.dtype(np.float32):
        raise RuntimeError(f"QC map was not saved as float32: {path}")
    if tuple(check.shape) != tuple(array.shape) or not np.array_equal(check.affine, reference.affine):
        raise RuntimeError(f"QC map grid changed while saving: {path}")


def _tsnr_statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError("Cannot summarize an empty or non-finite tSNR vector")
    percentiles = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "mean_tsnr": float(np.mean(values)),
        "p05_tsnr": float(percentiles[0]),
        "p25_tsnr": float(percentiles[1]),
        "median_tsnr": float(percentiles[2]),
        "p75_tsnr": float(percentiles[3]),
        "p95_tsnr": float(percentiles[4]),
    }


def _compute_run_tsnr(
    bold: nib.Nifti1Image,
    mask: np.ndarray,
    z_chunk: int,
    std_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load one compressed run once, then compute it in bounded z slabs."""
    spatial_shape = tuple(int(value) for value in bold.shape[:3])
    n_volumes = int(bold.shape[3])
    # A .nii.gz proxy may decompress from the beginning for each spatial slice
    # when indexed_gzip is unavailable. Materialize exactly one run, then use
    # slab-wise arithmetic; the array is released before the next run loads.
    data = np.asarray(bold.dataobj, dtype=np.float32)
    expected_data_shape = spatial_shape + (n_volumes,)
    if data.shape != expected_data_shape:
        raise RuntimeError(f"Unexpected BOLD data shape {data.shape}; expected {expected_data_shape}")
    run_tsnr = np.full(spatial_shape, np.nan, dtype=np.float32)
    run_valid = np.zeros(spatial_shape, dtype=bool)
    finite_voxels = 0
    positive_mean_voxels = 0

    for z_start in range(0, spatial_shape[2], z_chunk):
        z_stop = min(spatial_shape[2], z_start + z_chunk)
        mask_block = mask[:, :, z_start:z_stop]
        block = data[:, :, z_start:z_stop, :]
        expected_shape = mask_block.shape + (n_volumes,)
        if block.shape != expected_shape:
            raise RuntimeError(f"Unexpected BOLD slab shape {block.shape}; expected {expected_shape}")

        finite = np.all(np.isfinite(block), axis=3) & mask_block
        finite_voxels += int(np.sum(finite))
        local_tsnr = np.full(mask_block.shape, np.nan, dtype=np.float32)
        local_valid = np.zeros(mask_block.shape, dtype=bool)
        if np.any(finite):
            series = block[finite]
            means = np.mean(series, axis=1, dtype=np.float64)
            standard_deviations = np.std(series, axis=1, ddof=1, dtype=np.float64)
            positive_mean = np.isfinite(means) & (means > 0)
            positive_mean_voxels += int(np.sum(positive_mean))
            valid_values = (
                positive_mean
                & np.isfinite(standard_deviations)
                & (standard_deviations > std_epsilon)
            )
            ratios = np.full(means.shape, np.nan, dtype=np.float64)
            ratios[valid_values] = means[valid_values] / standard_deviations[valid_values]
            valid_values &= np.isfinite(ratios)
            local_valid[finite] = valid_values
            local_tsnr[finite] = ratios.astype(np.float32)

        run_tsnr[:, :, z_start:z_stop] = local_tsnr
        run_valid[:, :, z_start:z_stop] = local_valid

    counts = {
        "finite_timeseries_voxels": finite_voxels,
        "positive_mean_voxels": positive_mean_voxels,
        "valid_tsnr_voxels": int(np.sum(run_valid)),
    }
    if counts["valid_tsnr_voxels"] == 0:
        raise RuntimeError("No valid positive-mean, nonconstant in-mask voxels remain for tSNR")
    return run_tsnr, run_valid, counts


SUMMARY_FIELDS = (
    "row_type",
    "subject",
    "session",
    "run",
    "n_volumes",
    "tr_seconds",
    "grid_voxels",
    "mask_voxels",
    "finite_timeseries_voxels",
    "positive_mean_voxels",
    "valid_tsnr_voxels",
    "valid_fraction_of_mask",
    "mean_tsnr",
    "median_tsnr",
    "p05_tsnr",
    "p25_tsnr",
    "p75_tsnr",
    "p95_tsnr",
    "mask_union_voxels",
    "mask_all_runs_voxels",
    "valid_union_voxels",
    "valid_all_runs_voxels",
    "mean_mask_coverage_fraction",
    "mean_valid_coverage_fraction",
    "input_bold",
    "input_mask",
    "input_confounds",
)


def run_qc(
    fmriprep_root: Path,
    subject: int,
    output: Path,
    *,
    space: str = DEFAULT_SPACE,
    resolution: int = DEFAULT_RESOLUTION,
    expected_runs: Sequence[int] = DEFAULT_RUNS,
    z_chunk: int = 8,
    std_epsilon: float = 1e-6,
) -> Path:
    fmriprep_root = fmriprep_root.resolve()
    output = output.resolve()
    if not fmriprep_root.is_dir():
        raise FileNotFoundError(fmriprep_root)
    if subject < 1:
        raise ValueError("Subject must be a positive integer")
    if z_chunk < 1:
        raise ValueError("z_chunk must be at least 1")
    if not math.isfinite(std_epsilon) or std_epsilon < 0:
        raise ValueError("std_epsilon must be finite and nonnegative")
    if output == fmriprep_root or fmriprep_root in output.parents:
        raise ValueError("QC output must be outside the fMRIPrep derivative root")

    stage = output.parent / f".{output.name}.staging"
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing QC output: {output}")
    if stage.exists():
        raise FileExistsError(f"Refusing an abandoned QC staging path; inspect it first: {stage}")

    records = discover_runs(fmriprep_root, subject, space, resolution, expected_runs)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    per_run_dir = stage / "per_run"
    per_run_dir.mkdir()

    reference_bold: nib.Nifti1Image | None = None
    reference_shape: tuple[int, int, int] | None = None
    reference_affine: np.ndarray | None = None
    reference_tr: float | None = None
    mask_count: np.ndarray | None = None
    valid_count: np.ndarray | None = None
    tsnr_sum: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    provenance_records: list[dict[str, object]] = []
    map_paths: list[Path] = []
    total_volumes = 0

    for record in records:
        run = int(record["run"])
        bold_path = Path(record["bold"])
        mask_path = Path(record["mask"])
        confounds_path = Path(record["confounds"])
        bold = nib.load(str(bold_path))
        if len(bold.shape) != 4 or int(bold.shape[3]) < 2:
            raise RuntimeError(f"BOLD must be 4-D with at least two volumes: {bold_path}")
        spatial_shape = tuple(int(value) for value in bold.shape[:3])
        affine = np.asarray(bold.affine).copy()
        zooms = bold.header.get_zooms()
        tr = float(zooms[3])
        if not math.isfinite(tr) or tr <= 0:
            raise RuntimeError(f"BOLD has invalid TR {tr}: {bold_path}")

        if reference_bold is None:
            reference_bold = bold
            reference_shape = spatial_shape
            reference_affine = affine
            reference_tr = tr
            mask_count = np.zeros(spatial_shape, dtype=np.uint16)
            valid_count = np.zeros(spatial_shape, dtype=np.uint16)
            tsnr_sum = np.zeros(spatial_shape, dtype=np.float64)
        else:
            assert reference_shape is not None and reference_affine is not None
            _require_exact_grid(bold, reference_shape, reference_affine, str(bold_path))
            if tr != reference_tr:
                raise RuntimeError(f"TR mismatch for {bold_path}: {tr} != {reference_tr}")

        assert reference_shape is not None and reference_affine is not None
        assert mask_count is not None and valid_count is not None and tsnr_sum is not None
        mask, _ = _load_binary_mask(mask_path, reference_shape, reference_affine)
        n_volumes = int(bold.shape[3])
        confound_rows = _count_tsv_rows(confounds_path)
        if confound_rows != n_volumes:
            raise RuntimeError(
                f"Confound rows ({confound_rows}) do not match BOLD volumes ({n_volumes}) "
                f"for run {run:02d}: {confounds_path}"
            )

        run_tsnr, run_valid, counts = _compute_run_tsnr(
            bold, mask, z_chunk=z_chunk, std_epsilon=std_epsilon
        )
        mask_count += mask.astype(np.uint16)
        valid_count += run_valid.astype(np.uint16)
        tsnr_sum[run_valid] += run_tsnr[run_valid].astype(np.float64)
        total_volumes += n_volumes

        valid_map = np.full(reference_shape, np.nan, dtype=np.float32)
        valid_map[mask] = 0.0
        valid_map[run_valid] = 1.0
        output_stem = (
            f"{record['prefix']}_space-{space}_res-{resolution}"
        )
        tsnr_path = per_run_dir / f"{output_stem}_desc-tsnr_map.nii.gz"
        valid_path = per_run_dir / f"{output_stem}_desc-valid_tsnr_mask.nii.gz"
        _save_float32_map(run_tsnr, bold, tsnr_path)
        _save_float32_map(valid_map, bold, valid_path)
        map_paths.extend([tsnr_path, valid_path])

        statistics = _tsnr_statistics(run_tsnr[run_valid])
        mask_voxels = int(np.sum(mask))
        row: dict[str, object] = {field: "" for field in SUMMARY_FIELDS}
        row.update(
            {
                "row_type": "run",
                "subject": f"Sub{subject}",
                "session": record["session"],
                "run": run,
                "n_volumes": n_volumes,
                "tr_seconds": tr,
                "grid_voxels": int(np.prod(reference_shape)),
                "mask_voxels": mask_voxels,
                **counts,
                "valid_fraction_of_mask": counts["valid_tsnr_voxels"] / mask_voxels,
                **statistics,
                "input_bold": str(bold_path.resolve()),
                "input_mask": str(mask_path.resolve()),
                "input_confounds": str(confounds_path.resolve()),
            }
        )
        rows.append(row)
        provenance_records.append(
            {
                "run": run,
                "session": record["session"],
                "n_volumes": n_volumes,
                "tr_seconds": tr,
                "bold": str(bold_path.resolve()),
                "mask": str(mask_path.resolve()),
                "json": str(Path(record["json"]).resolve()),
                "confounds": str(confounds_path.resolve()),
                "tsnr_map": str(tsnr_path.relative_to(stage)),
                "valid_map": str(valid_path.relative_to(stage)),
            }
        )

        # Release the proxy and all per-run arrays before loading the next run.
        del bold, mask, run_tsnr, run_valid, valid_map

    assert reference_bold is not None and reference_shape is not None
    assert mask_count is not None and valid_count is not None and tsnr_sum is not None
    n_runs = len(records)
    mask_union = mask_count > 0
    valid_union = valid_count > 0
    aggregate_tsnr = np.full(reference_shape, np.nan, dtype=np.float32)
    aggregate_tsnr[valid_union] = (tsnr_sum[valid_union] / valid_count[valid_union]).astype(
        np.float32
    )
    mask_coverage = np.full(reference_shape, np.nan, dtype=np.float32)
    valid_coverage = np.full(reference_shape, np.nan, dtype=np.float32)
    valid_run_count_map = np.full(reference_shape, np.nan, dtype=np.float32)
    mask_coverage[mask_union] = mask_count[mask_union].astype(np.float32) / n_runs
    valid_coverage[mask_union] = valid_count[mask_union].astype(np.float32) / n_runs
    valid_run_count_map[mask_union] = valid_count[mask_union].astype(np.float32)

    aggregate_prefix = f"sub-{subject:02d}_space-{space}_res-{resolution}"
    aggregate_paths = {
        "mean_tsnr": stage / f"{aggregate_prefix}_desc-mean_tsnr_map.nii.gz",
        "brainmask_coverage_fraction": stage
        / f"{aggregate_prefix}_desc-brainmask_coverage_fraction_map.nii.gz",
        "valid_tsnr_coverage_fraction": stage
        / f"{aggregate_prefix}_desc-valid_tsnr_coverage_fraction_map.nii.gz",
        "valid_tsnr_run_count": stage
        / f"{aggregate_prefix}_desc-valid_tsnr_run_count_map.nii.gz",
    }
    _save_float32_map(aggregate_tsnr, reference_bold, aggregate_paths["mean_tsnr"])
    _save_float32_map(mask_coverage, reference_bold, aggregate_paths["brainmask_coverage_fraction"])
    _save_float32_map(valid_coverage, reference_bold, aggregate_paths["valid_tsnr_coverage_fraction"])
    _save_float32_map(valid_run_count_map, reference_bold, aggregate_paths["valid_tsnr_run_count"])
    map_paths.extend(aggregate_paths.values())

    aggregate_stats = _tsnr_statistics(aggregate_tsnr[valid_union])
    aggregate_row: dict[str, object] = {field: "" for field in SUMMARY_FIELDS}
    aggregate_row.update(
        {
            "row_type": "aggregate_equal_run_mean",
            "subject": f"Sub{subject}",
            "session": "all",
            "run": "all",
            "n_volumes": total_volumes,
            "tr_seconds": reference_tr,
            "grid_voxels": int(np.prod(reference_shape)),
            "mask_voxels": int(np.sum(mask_union)),
            "finite_timeseries_voxels": "",
            "positive_mean_voxels": "",
            "valid_tsnr_voxels": int(np.sum(valid_union)),
            "valid_fraction_of_mask": float(np.sum(valid_union) / np.sum(mask_union)),
            **aggregate_stats,
            "mask_union_voxels": int(np.sum(mask_union)),
            "mask_all_runs_voxels": int(np.sum(mask_count == n_runs)),
            "valid_union_voxels": int(np.sum(valid_union)),
            "valid_all_runs_voxels": int(np.sum(valid_count == n_runs)),
            "mean_mask_coverage_fraction": float(np.mean(mask_coverage[mask_union])),
            "mean_valid_coverage_fraction": float(np.mean(valid_coverage[mask_union])),
        }
    )
    rows.append(aggregate_row)

    summary_path = stage / "tsnr_summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    script_path = Path(__file__).resolve()
    provenance = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": subject,
        "fmriprep_root": str(fmriprep_root),
        "output_space": space,
        "resolution": resolution,
        "expected_runs": [int(run) for run in sorted(expected_runs)],
        "z_chunk": z_chunk,
        "std_epsilon": std_epsilon,
        "tsnr_definition": "temporal mean / sample temporal standard deviation (ddof=1)",
        "aggregate_definition": "equal-run mean of valid per-run tSNR; native run lengths retained",
        "valid_voxel_definition": (
            "inside run mask; every time point finite; positive temporal mean; "
            "finite temporal SD greater than std_epsilon"
        ),
        "preprocessing_applied_by_this_utility": [],
        "script": str(script_path),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "runs": provenance_records,
        "aggregate_maps": {
            key: str(path.relative_to(stage)) for key, path in aggregate_paths.items()
        },
        "summary_tsv": str(summary_path.relative_to(stage)),
    }
    (stage / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    expected_map_count = 2 * n_runs + len(aggregate_paths)
    if len(map_paths) != expected_map_count or not all(path.is_file() for path in map_paths):
        raise RuntimeError(
            f"QC map completeness check failed: {len(map_paths)} paths; expected {expected_map_count}"
        )
    if not summary_path.is_file() or not (stage / "provenance.json").is_file():
        raise RuntimeError("QC summary/provenance completeness check failed")

    stage.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmriprep-root", required=True, type=Path)
    parser.add_argument("--subject", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--space", default=DEFAULT_SPACE)
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION, type=int)
    parser.add_argument("--expected-runs", nargs="+", type=int, default=list(DEFAULT_RUNS))
    parser.add_argument("--z-chunk", type=int, default=8)
    parser.add_argument("--std-epsilon", type=float, default=1e-6)
    args = parser.parse_args()
    result = run_qc(
        args.fmriprep_root,
        args.subject,
        args.output,
        space=args.space,
        resolution=args.resolution,
        expected_runs=args.expected_runs,
        z_chunk=args.z_chunk,
        std_epsilon=args.std_epsilon,
    )
    print(f"Quantitative fMRIPrep tSNR QC complete: {result}")


if __name__ == "__main__":
    main()
