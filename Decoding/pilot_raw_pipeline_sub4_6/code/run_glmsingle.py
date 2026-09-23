#!/usr/bin/env python3
"""Estimate leakage-aware single-trial betas from fMRIPrep outputs.

The wrapper uses masked 2-D data to keep GLMsingle memory bounded, 120 shared
stimulus-identity design columns, actual acquisition-session indicators, a 24P
motion model, and explicit FD>0.5 spike regressors. Outputs are isolated by
subject/branch and validated before the command exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PILOT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / ".codex_work" / "pilot_pipeline" / "vendor" / "GLMsingle"
GLMSINGLE_COMMIT = "1ab54a65edd3ea41a6133d4b4ecb78a9c7296684"
GLMSINGLE_SOURCE_RELATIVE = Path("glmsingle") / "glmsingle.py"
# Git for Windows and WSL may materialize different line endings on the shared
# checkout. Pin the exact source text after canonicalizing CRLF to LF.
GLMSINGLE_PATCHED_SOURCE_SHA256 = (
    "df98a59a0fffba540cb88033a6a698b114982ee5be2360869fd7fbe4a911a3fa"
)
GLMSINGLE_ZERO_PC_PATCH = (
    PILOT_ROOT / "code" / "patches" / "glmsingle_zero_pc_extra_regressors.patch"
)
GLMSINGLE_MIXED_RUN_FIR_PATCH = (
    PILOT_ROOT / "code" / "patches" / "glmsingle_mixed_run_fir_pairing.patch"
)
# Keep the original name as an alias because completed Sub4/Sub6 provenance
# uses this field for the immutable zero-PC patch.
GLMSINGLE_PATCH = GLMSINGLE_ZERO_PC_PATCH
GLMSINGLE_PATCH_SPECS = (
    (
        "zero_pc_extra_regressors",
        GLMSINGLE_ZERO_PC_PATCH,
        "b8f109d07f5d22299ae80d1303fa29fa55eec9ffaf555a63c885a8764ca625d4",
    ),
    (
        "mixed_run_fir_pairing",
        GLMSINGLE_MIXED_RUN_FIR_PATCH,
        "62a8a803ca4d6b1e459064a177a65a45d3fc152f02dd29dc94dac12a17fd729c",
    ),
)
sys.path.insert(0, str(VENDOR_ROOT))

from glmsingle.glmsingle import GLM_single, _merge_extra_regressors  # noqa: E402


TR = 1.8
STIMDUR = 3.0
SEED = 20260728
DEFAULT_FRACS = tuple(round(1.0 - 0.05 * index, 2) for index in range(20))
SESSION_INDICATORS = {
    4: [1] * 10,
    5: [1, 2, 1, 1, 1, 2, 2, 2, 2, 2],
    6: [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
}
# Sub5's February (session 1) acquisitions contain 224 volumes, whereas its
# March (session 2) acquisitions contain 218.  Keep this in run-number order:
# temporal length is a per-run property and must not be homogenized across the
# two acquisition sessions.
EXPECTED_RUN_N_VOLUMES = {
    5: (224, 218, 224, 224, 224, 218, 218, 218, 218, 218),
}
MOTION_BASE = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
MOTION_SUFFIXES = ["", "_derivative1", "_power2", "_derivative1_power2"]


def _vendor_git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    command = [
        "git",
        "-c",
        f"safe.directory={VENDOR_ROOT}",
        "-c",
        "core.fileMode=false",
        "-c",
        "core.autocrlf=true",
        "-C",
        str(VENDOR_ROOT),
        *arguments,
    ]
    try:
        return subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError):
            detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Cannot validate pinned GLMsingle vendor with {' '.join(command)}: {detail or error}"
        ) from error


def _source_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()


def validate_glmsingle_vendor() -> dict:
    """Fail closed unless the checkout is exactly the pinned ordered patch state."""
    source_path = VENDOR_ROOT / GLMSINGLE_SOURCE_RELATIVE
    if not source_path.is_file():
        raise RuntimeError(f"Missing pinned GLMsingle source: {source_path}")

    commit = _vendor_git("rev-parse", "HEAD").stdout.decode("ascii").strip()
    if commit != GLMSINGLE_COMMIT:
        raise RuntimeError(
            f"GLMsingle HEAD is {commit}; expected pinned commit {GLMSINGLE_COMMIT}"
        )

    verified_patches = []
    for order, (name, path, expected_sha256) in enumerate(GLMSINGLE_PATCH_SPECS, start=1):
        if not path.is_file():
            raise RuntimeError(f"Missing GLMsingle patch {name}: {path}")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"GLMsingle patch {name} SHA-256 is {actual_sha256}; "
                f"expected {expected_sha256}"
            )
        verified_patches.append(
            {
                "order": order,
                "name": name,
                "file": str(path),
                "sha256": actual_sha256,
            }
        )

    # Reconstruct the source from the pinned commit and patches in declared
    # order. This proves that the patch bytes, their order, and the live vendor
    # source all describe one exact state rather than merely sharing labels.
    base_source = _vendor_git(
        "show", f"{GLMSINGLE_COMMIT}:{GLMSINGLE_SOURCE_RELATIVE.as_posix()}"
    ).stdout
    with tempfile.TemporaryDirectory(prefix="glmsingle-vendor-check-") as temp_name:
        temp_root = Path(temp_name)
        reconstructed_path = temp_root / GLMSINGLE_SOURCE_RELATIVE
        reconstructed_path.parent.mkdir(parents=True)
        reconstructed_path.write_bytes(base_source)
        for name, patch_path, _ in GLMSINGLE_PATCH_SPECS:
            try:
                subprocess.run(
                    ["git", "apply", "--whitespace=nowarn", str(patch_path)],
                    cwd=temp_root,
                    check=True,
                    capture_output=True,
                )
            except (OSError, subprocess.CalledProcessError) as error:
                detail = ""
                if isinstance(error, subprocess.CalledProcessError):
                    detail = error.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"Cannot apply ordered GLMsingle patch {name}: {detail or error}"
                ) from error
        reconstructed_sha256 = _source_sha256(reconstructed_path.read_bytes())

    if reconstructed_sha256 != GLMSINGLE_PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            "Ordered GLMsingle patches reconstruct source SHA-256 "
            f"{reconstructed_sha256}; expected {GLMSINGLE_PATCHED_SOURCE_SHA256}"
        )
    actual_source_bytes = source_path.read_bytes()
    actual_source_sha256 = _source_sha256(actual_source_bytes)
    if actual_source_sha256 != reconstructed_sha256:
        raise RuntimeError(
            f"Live GLMsingle source SHA-256 is {actual_source_sha256}; "
            f"ordered patches require {reconstructed_sha256}"
        )

    changed_files = [
        item
        for item in _vendor_git(
            "diff", "--ignore-cr-at-eol", "--name-only", GLMSINGLE_COMMIT, "--"
        )
        .stdout.decode("utf-8")
        .splitlines()
        if item
    ]
    expected_changed_files = [GLMSINGLE_SOURCE_RELATIVE.as_posix()]
    if changed_files != expected_changed_files:
        raise RuntimeError(
            f"Unexpected tracked GLMsingle changes {changed_files}; "
            f"expected {expected_changed_files}"
        )
    return {
        "validated": True,
        "commit": commit,
        "source_file": str(source_path),
        "source_sha256": actual_source_sha256,
        "source_raw_bytes_sha256": hashlib.sha256(actual_source_bytes).hexdigest(),
        "tracked_changed_files": changed_files,
        "patches": verified_patches,
    }


def subject_dirs(root: Path, subject: int) -> list[Path]:
    dirs = [root / f"sub-{subject:02d}", root / f"sub-{subject}"]
    return [path for path in dirs if path.exists()]


def run_number(path: Path) -> int:
    match = re.search(r"_run-(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse run number from {path}")
    return int(match.group(1))


def discover_run_files(root: Path, subject: int, space: str) -> list[dict[str, Path]]:
    candidates: list[Path] = []
    for subdir in subject_dirs(root, subject):
        candidates.extend(subdir.glob(f"**/func/*_space-{space}_res-2_desc-preproc_bold.nii"))
        candidates.extend(subdir.glob(f"**/func/*_space-{space}_res-2_desc-preproc_bold.nii.gz"))
    by_run: dict[int, Path] = {}
    for path in sorted(candidates):
        run = run_number(path)
        # Prefer uncompressed NIfTI when both representations exist.
        if run not in by_run or path.suffix == ".nii":
            by_run[run] = path
    if sorted(by_run) != list(range(1, 11)):
        raise RuntimeError(f"Expected runs 1-10 for Sub{subject}; found {sorted(by_run)}")
    records = []
    for run in range(1, 11):
        bold = by_run[run]
        stem_prefix = bold.name.split(f"_space-{space}")[0]
        mask_hits = sorted(bold.parent.glob(f"{stem_prefix}_space-{space}_res-2_desc-brain_mask.nii*"))
        if len(mask_hits) == 2 and {path.name.removesuffix(".gz") for path in mask_hits} == {mask_hits[0].name.removesuffix(".gz")}:
            mask_hits = [next(path for path in mask_hits if path.suffix == ".nii")]
        confound_hits = sorted(bold.parent.glob(f"{stem_prefix}_desc-confounds_timeseries.tsv"))
        json_hits = sorted(bold.parent.glob(f"{stem_prefix}_space-{space}_res-2_desc-preproc_bold.json"))
        if len(mask_hits) != 1 or len(confound_hits) != 1 or len(json_hits) != 1:
            raise RuntimeError(
                f"Missing/ambiguous companion files for {bold}: mask={mask_hits}, "
                f"confounds={confound_hits}, json={json_hits}"
            )
        records.append({"bold": bold, "mask": mask_hits[0], "confounds": confound_hits[0], "json": json_hits[0]})
    return records


def discover_events(bids_root: Path, subject: int, run: int) -> Path:
    hits = []
    for subdir in subject_dirs(bids_root, subject):
        hits.extend(subdir.glob(f"**/func/*_run-{run:02d}_events.tsv"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected one events file for Sub{subject} Run{run:02d}; found {hits}")
    return hits[0]


def validate_run_n_volumes(records: list[dict[str, Path]], subject: int) -> list[int]:
    """Return per-run lengths and enforce known acquisition-specific invariants."""
    observed = [int(nib.load(record["bold"]).shape[-1]) for record in records]
    expected = EXPECTED_RUN_N_VOLUMES.get(subject)
    if expected is not None and tuple(observed) != expected:
        raise RuntimeError(
            f"Sub{subject} BOLD volumes by run are {observed}; expected {list(expected)}. "
            "Do not pad, crop, or mix the February and March derivatives."
        )
    return observed


def intersection_mask(records: list[dict[str, Path]]) -> tuple[np.ndarray, nib.Nifti1Image]:
    reference = nib.load(records[0]["mask"])
    mask = np.asarray(reference.dataobj) > 0
    for record in records[1:]:
        image = nib.load(record["mask"])
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5):
            raise RuntimeError("Run masks do not share a grid")
        mask &= np.asarray(image.dataobj) > 0
    return mask, reference


def build_nuisance(tsv: Path, n_time: int) -> tuple[np.ndarray, list[str], dict]:
    frame = pd.read_csv(tsv, sep="\t")
    if len(frame) != n_time:
        raise RuntimeError(f"Confound rows ({len(frame)}) do not match BOLD volumes ({n_time}) in {tsv}")
    continuous_names = [f"{base}{suffix}" for base in MOTION_BASE for suffix in MOTION_SUFFIXES]
    missing = [name for name in continuous_names if name not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing motion columns in {tsv}: {missing}")
    continuous = frame[continuous_names].fillna(0.0).to_numpy(np.float32)
    mean = continuous.mean(axis=0, keepdims=True)
    std = continuous.std(axis=0, ddof=1, keepdims=True)
    keep = np.isfinite(std[0]) & (std[0] > 1e-7)
    continuous = (continuous[:, keep] - mean[:, keep]) / std[:, keep]
    names = [name for name, flag in zip(continuous_names, keep) if flag]

    fd = frame.get("framewise_displacement", pd.Series(np.zeros(n_time))).fillna(0.0).to_numpy(float)
    spike_index = np.flatnonzero(fd > 0.5)
    spikes = np.zeros((n_time, len(spike_index)), dtype=np.float32)
    if len(spike_index):
        spikes[spike_index, np.arange(len(spike_index))] = 1.0
    names.extend([f"fd_spike_{index:03d}" for index in spike_index])
    nuisance = np.c_[continuous, spikes].astype(np.float32, copy=False)
    if nuisance.shape[1] and np.linalg.matrix_rank(nuisance) < nuisance.shape[1]:
        # Remove linearly dependent columns deterministically with pivoted QR.
        from scipy.linalg import qr

        _, r, pivots = qr(nuisance, mode="economic", pivoting=True)
        rank = int(np.sum(np.abs(np.diag(r)) > 1e-7))
        selected = sorted(pivots[:rank])
        nuisance = nuisance[:, selected]
        names = [names[index] for index in selected]
    qc = {
        "n_volumes": n_time,
        "n_motion_columns": int(np.sum(keep)),
        "n_fd_spikes_gt_0p5": int(len(spike_index)),
        "mean_fd": float(np.mean(fd)),
        "max_fd": float(np.max(fd)),
        "n_nuisance_columns_after_rank_filter": int(nuisance.shape[1]),
    }
    return nuisance, names, qc


def build_design(
    records: list[dict[str, Path]], bids_root: Path, subject: int
) -> tuple[list[np.ndarray], pd.DataFrame, list[float]]:
    event_frames = []
    for run, record in enumerate(records, start=1):
        events = pd.read_csv(discover_events(bids_root, subject, run), sep="\t")
        if len(events) != 60:
            raise RuntimeError(f"Sub{subject} Run{run:02d} has {len(events)} events, expected 60")
        if "duration" not in events or not np.allclose(events["duration"], STIMDUR, atol=1e-6):
            raise RuntimeError(
                f"Sub{subject} Run{run:02d} event durations do not all equal {STIMDUR} s"
            )
        events.insert(0, "events_file_row", np.arange(len(events), dtype=int))
        # GLMsingle numbers trials by scanning design rows from first to last,
        # irrespective of row order in the BIDS events file.
        events = events.sort_values(["onset", "events_file_row"], kind="stable").reset_index(drop=True)
        events.insert(0, "run", run)
        event_frames.append(events)
    manifest = pd.concat(event_frames, ignore_index=True)
    identities = sorted(manifest["image_id"].astype(str).unique())
    if len(identities) != 120 or not np.all(manifest["image_id"].value_counts().to_numpy() == 5):
        raise RuntimeError("Expected 120 image identities repeated exactly five times")
    identity_to_column = {identity: index for index, identity in enumerate(identities)}
    manifest["onset_bin"] = np.full(len(manifest), -1, dtype=int)
    manifest["design_column"] = np.full(len(manifest), -1, dtype=int)
    designs: list[np.ndarray] = []
    start_times: list[float] = []
    for run, record in enumerate(records, start=1):
        n_time = nib.load(record["bold"]).shape[-1]
        metadata = json.loads(record["json"].read_text(encoding="utf-8"))
        metadata_tr = float(metadata.get("RepetitionTime", TR))
        if not np.isclose(metadata_tr, TR, atol=1e-6):
            raise RuntimeError(
                f"Sub{subject} Run{run:02d} has TR={metadata_tr}, expected {TR}"
            )
        start_time = float(metadata.get("StartTime", 0.0))
        start_times.append(start_time)
        events = manifest[manifest["run"] == run].copy()
        onset_bins = np.rint((events["onset"].to_numpy(float) - start_time) / TR).astype(int)
        if len(np.unique(onset_bins)) != 60 or np.any(onset_bins < 0) or np.any(onset_bins >= n_time):
            raise RuntimeError(f"Invalid/colliding onset bins for Sub{subject} Run{run:02d}: {onset_bins}")
        design = np.zeros((n_time, 120), dtype=np.float32)
        columns = [identity_to_column[str(value)] for value in events["image_id"]]
        design[onset_bins, columns] = 1.0
        designs.append(design)
        manifest.loc[events.index, "onset_bin"] = onset_bins
        manifest.loc[events.index, "design_column"] = columns
    if [int(item.sum()) for item in designs] != [60] * 10:
        raise RuntimeError("Design does not contain 60 trials in every run")
    emitted_columns = []
    for design in designs:
        for row in np.flatnonzero(design.sum(axis=1)):
            emitted_columns.append(int(np.flatnonzero(design[row])[0]))
    if emitted_columns != manifest["design_column"].astype(int).tolist():
        raise RuntimeError("Trial manifest order does not match GLMsingle beta chronology")
    manifest.insert(0, "beta_index", np.arange(len(manifest), dtype=int))
    return designs, manifest, start_times


def load_masked_data(
    records: list[dict[str, Path]], flat_indices: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]], list[dict]]:
    data: list[np.ndarray] = []
    nuisance: list[np.ndarray] = []
    nuisance_names: list[list[str]] = []
    motion_qc: list[dict] = []
    for run, record in enumerate(records, start=1):
        image = nib.load(record["bold"])
        coords = np.column_stack(np.unravel_index(flat_indices, image.shape[:3]))
        # Materialize/decompress each run sequentially, then select voxels.
        # Proxy slicing a narrow XYZ block causes millions of small seeks in
        # NIfTI's on-disk order and is dramatically slower on NTFS/SMB.
        # get_fdata forces a sequential in-memory read. Keeping an on-disk
        # memmap here makes voxel-by-time advanced indexing degenerate into
        # millions of tiny filesystem reads.
        volume = image.get_fdata(dtype=np.float32, caching="unchanged")
        matrix = np.asarray(
            volume[coords[:, 0], coords[:, 1], coords[:, 2], :],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"Nonfinite BOLD values in Sub run {run}")
        data.append(np.ascontiguousarray(matrix, dtype=np.float32))
        confounds, names, qc = build_nuisance(record["confounds"], matrix.shape[1])
        nuisance.append(confounds)
        nuisance_names.append(names)
        qc["run"] = run
        motion_qc.append(qc)
        del volume, matrix
        print(f"Loaded run {run:02d}: {data[-1].shape}, nuisance={nuisance[-1].shape}", flush=True)
    return data, nuisance, nuisance_names, motion_qc


def validate_hdf5(
    path: Path,
    n_voxels: int,
    allowed_fracs: np.ndarray | None = None,
    beta_block_voxels: int = 8192,
) -> dict:
    required_datasets = {"betasmd", "HRFindex", "FRACvalue", "R2"}
    if not path.exists():
        raise RuntimeError(f"Missing GLMsingle output: {path}")
    with h5py.File(path, "r") as handle:
        missing = required_datasets.difference(handle.keys())
        if missing:
            raise RuntimeError(f"Missing Type-D HDF5 keys: {sorted(missing)}")
        if "pcnum" in handle.attrs:
            pcnum = int(handle.attrs["pcnum"])
        elif "pcnum" in handle:
            pcnum = int(np.asarray(handle["pcnum"]).reshape(-1)[0])
        else:
            raise RuntimeError("Missing Type-D HDF5 scalar: pcnum")
        shape = tuple(handle["betasmd"].shape)
        expected_shape = (n_voxels, 1, 1, 600)
        if shape != expected_shape:
            raise RuntimeError(f"Unexpected beta shape {shape}; expected {expected_shape}")
        expected_spatial = (n_voxels, 1, 1)
        for key in ("HRFindex", "FRACvalue", "R2"):
            if tuple(handle[key].shape) != expected_spatial:
                raise RuntimeError(
                    f"Unexpected {key} shape {handle[key].shape}; expected {expected_spatial}"
                )
        if beta_block_voxels < 1:
            raise ValueError("beta_block_voxels must be positive")
        beta_min = np.inf
        beta_max = -np.inf
        beta_dataset = handle["betasmd"]
        for start in range(0, n_voxels, beta_block_voxels):
            stop = min(n_voxels, start + beta_block_voxels)
            beta_block = np.asarray(beta_dataset[start:stop, ..., :])
            if not np.all(np.isfinite(beta_block)):
                bad = np.argwhere(~np.isfinite(beta_block))[0]
                raise RuntimeError(
                    "Nonfinite Type-D beta at masked voxel "
                    f"{start + int(bad[0])}, trial {int(bad[-1])}"
                )
            beta_min = min(beta_min, float(np.min(beta_block)))
            beta_max = max(beta_max, float(np.max(beta_block)))

        spatial_values = {
            key: np.asarray(handle[key], dtype=np.float32).reshape(-1)
            for key in ("HRFindex", "FRACvalue", "R2")
        }
        for key, values in spatial_values.items():
            if not np.all(np.isfinite(values)):
                bad = int(np.flatnonzero(~np.isfinite(values))[0])
                raise RuntimeError(f"Nonfinite {key} at masked voxel {bad}")

        frac_values = spatial_values["FRACvalue"]
        canonical_distribution: dict[str, int] | None = None
        minimum_fraction: float
        minimum_fraction_count: int
        if allowed_fracs is not None:
            canonical_allowed = np.asarray(allowed_fracs, dtype=np.float64).reshape(-1)
            allowed = canonical_allowed.astype(np.float32)
            distances = np.abs(frac_values[:, None] - allowed[None, :])
            nearest = np.argmin(distances, axis=1)
            matches = np.isclose(
                frac_values,
                allowed[nearest],
                rtol=1e-6,
                atol=1e-6,
            )
            if not np.all(matches):
                bad = int(np.flatnonzero(~matches)[0])
                raise RuntimeError(
                    f"FRACvalue {float(frac_values[bad])} at masked voxel {bad} "
                    "is outside the requested fractional-ridge grid"
                )
            canonical_distribution = {
                format(float(canonical_allowed[index]), ".12g"): int(np.sum(nearest == index))
                for index in range(len(canonical_allowed))
                if np.any(nearest == index)
            }
            minimum_index = int(np.argmin(canonical_allowed))
            minimum_fraction = float(canonical_allowed[minimum_index])
            minimum_fraction_count = int(np.sum(nearest == minimum_index))
        unique_fracs, frac_counts = np.unique(frac_values, return_counts=True)
        if allowed_fracs is None:
            minimum_fraction = float(unique_fracs[0])
            minimum_fraction_count = int(frac_counts[0])
        return {
            "betasmd_shape": shape,
            "betasmd_dtype": str(beta_dataset.dtype),
            "betasmd_all_values_finite": True,
            "betasmd_min": beta_min,
            "betasmd_max": beta_max,
            "pcnum": pcnum,
            "datasets": sorted(handle.keys()),
            "attributes": sorted(handle.attrs.keys()),
            "hrfindex_min": float(np.min(spatial_values["HRFindex"])),
            "hrfindex_max": float(np.max(spatial_values["HRFindex"])),
            "r2_min": float(np.min(spatial_values["R2"])),
            "r2_max": float(np.max(spatial_values["R2"])),
            "fractional_ridge_distribution": canonical_distribution
            or {
                format(float(value), ".8g"): int(count)
                for value, count in zip(unique_fracs, frac_counts)
            },
            "minimum_requested_fraction": minimum_fraction,
            "minimum_fraction_selected_count": minimum_fraction_count,
            "minimum_fraction_selected_proportion": minimum_fraction_count / n_voxels,
        }


def run(args: argparse.Namespace) -> None:
    np.random.seed(SEED)
    if args.fixed_frac is not None and args.fracs is not None:
        raise ValueError("Use either --fixed-frac or --fracs, not both")
    if args.fixed_frac is not None and not 0 < args.fixed_frac <= 1:
        raise ValueError("--fixed-frac must be in (0, 1]")
    if args.fracs is not None and any(not 0 < value <= 1 for value in args.fracs):
        raise ValueError("Every --fracs value must be in (0, 1]")
    if args.max_voxels is not None and args.max_voxels <= 0:
        raise ValueError("--max-voxels must be positive")
    vendor_validation = validate_glmsingle_vendor()
    records = discover_run_files(args.fmriprep_root, args.subject, args.space)
    run_n_volumes = validate_run_n_volumes(records, args.subject)
    mask, reference = intersection_mask(records)
    for record in records:
        bold_image = nib.load(record["bold"])
        if bold_image.shape[:3] != reference.shape or not np.allclose(
            bold_image.affine, reference.affine, atol=1e-5
        ):
            raise RuntimeError(f"BOLD and mask grids differ for {record['bold']}")
    flat_indices = np.flatnonzero(mask.ravel())
    if not len(flat_indices):
        raise RuntimeError("The across-run intersection brain mask is empty")
    if args.max_voxels and len(flat_indices) > args.max_voxels:
        # A contiguous central block keeps smoke-test I/O efficient on the SMB
        # share while sampling brain rather than the inferior mask edge.
        start = (len(flat_indices) - args.max_voxels) // 2
        flat_indices = flat_indices[start : start + args.max_voxels]
        limited = np.zeros(mask.size, dtype=bool)
        limited[flat_indices] = True
        mask = limited.reshape(mask.shape)
    output = args.output
    if output.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output exists; use --overwrite only for this isolated target: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    scratch = output / "scratch"
    scratch.mkdir()

    mask_header = reference.header.copy()
    mask_header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, mask_header), output / "analysis_mask.nii.gz")
    np.save(output / "flat_mask_indices.npy", flat_indices)
    designs, trial_manifest, start_times = build_design(records, args.bids_root, args.subject)
    trial_manifest.to_csv(output / "trial_manifest.tsv", sep="\t", index=False)
    data, nuisance, nuisance_names, motion_qc = load_masked_data(records, flat_indices)
    pd.DataFrame(motion_qc).to_csv(output / "motion_qc.tsv", sep="\t", index=False)
    (output / "nuisance_columns.json").write_text(json.dumps(nuisance_names, indent=2) + "\n")

    if args.fracs is not None:
        requested_fracs = [float(value) for value in args.fracs]
    elif args.fixed_frac is not None:
        requested_fracs = [float(args.fixed_frac)]
    else:
        requested_fracs = list(DEFAULT_FRACS)
    fracs = np.asarray(requested_fracs, dtype=np.float32)
    params = {
        "wantlibrary": 1,
        "wantglmdenoise": 1,
        "wantfracridge": 1,
        "wantlss": 0,
        "n_pcs": args.n_pcs,
        "pcstop": 1.05,
        "fracs": fracs,
        "wantautoscale": 1,
        "wantpercentbold": 1,
        "xvalscheme": np.arange(10, dtype=np.int64),
        "sessionindicator": np.asarray(SESSION_INDICATORS[args.subject], dtype=np.int64),
        "extra_regressors": nuisance,
        "maxpolydeg": [3] * 10,
        "chunklen": args.chunklen,
        "wantfileoutputs": [0, 1, 0, 1],
        "wantmemoryoutputs": [0, 0, 0, 0],
        "wanthdf5": 1,
        "brainexclude": False,
        "pcR2cutoffmask": 1,
        "seed": SEED,
    }
    provenance = {
        "subject": args.subject,
        "space": args.space,
        "fmriprep_root": str(args.fmriprep_root),
        "bids_root": str(args.bids_root),
        "n_voxels": int(len(flat_indices)),
        "run_n_volumes": run_n_volumes,
        "bold_file_sizes_bytes": [record["bold"].stat().st_size for record in records],
        "bold_files": [str(record["bold"]) for record in records],
        "slice_reference_start_times_seconds": start_times,
        "sessionindicator": SESSION_INDICATORS[args.subject],
        "glmsingle_commit": vendor_validation["commit"],
        "glmsingle_local_motion_regressor_patch": callable(_merge_extra_regressors),
        "glmsingle_patch_file": str(GLMSINGLE_PATCH),
        "glmsingle_patch_sha256": GLMSINGLE_PATCH_SPECS[0][2],
        "glmsingle_vendor_validation": vendor_validation,
        "fracs": requested_fracs,
        "n_pcs": args.n_pcs,
        "seed": SEED,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    previous = Path.cwd()
    os.chdir(scratch)
    try:
        GLM_single(params).fit(
            designs,
            data,
            STIMDUR,
            TR,
            outputdir=str(output / "glmsingle"),
            figuredir=None,
        )
    finally:
        os.chdir(previous)
    validation = validate_hdf5(
        output / "glmsingle" / "TYPED_FITHRF_GLMDENOISE_RR.hdf5",
        len(flat_indices),
        allowed_fracs=np.asarray(requested_fracs, dtype=np.float64),
    )
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=[4, 5, 6], required=True)
    parser.add_argument("--fmriprep-root", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--space", default="MNI152NLin6Asym")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunklen", type=int, default=10000)
    parser.add_argument("--n-pcs", type=int, default=10)
    parser.add_argument("--fixed-frac", type=float)
    parser.add_argument(
        "--fracs",
        type=float,
        nargs="+",
        help=(
            "Explicit cross-validated fractional-ridge grid override. Omit for the "
            "primary 20-value grid from 1.00 through 0.05 in 0.05 steps."
        ),
    )
    parser.add_argument("--max-voxels", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
