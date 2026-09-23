#!/usr/bin/env python3
"""Shared, fail-closed source and onset validation for the 28-subject rerun.

Discovery may recurse through the explicitly configured acquisition folders, but
production selection never uses a glob: every NIfTI is named in an approved TSV.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat


SELECTION_COLUMNS = [
    "asset_id",
    "asset_type",
    "subject",
    "session",
    "experimental_run",
    "source_run_label",
    "source_relpath",
    "source_json_relpath",
    "log_relpath",
    "onset_mat_relpath",
    "sdc_mode",
    "fmap_role",
    "approval_status",
    "source_size_bytes",
    "source_sha256",
    "source_json_size_bytes",
    "source_json_sha256",
    "onset_mat_size_bytes",
    "onset_mat_sha256",
    "notes",
]

VALID_SDC = {"pepolar", "gre", "syn"}
VALID_FMAP_ROLES = {
    "pepolar": ("epi_ap", "epi_pa"),
    "gre": ("magnitude1", "magnitude2", "phasediff"),
    "syn": (),
}
GLOB_CHARS = set("*?[]")
RUN_RE = re.compile(r"\bRUN\s*0*(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ExpectedRun:
    subject: int
    session: int
    experimental_run: int
    source_run_label: int
    source_dir: str
    log_relpath: str
    sdc_mode: str

    @property
    def asset_id(self) -> str:
        return f"sub-{self.subject:02d}_ses-{self.session:02d}_run-{self.experimental_run:02d}_bold"


@dataclass(frozen=True)
class ExpectedAsset:
    asset_id: str
    asset_type: str
    subject: int
    session: int
    experimental_run: int | None
    source_run_label: int | None
    source_dir: str
    log_relpath: str
    sdc_mode: str
    fmap_role: str


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    asset_id: str
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "asset_id": self.asset_id,
            "message": self.message,
        }


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise ValueError("Unsupported or missing cohort config schema_version")
    remaining = [int(value) for value in config["cohort"]["remaining_subjects"]]
    expected = [1, 2, 7, 9, *range(11, 32)]
    if remaining != expected:
        raise ValueError(f"Remaining cohort must be exactly {expected}; found {remaining}")
    if sorted(map(int, config["subjects"])) != expected:
        raise ValueError("Config subject keys do not exactly match the remaining cohort")
    identity = config.get("input_identities", {}).get("stimuli_600trials_csv", {})
    required_identity = {"path", "size_bytes", "sha256"}
    if set(identity) != required_identity:
        raise ValueError(
            "input_identities.stimuli_600trials_csv must contain exactly "
            f"{sorted(required_identity)}"
        )
    if identity["path"] != config["paths"]["stimuli_csv"]:
        raise ValueError("Frozen stimuli path must exactly equal paths.stimuli_csv")
    if int(identity["size_bytes"]) <= 0 or not re.fullmatch(
        r"[0-9a-f]{64}", str(identity["sha256"]).casefold()
    ):
        raise ValueError("Frozen stimuli size/SHA-256 is invalid")
    geometry = config.get("task", {}).get("acquisition_geometry", {})
    if geometry.get("native_nifti_voxel_spacing_mm") != [1.7966, 1.7966, 2.25]:
        raise ValueError("Native acquisition spacing must remain frozen at 1.7966x1.7966x2.25 mm")
    flags = config.get("task", {}).get("raw_bold_volume_qc", {}).get("flagged_runs", [])
    observed_flags = {
        str(flag.get("asset_id")): int(flag.get("observed_volumes", -1)) for flag in flags
    }
    if observed_flags != {
        "sub-29_ses-01_run-06_bold": 216,
        "sub-30_ses-01_run-06_bold": 210,
    }:
        raise ValueError("Raw BOLD volume QC flags must pin Sub29/Sub30 run 6")
    return config


def _subject_subset(config: Mapping, subjects: Iterable[int] | None) -> list[int]:
    cohort = [int(value) for value in config["cohort"]["remaining_subjects"]]
    if subjects is None:
        return cohort
    requested = sorted(set(int(value) for value in subjects))
    unknown = sorted(set(requested) - set(cohort))
    if unknown:
        raise ValueError(f"Subjects outside the remaining cohort: {unknown}")
    return requested


def iter_expected_runs(config: Mapping, subjects: Iterable[int] | None = None) -> Iterator[ExpectedRun]:
    for subject in _subject_subset(config, subjects):
        subject_cfg = config["subjects"][str(subject)]
        log_template = str(subject_cfg["log_filename_template"])
        for session_cfg in subject_cfg["sessions"]:
            session = int(session_cfg["session"])
            source_dir = str(session_cfg["source_dir"])
            sdc_mode = str(session_cfg["sdc_mode"]).lower()
            if sdc_mode not in VALID_SDC:
                raise ValueError(f"Invalid SDC mode for Sub{subject} ses{session}: {sdc_mode}")
            overrides = {
                int(key): int(value)
                for key, value in session_cfg.get("source_run_overrides", {}).items()
            }
            for experimental_run in map(int, session_cfg["experimental_runs"]):
                yield ExpectedRun(
                    subject=subject,
                    session=session,
                    experimental_run=experimental_run,
                    source_run_label=overrides.get(experimental_run, experimental_run),
                    source_dir=source_dir,
                    log_relpath=(
                        PurePosixPath(f"Sub{subject}")
                        / "LogFiles"
                        / log_template.format(run=experimental_run)
                    ).as_posix(),
                    sdc_mode=sdc_mode,
                )


def iter_expected_assets(config: Mapping, subjects: Iterable[int] | None = None) -> Iterator[ExpectedAsset]:
    runs = list(iter_expected_runs(config, subjects))
    for run in runs:
        yield ExpectedAsset(
            asset_id=run.asset_id,
            asset_type="bold",
            subject=run.subject,
            session=run.session,
            experimental_run=run.experimental_run,
            source_run_label=run.source_run_label,
            source_dir=run.source_dir,
            log_relpath=run.log_relpath,
            sdc_mode=run.sdc_mode,
            fmap_role="",
        )
    sessions: dict[tuple[int, int], ExpectedRun] = {}
    for run in runs:
        sessions.setdefault((run.subject, run.session), run)
    for (subject, session), exemplar in sorted(sessions.items()):
        yield ExpectedAsset(
            asset_id=f"sub-{subject:02d}_ses-{session:02d}_T1w",
            asset_type="t1w",
            subject=subject,
            session=session,
            experimental_run=None,
            source_run_label=None,
            source_dir=exemplar.source_dir,
            log_relpath="",
            sdc_mode=exemplar.sdc_mode,
            fmap_role="",
        )
        for role in VALID_FMAP_ROLES[exemplar.sdc_mode]:
            yield ExpectedAsset(
                asset_id=f"sub-{subject:02d}_ses-{session:02d}_{role}",
                asset_type="fmap",
                subject=subject,
                session=session,
                experimental_run=None,
                source_run_label=None,
                source_dir=exemplar.source_dir,
                log_relpath="",
                sdc_mode=exemplar.sdc_mode,
                fmap_role=role,
            )


def sidecar_for_nifti(path: Path) -> Path:
    if not path.name.endswith(".nii.gz"):
        raise ValueError(f"Expected .nii.gz source: {path}")
    return path.with_name(path.name[:-7] + ".json")


def read_verified_bytes(
    path: Path,
    expected_size: int | str,
    expected_sha256: str,
    *,
    label: str,
) -> bytes:
    """Read one frozen input once and verify the bytes that will be consumed."""
    if expected_size == "" or not expected_sha256:
        raise RuntimeError(f"Missing frozen size/SHA-256 for {label}: {path}")
    payload = path.read_bytes()
    observed_size = len(payload)
    wanted_size = int(expected_size)
    if observed_size != wanted_size:
        raise RuntimeError(
            f"{label} size changed: frozen {wanted_size}, current {observed_size}: {path}"
        )
    observed_hash = hashlib.sha256(payload).hexdigest()
    if observed_hash.casefold() != str(expected_sha256).casefold():
        raise RuntimeError(f"{label} SHA-256 changed: {path}")
    return payload


def verify_file_identity(
    path: Path,
    expected_size: int | str,
    expected_sha256: str,
    *,
    label: str,
) -> None:
    if expected_size == "" or not expected_sha256:
        raise RuntimeError(f"Missing frozen size/SHA-256 for {label}: {path}")
    observed_size = path.stat().st_size
    wanted_size = int(expected_size)
    if observed_size != wanted_size:
        raise RuntimeError(
            f"{label} size changed: frozen {wanted_size}, current {observed_size}: {path}"
        )
    observed_hash = sha256_file(path)
    if observed_hash.casefold() != str(expected_sha256).casefold():
        raise RuntimeError(f"{label} SHA-256 changed: {path}")


def read_json(
    path: Path,
    *,
    expected_size: int | str | None = None,
    expected_sha256: str | None = None,
    label: str = "JSON sidecar",
) -> dict:
    if expected_size is None and expected_sha256 is None:
        payload = path.read_bytes()
    elif expected_size is None or expected_sha256 is None:
        raise ValueError("Both expected_size and expected_sha256 are required together")
    else:
        payload = read_verified_bytes(
            path,
            expected_size,
            expected_sha256,
            label=label,
        )
    value = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON sidecar is not an object: {path}")
    return value


def metadata_text(metadata: Mapping) -> str:
    parts = [
        str(metadata.get("SeriesDescription", "")),
        str(metadata.get("ProtocolName", "")),
        " ".join(map(str, metadata.get("ImageType", []) or [])),
        " ".join(map(str, metadata.get("ImageTypeText", []) or [])),
    ]
    return " | ".join(parts)


def is_moco(metadata: Mapping, filename: str = "") -> bool:
    text = metadata_text(metadata).casefold()
    return (
        "mocoseries" in text
        or bool(re.search(r"(^|[^a-z])moco([^a-z]|$)", text))
        or bool(re.search(r"_moco(?:_|\.)", filename, flags=re.IGNORECASE))
    )


def metadata_run_label(metadata: Mapping) -> int | None:
    labels: set[int] = set()
    for key in ("SeriesDescription", "ProtocolName"):
        match = RUN_RE.search(str(metadata.get(key, "")))
        if match:
            labels.add(int(match.group(1)))
    if len(labels) > 1:
        raise ValueError(f"Conflicting run labels in JSON metadata: {sorted(labels)}")
    return next(iter(labels), None)


def classify_source(path: Path, metadata: Mapping) -> tuple[str, str, str]:
    """Return asset type, fmap role, and exclusion reason."""
    name = path.name.casefold()
    text = metadata_text(metadata).casefold()
    if is_moco(metadata, path.name):
        if "bold" in text or "run" in text or "bold" in name:
            return "bold", "", "moco_series"
        if "fmap" in text or "distmap" in text or "fmap" in name or "distmap" in name:
            return "fmap", "", "moco_series"
        return "other", "", "moco_series"
    if ("bold" in text or "bold" in name) and metadata_run_label(metadata) is not None:
        return "bold", "", ""
    if "mprage" in text or "mprage" in name:
        return "t1w", "", ""
    if any(token in text or token in name for token in ("distmap", "ap-fmap", "pa-fmap")):
        if "ap-fmap" in text or "ap-fmap" in name or "distmap_ap" in name:
            return "fmap", "epi_ap", ""
        if "pa-fmap" in text or "pa-fmap" in name or "distmap_pa" in name:
            return "fmap", "epi_pa", ""
        return "fmap", "", "unknown_pepolar_direction"
    if "gre_field_mapping" in text or "gre_field_mapping" in name:
        image_text = " ".join(
            [
                *map(str, metadata.get("ImageType", []) or []),
                *map(str, metadata.get("ImageTypeText", []) or []),
            ]
        ).casefold()
        if "phase" in image_text or name.endswith("_ph.nii.gz"):
            return "fmap", "phasediff", ""
        if re.search(r"_e1\.nii\.gz$", name):
            return "fmap", "magnitude1", ""
        if re.search(r"_e2\.nii\.gz$", name):
            return "fmap", "magnitude2", ""
        return "fmap", "", "unknown_gre_component"
    return "other", "", "not_a_required_asset"


def is_norm_gre(metadata: Mapping) -> bool:
    values = [
        *map(str, metadata.get("ImageType", []) or []),
        *map(str, metadata.get("ImageTypeText", []) or []),
    ]
    return any(value.casefold() == "norm" for value in values)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_selection(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != SELECTION_COLUMNS:
            raise ValueError(
                f"Selection columns must be exactly {SELECTION_COLUMNS}; found {reader.fieldnames}"
            )
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def safe_relative_path(root: Path, relpath: str) -> tuple[Path | None, str | None]:
    if not relpath:
        return None, "empty relative path"
    if any(char in relpath for char in GLOB_CHARS):
        return None, "glob characters are forbidden"
    pure = PurePosixPath(relpath.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        return None, "path must be relative and cannot contain '..'"
    candidate = root.joinpath(*pure.parts)
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return None, "path resolves outside configured root"
    return candidate, None


def _safe_source_path(raw_root: Path, relpath: str) -> tuple[Path | None, str | None]:
    return safe_relative_path(raw_root, relpath)


def _int_field(row: Mapping[str, str], key: str) -> int | None:
    value = row.get(key, "")
    return int(value) if value != "" else None


def expected_row_dict(asset: ExpectedAsset) -> dict[str, object]:
    return {
        "asset_type": asset.asset_type,
        "subject": asset.subject,
        "session": asset.session,
        "experimental_run": asset.experimental_run,
        "source_run_label": asset.source_run_label,
        "log_relpath": asset.log_relpath,
        "onset_mat_relpath": asset.log_relpath,
        "sdc_mode": asset.sdc_mode,
        "fmap_role": asset.fmap_role,
    }


def stimuli_identity(config: Mapping) -> Mapping:
    try:
        identity = config["input_identities"]["stimuli_600trials_csv"]
        size = int(identity["size_bytes"])
        sha256 = str(identity["sha256"])
        frozen_path = str(identity["path"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing/invalid frozen stimuli identity: {exc}") from exc
    if frozen_path != str(config["paths"]["stimuli_csv"]):
        raise RuntimeError("Frozen stimuli path differs from paths.stimuli_csv")
    if size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", sha256.casefold()):
        raise RuntimeError("Frozen stimuli size/SHA-256 is invalid")
    return {"path": frozen_path, "size_bytes": size, "sha256": sha256.casefold()}


def verify_stimuli_identity(config: Mapping, stimuli_csv: Path) -> None:
    identity = stimuli_identity(config)
    verify_file_identity(
        stimuli_csv,
        identity["size_bytes"],
        identity["sha256"],
        label="stimuli_600trials.csv",
    )


def _identity_issue(
    issues: list[Issue],
    *,
    asset_id: str,
    path: Path,
    size_value: str,
    hash_value: str,
    size_field: str,
    hash_field: str,
    label: str,
    require_hash: bool,
    verify_hashes: bool,
) -> None:
    if not path.is_file():
        issues.append(Issue("error", f"missing_{label}", asset_id, str(path)))
        return
    if size_value == "":
        issues.append(Issue("error", f"missing_{size_field}", asset_id, f"{size_field} is required"))
    else:
        try:
            frozen_size = int(size_value)
        except ValueError:
            issues.append(Issue("error", f"invalid_{size_field}", asset_id, size_value))
        else:
            observed_size = path.stat().st_size
            if frozen_size != observed_size:
                issues.append(
                    Issue(
                        "error",
                        f"{label}_size_changed",
                        asset_id,
                        f"Manifest size {frozen_size} != current size {observed_size}: {path}",
                    )
                )
    if not hash_value:
        issues.append(
            Issue(
                "error" if require_hash else "warning",
                f"missing_{hash_field}",
                asset_id,
                f"{hash_field} is required for production",
            )
        )
    elif not re.fullmatch(r"[0-9a-f]{64}", hash_value.casefold()):
        issues.append(Issue("error", f"invalid_{hash_field}", asset_id, hash_value))
    elif verify_hashes:
        observed_hash = sha256_file(path)
        if observed_hash.casefold() != hash_value.casefold():
            issues.append(
                Issue("error", f"{label}_hash_changed", asset_id, f"SHA-256 mismatch: {path}")
            )


def _metadata_matches_asset(
    path: Path, metadata: Mapping, expected: ExpectedAsset
) -> list[str]:
    problems: list[str] = []
    asset_type, fmap_role, exclusion = classify_source(path, metadata)
    if exclusion:
        problems.append(f"source classifier rejected asset: {exclusion}")
    if asset_type != expected.asset_type:
        problems.append(f"metadata classifies source as {asset_type}, expected {expected.asset_type}")
    if expected.asset_type == "fmap" and fmap_role != expected.fmap_role:
        problems.append(f"metadata fmap role is {fmap_role!r}, expected {expected.fmap_role!r}")
    if expected.asset_type == "bold":
        try:
            observed = metadata_run_label(metadata)
        except ValueError as exc:
            problems.append(str(exc))
        else:
            if observed != expected.source_run_label:
                problems.append(
                    f"JSON scanner run label is {observed}, expected {expected.source_run_label}; "
                    f"logical experimental run remains {expected.experimental_run}"
                )
    return problems


def audit_selection(
    config: Mapping,
    selection_path: Path,
    raw_root: Path,
    log_root: Path,
    stimuli_csv: Path,
    subjects: Iterable[int] | None = None,
    *,
    require_approved: bool = True,
    check_onsets: bool = True,
    verify_hashes: bool = False,
) -> tuple[list[dict[str, str]], list[Issue]]:
    expected_assets = {asset.asset_id: asset for asset in iter_expected_assets(config, subjects)}
    rows = read_selection(selection_path)
    issues: list[Issue] = []
    try:
        frozen_stimuli = stimuli_identity(config)
    except Exception as exc:  # noqa: BLE001 - surface malformed provenance
        issues.append(Issue("error", "invalid_stimuli_freeze", "stimuli_600trials.csv", str(exc)))
    else:
        _identity_issue(
            issues,
            asset_id="stimuli_600trials.csv",
            path=stimuli_csv,
            size_value=str(frozen_stimuli["size_bytes"]),
            hash_value=str(frozen_stimuli["sha256"]),
            size_field="stimuli_size_bytes",
            hash_field="stimuli_sha256",
            label="stimuli_csv",
            require_hash=True,
            verify_hashes=verify_hashes,
        )
    selected_subjects = set(_subject_subset(config, subjects))
    rows = [row for row in rows if int(row["subject"] or -1) in selected_subjects]

    by_id: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_id.setdefault(row["asset_id"], []).append(row)
    for asset_id in sorted(expected_assets):
        count = len(by_id.get(asset_id, []))
        if count != 1:
            issues.append(Issue("error", "expected_asset_count", asset_id, f"Expected one row; found {count}"))
    for asset_id in sorted(set(by_id) - set(expected_assets)):
        issues.append(Issue("error", "unexpected_asset", asset_id, "Asset is not in the frozen cohort plan"))

    used_sources: dict[str, str] = {}
    metadata_by_id: dict[str, Mapping] = {}
    for asset_id, expected in expected_assets.items():
        if len(by_id.get(asset_id, [])) != 1:
            continue
        row = by_id[asset_id][0]
        if require_approved and row["approval_status"] != "approved":
            issues.append(
                Issue(
                    "error",
                    "not_approved",
                    asset_id,
                    f"approval_status must be 'approved'; found {row['approval_status']!r}",
                )
            )
        observed_fields = {
            "asset_type": row["asset_type"],
            "subject": _int_field(row, "subject"),
            "session": _int_field(row, "session"),
            "experimental_run": _int_field(row, "experimental_run"),
            "source_run_label": _int_field(row, "source_run_label"),
            "log_relpath": row["log_relpath"],
            "onset_mat_relpath": row["onset_mat_relpath"],
            "sdc_mode": row["sdc_mode"],
            "fmap_role": row["fmap_role"],
        }
        for field, expected_value in expected_row_dict(expected).items():
            if observed_fields[field] != expected_value:
                issues.append(
                    Issue(
                        "error",
                        "plan_mismatch",
                        asset_id,
                        f"{field}={observed_fields[field]!r}; frozen plan requires {expected_value!r}",
                    )
                )
        source, path_problem = _safe_source_path(raw_root, row["source_relpath"])
        if path_problem:
            issues.append(Issue("error", "unsafe_source_path", asset_id, path_problem))
            continue
        assert source is not None
        rel_parts = PurePosixPath(row["source_relpath"].replace("\\", "/")).parts
        expected_dir_parts = PurePosixPath(expected.source_dir).parts
        if tuple(part.casefold() for part in rel_parts[: len(expected_dir_parts)]) != tuple(
            part.casefold() for part in expected_dir_parts
        ):
            issues.append(
                Issue(
                    "error",
                    "wrong_acquisition_directory",
                    asset_id,
                    f"Source must be within configured same-session directory {expected.source_dir}",
                )
            )
        source_key = str(source.resolve()).casefold()
        if source_key in used_sources:
            issues.append(
                Issue(
                    "error",
                    "duplicate_source",
                    asset_id,
                    f"Source already assigned to {used_sources[source_key]}",
                )
            )
        used_sources[source_key] = asset_id
        _identity_issue(
            issues,
            asset_id=asset_id,
            path=source,
            size_value=row["source_size_bytes"],
            hash_value=row["source_sha256"],
            size_field="source_size_bytes",
            hash_field="source_sha256",
            label="nifti",
            require_hash=require_approved,
            verify_hashes=verify_hashes,
        )

        sidecar = sidecar_for_nifti(source)
        sidecar_path, sidecar_problem = safe_relative_path(raw_root, row["source_json_relpath"])
        if sidecar_problem:
            issues.append(Issue("error", "unsafe_source_json_path", asset_id, sidecar_problem))
            sidecar_path = None
        elif sidecar_path is not None and sidecar_path.resolve() != sidecar.resolve():
            issues.append(
                Issue(
                    "error",
                    "source_json_path_mismatch",
                    asset_id,
                    f"Manifest JSON {sidecar_path} is not the NIfTI sidecar {sidecar}",
                )
            )
        if sidecar_path is not None:
            _identity_issue(
                issues,
                asset_id=asset_id,
                path=sidecar_path,
                size_value=row["source_json_size_bytes"],
                hash_value=row["source_json_sha256"],
                size_field="source_json_size_bytes",
                hash_field="source_json_sha256",
                label="source_json",
                require_hash=True,
                verify_hashes=verify_hashes,
            )
            if sidecar_path.is_file():
                try:
                    metadata = read_json(sidecar_path)
                except Exception as exc:  # noqa: BLE001 - record malformed metadata
                    issues.append(Issue("error", "invalid_sidecar", asset_id, str(exc)))
                else:
                    metadata_by_id[asset_id] = metadata
                    for problem in _metadata_matches_asset(source, metadata, expected):
                        issues.append(Issue("error", "metadata_mismatch", asset_id, problem))

        onset_fields = (
            row["onset_mat_relpath"],
            row["onset_mat_size_bytes"],
            row["onset_mat_sha256"],
        )
        if expected.asset_type == "bold":
            onset_path, onset_problem = safe_relative_path(log_root, row["onset_mat_relpath"])
            if onset_problem:
                issues.append(Issue("error", "unsafe_onset_mat_path", asset_id, onset_problem))
            else:
                assert onset_path is not None
                _identity_issue(
                    issues,
                    asset_id=asset_id,
                    path=onset_path,
                    size_value=row["onset_mat_size_bytes"],
                    hash_value=row["onset_mat_sha256"],
                    size_field="onset_mat_size_bytes",
                    hash_field="onset_mat_sha256",
                    label="onset_mat",
                    require_hash=True,
                    verify_hashes=verify_hashes,
                )
        elif any(onset_fields):
            issues.append(
                Issue(
                    "error",
                    "unexpected_onset_identity",
                    asset_id,
                    "Non-BOLD rows must leave all onset MAT identity fields blank",
                )
            )

    # GRE component semantics are checked jointly; loose filename matching is insufficient.
    sessions: dict[tuple[int, int], list[ExpectedAsset]] = {}
    for asset in expected_assets.values():
        sessions.setdefault((asset.subject, asset.session), []).append(asset)
    for (subject, session), assets in sessions.items():
        sdc_mode = assets[0].sdc_mode
        fmap_assets = {asset.fmap_role: asset for asset in assets if asset.asset_type == "fmap"}
        if sdc_mode == "gre" and all(asset.asset_id in metadata_by_id for asset in fmap_assets.values()):
            magnitude1 = metadata_by_id[fmap_assets["magnitude1"].asset_id]
            magnitude2 = metadata_by_id[fmap_assets["magnitude2"].asset_id]
            phase = metadata_by_id[fmap_assets["phasediff"].asset_id]
            try:
                echo1 = float(magnitude1["EchoTime"])
                echo2 = float(magnitude2["EchoTime"])
                phase_echo = float(phase["EchoTime"])
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(
                    Issue(
                        "error",
                        "gre_echo_metadata",
                        f"sub-{subject:02d}_ses-{session:02d}",
                        f"Missing/invalid GRE EchoTime metadata: {exc}",
                    )
                )
            else:
                if not (0 < echo1 < echo2) or not np.isclose(phase_echo, echo2):
                    issues.append(
                        Issue(
                            "error",
                            "gre_echo_metadata",
                            f"sub-{subject:02d}_ses-{session:02d}",
                            f"Expected 0 < EchoTime1 < EchoTime2 and phase EchoTime=EchoTime2; "
                            f"found {echo1}, {echo2}, {phase_echo}",
                        )
                    )

    if check_onsets:
        onset_rows = audit_all_onsets(config, log_root, stimuli_csv, subjects)
        for row in onset_rows:
            if row["status"] != "passed":
                issues.append(
                    Issue(
                        "error",
                        "onset_validation_failed",
                        row["asset_id"],
                        row["message"],
                    )
                )
    return rows, issues


def load_stimuli(
    stimuli_csv: Path,
    *,
    expected_size: int | str | None = None,
    expected_sha256: str | None = None,
) -> pd.DataFrame:
    if expected_size is None and expected_sha256 is None:
        stimuli = pd.read_csv(stimuli_csv)
    elif expected_size is None or expected_sha256 is None:
        raise ValueError("Both expected_size and expected_sha256 are required together")
    else:
        payload = read_verified_bytes(
            stimuli_csv,
            expected_size,
            expected_sha256,
            label="stimuli_600trials.csv",
        )
        stimuli = pd.read_csv(BytesIO(payload))
    required = {"run", "OriginalOrder", "img", "group", "AI", "emotion_type"}
    missing = sorted(required - set(stimuli.columns))
    if missing:
        raise ValueError(f"Stimulus manifest missing columns: {missing}")
    return stimuli


def extract_events(
    log_path: Path,
    expected: pd.DataFrame,
    *,
    expected_size: int | str | None = None,
    expected_sha256: str | None = None,
) -> pd.DataFrame:
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    if expected_size is None and expected_sha256 is None:
        data = loadmat(log_path, simplify_cells=True)
    elif expected_size is None or expected_sha256 is None:
        raise ValueError("Both expected_size and expected_sha256 are required together")
    else:
        payload = read_verified_bytes(
            log_path,
            expected_size,
            expected_sha256,
            label="onset MAT",
        )
        data = loadmat(BytesIO(payload), simplify_cells=True)
    if "dataLog" not in data:
        raise RuntimeError(f"No dataLog in {log_path}")
    data_log = data["dataLog"]
    rows = [row for row in data_log[1:] if len(row) > 3 and isinstance(row[1], str)]
    fix_rows = [row for row in rows if row[1] == "FixOn"]
    stim_rows = [row for row in rows if row[1] == "Stim on"]
    if len(fix_rows) != 1 or len(stim_rows) != 60:
        raise RuntimeError(
            f"Expected one FixOn and 60 Stim on rows; found {len(fix_rows)} and {len(stim_rows)}"
        )
    expected = expected.sort_values("OriginalOrder")
    order = pd.to_numeric(expected["OriginalOrder"], errors="coerce")
    if len(expected) != 60 or order.isna().any() or order.duplicated().any():
        raise RuntimeError("Stimulus manifest must contain 60 unique finite OriginalOrder values")
    raw_names = [str(row[2]) for row in stim_rows]
    expected_names = expected["img"].astype(str).tolist()
    if raw_names != expected_names:
        mismatch = next(
            (index for index, pair in enumerate(zip(raw_names, expected_names)) if pair[0] != pair[1]),
            None,
        )
        raise RuntimeError(f"Stimulus order/name mismatch at trial {mismatch}")
    fix_onset = float(fix_rows[0][3])
    onsets = np.asarray([float(row[3]) - fix_onset for row in stim_rows], dtype=np.float64)
    if not np.all(np.isfinite(onsets)) or not np.all(np.diff(onsets) > 0):
        raise RuntimeError("Onsets must be finite and strictly increasing")
    events = expected.copy()
    events.insert(0, "onset", onsets)
    events.insert(1, "duration", 3.0)
    events["trial_type"] = events["group"]
    events["image_id"] = events["img"].str.replace(r"\.jpg$", "", regex=True)
    events["source"] = np.where(events["AI"].str.upper().eq("YES"), "ai", "natural")
    events["valence"] = events["emotion_type"]
    return events[
        ["onset", "duration", "trial_type", "image_id", "source", "valence", "OriginalOrder"]
    ]


def audit_all_onsets(
    config: Mapping,
    log_root: Path,
    stimuli_csv: Path,
    subjects: Iterable[int] | None = None,
) -> list[dict[str, object]]:
    frozen_stimuli = stimuli_identity(config)
    stimuli = load_stimuli(
        stimuli_csv,
        expected_size=frozen_stimuli["size_bytes"],
        expected_sha256=frozen_stimuli["sha256"],
    )
    results: list[dict[str, object]] = []
    for run in iter_expected_runs(config, subjects):
        log_path = log_root.joinpath(*PurePosixPath(run.log_relpath).parts)
        try:
            log_size = log_path.stat().st_size
            log_hash = sha256_file(log_path)
            events = extract_events(
                log_path,
                stimuli[stimuli["run"] == run.experimental_run],
                expected_size=log_size,
                expected_sha256=log_hash,
            )
        except Exception as exc:  # noqa: BLE001 - record every failed gate
            results.append(
                {
                    "asset_id": run.asset_id,
                    "subject": run.subject,
                    "session": run.session,
                    "experimental_run": run.experimental_run,
                    "log_relpath": run.log_relpath,
                    "onset_mat_size_bytes": "",
                    "onset_mat_sha256": "",
                    "status": "failed",
                    "n_events": "",
                    "first_onset": "",
                    "last_onset": "",
                    "message": str(exc),
                }
            )
        else:
            results.append(
                {
                    "asset_id": run.asset_id,
                    "subject": run.subject,
                    "session": run.session,
                    "experimental_run": run.experimental_run,
                    "log_relpath": run.log_relpath,
                    "onset_mat_size_bytes": log_size,
                    "onset_mat_sha256": log_hash,
                    "status": "passed",
                    "n_events": len(events),
                    "first_onset": float(events["onset"].iloc[0]),
                    "last_onset": float(events["onset"].iloc[-1]),
                    "message": "",
                }
            )
    return results


def bids_bold_relpath(run: ExpectedRun) -> Path:
    return (
        Path(f"sub-{run.subject:02d}")
        / f"ses-{run.session:02d}"
        / "func"
        / f"sub-{run.subject:02d}_ses-{run.session:02d}_task-iaps_run-{run.experimental_run:02d}_bold.nii.gz"
    )


def write_tsv(path: Path, rows: Sequence[Mapping], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
