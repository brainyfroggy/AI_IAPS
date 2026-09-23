#!/usr/bin/env python3
"""Shared fail-closed helpers for the frozen full-cohort analysis.

This module is deliberately independent of the expensive fMRI libraries so
that cohort/configuration and tabular-completeness gates can be unit tested on
any audit machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


COHORT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = COHORT_ROOT / "ANALYSIS_FREEZE.json"
DEFAULT_COHORT_CONFIG = COHORT_ROOT / "config" / "cohort.json"

PIPELINES = (
    "legacy_lsa_8mm",
    "glmsingle_typed_0mm",
    "glmsingle_typed_3mm",
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

CONTRASTS = (
    "within_natural_pleasant_vs_neutral",
    "within_ai_pleasant_vs_neutral",
    "within_natural_unpleasant_vs_neutral",
    "within_ai_unpleasant_vs_neutral",
    "train_natural_pleasant_vs_neutral_test_ai",
    "train_ai_pleasant_vs_neutral_test_natural",
    "train_natural_unpleasant_vs_neutral_test_ai",
    "train_ai_unpleasant_vs_neutral_test_natural",
)

WITHIN_CONTRASTS = frozenset(CONTRASTS[:4])
HISTORICAL_METHOD = "historical_avg_random"
RUNWISE_METHOD = "runwise_loro_means"


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_freeze(path: Path = DEFAULT_FREEZE) -> dict:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    included = [int(value) for value in freeze.get("included_subjects", [])]
    if len(included) != 28 or len(set(included)) != 28:
        raise ValueError("ANALYSIS_FREEZE must contain 28 unique included subjects")
    excluded = {int(value) for value in freeze.get("excluded_subjects", {})}
    if excluded.intersection(included):
        raise ValueError("Included and excluded subjects overlap")
    if tuple(freeze.get("erp_decoding", {}).get("roi_order", [])) != ROI_ORDER:
        raise ValueError("Frozen ROI order differs from the 17-region analysis contract")
    if tuple(freeze.get("contrasts", [])) != CONTRASTS:
        raise ValueError("Frozen contrast order differs from the analysis contract")
    frozen_pipelines = tuple(item.get("id") for item in freeze.get("pipelines", []))
    if frozen_pipelines != PIPELINES:
        raise ValueError("Frozen pipeline order differs from the analysis contract")
    return freeze


def load_cohort_config(path: Path = DEFAULT_COHORT_CONFIG) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    included = [int(value) for value in config.get("cohort", {}).get("included_subjects", [])]
    if len(included) != 28 or len(set(included)) != 28:
        raise ValueError("cohort.json must contain 28 unique included subjects")
    return config


def validate_subject_in_freeze(subject: int, freeze: Mapping) -> None:
    included = {int(value) for value in freeze["included_subjects"]}
    if subject not in included:
        raise ValueError(f"Sub{subject} is not in the frozen included cohort")


def subject_label(subject: int) -> str:
    return f"Sub{int(subject)}"


def bids_subject_label(subject: int) -> str:
    return f"sub-{int(subject):02d}"


def expected_session_indicators(
    cohort_config: Mapping, subject: int
) -> list[int] | None:
    """Return the frozen acquisition-session number for logical runs 1--10.

    Pilot subjects may be absent from the new-batch source config; in that case
    the caller must derive the values from the BIDS ``ses-`` entities.
    """

    entry = cohort_config.get("subjects", {}).get(str(subject))
    if entry is None:
        return None
    mapping: dict[int, int] = {}
    for session in entry.get("sessions", []):
        session_number = int(session["session"])
        for run in session.get("experimental_runs", []):
            run_number = int(run)
            if run_number in mapping:
                raise ValueError(f"Sub{subject} run {run_number} is assigned twice")
            mapping[run_number] = session_number
    if sorted(mapping) != list(range(1, 11)):
        raise ValueError(
            f"Sub{subject} session config must map logical runs 1-10; "
            f"found {sorted(mapping)}"
        )
    return [mapping[run] for run in range(1, 11)]


def session_from_bids_path(path: Path) -> int | None:
    match = re.search(r"(?:^|[_\\/])ses-(\d+)(?:[_\\/]|$)", str(path))
    return int(match.group(1)) if match else None


def resolve_session_indicators(
    bold_paths: Sequence[Path],
    subject: int,
    cohort_config: Mapping,
) -> tuple[list[int], str]:
    """Resolve run-order session indicators and cross-check config vs BIDS."""

    if len(bold_paths) != 10:
        raise ValueError(f"Sub{subject} requires exactly 10 BOLD paths")
    bids_values = [session_from_bids_path(path) for path in bold_paths]
    configured = expected_session_indicators(cohort_config, subject)
    if all(value is not None for value in bids_values):
        observed = [int(value) for value in bids_values]
        if configured is not None and observed != configured:
            raise ValueError(
                f"Sub{subject} BIDS sessions {observed} disagree with frozen "
                f"cohort config {configured}"
            )
        return observed, "BIDS ses entities cross-checked against cohort config" if configured else "BIDS ses entities"
    if any(value is not None for value in bids_values):
        raise ValueError(f"Sub{subject} mixes BOLD paths with and without ses entities")
    if configured is not None:
        return configured, "cohort config (BIDS paths have no ses entity)"
    # A session-less BIDS subject can only represent one acquisition session.
    return [1] * 10, "single-session BIDS fallback"


def quick_file_signature(path: Path, edge_bytes: int = 1024 * 1024) -> dict:
    """Audit signature without hashing an entire multi-gigabyte derivative."""

    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(edge_bytes))
        if stat.st_size > edge_bytes:
            handle.seek(max(0, stat.st_size - edge_bytes))
            digest.update(handle.read(edge_bytes))
    result["sha256_first_and_last_1MiB"] = digest.hexdigest()
    return result


def directory_file_set_signature(paths: Iterable[Path]) -> dict:
    rows = []
    for path in sorted((Path(value) for value in paths), key=lambda item: item.name):
        stat = path.stat()
        rows.append((path.name, int(stat.st_size), int(stat.st_mtime_ns)))
    return {"n_files": len(rows), "entries_sha256": stable_json_sha256(rows)}


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} lacks required columns: {missing}")


def validate_subject_result_rows(
    frame: pd.DataFrame, subject: int, method: str
) -> None:
    name = f"{subject_label(subject)} {method} subject results"
    required = (
        "subject",
        "pipeline",
        "analysis_kind",
        "contrast",
        "contrast_label",
        "roi",
        "n_voxels",
        "accuracy",
    )
    _require_columns(frame, required, name)
    if "erp_method" in frame and set(frame["erp_method"]) != {method}:
        raise ValueError(f"{name} has the wrong erp_method")
    expected_subject = subject_label(subject)
    if set(frame["subject"]) != {expected_subject}:
        raise ValueError(f"{name} contains a different subject")
    keys = ["subject", "pipeline", "contrast", "roi"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{name} contains duplicate result keys")
    expected = {
        (expected_subject, pipeline, contrast, roi)
        for pipeline in PIPELINES
        for contrast in CONTRASTS
        for roi in ROI_ORDER
    }
    observed = set(frame[keys].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(
            f"{name} key mismatch: missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}"
        )
    expected_kind = {
        contrast: ("within" if contrast in WITHIN_CONTRASTS else "cross")
        for contrast in CONTRASTS
    }
    if any(
        kind != expected_kind[contrast]
        for contrast, kind in frame[["contrast", "analysis_kind"]].itertuples(index=False)
    ):
        raise ValueError(f"{name} contains an incorrect analysis_kind")
    if not pd.to_numeric(frame["accuracy"], errors="coerce").between(0, 1).all():
        raise ValueError(f"{name} contains a nonfinite or out-of-range accuracy")
    if not (pd.to_numeric(frame["n_voxels"], errors="coerce") >= 10).all():
        raise ValueError(f"{name} contains fewer than 10 ROI voxels")
    voxel_counts = frame.groupby(["pipeline", "roi"], sort=False)["n_voxels"].nunique()
    if not (voxel_counts == 1).all():
        raise ValueError(f"{name} has inconsistent voxel counts across contrasts")


def validate_fold_result_rows(
    frame: pd.DataFrame, subject: int, method: str
) -> None:
    name = f"{subject_label(subject)} {method} fold results"
    required = ("subject", "pipeline", "contrast", "roi", "accuracy")
    _require_columns(frame, required, name)
    if "erp_method" in frame and set(frame["erp_method"]) != {method}:
        raise ValueError(f"{name} has the wrong erp_method")
    if set(frame["subject"]) != {subject_label(subject)}:
        raise ValueError(f"{name} contains a different subject")
    if not pd.to_numeric(frame["accuracy"], errors="coerce").between(0, 1).all():
        raise ValueError(f"{name} contains a nonfinite or out-of-range accuracy")

    group_keys = ["subject", "pipeline", "contrast", "roi"]
    counts = frame.groupby(group_keys, sort=False).size()
    expected_groups = {
        (subject_label(subject), pipeline, contrast, roi)
        for pipeline in PIPELINES
        for contrast in CONTRASTS
        for roi in ROI_ORDER
    }
    observed_groups = set(counts.index.tolist())
    if observed_groups != expected_groups:
        raise ValueError(
            f"{name} group-key mismatch: missing={len(expected_groups - observed_groups)}, "
            f"unexpected={len(observed_groups - expected_groups)}"
        )
    if method == HISTORICAL_METHOD:
        _require_columns(frame, ("repeat", "fold"), name)
        expected_counts = frame["contrast"].map(
            lambda contrast: 80 if contrast in WITHIN_CONTRASTS else 20
        )
        actual_counts = frame[group_keys].merge(
            counts.rename("group_count").reset_index(), on=group_keys, how="left"
        )["group_count"]
        if not np.array_equal(actual_counts.to_numpy(), expected_counts.to_numpy()):
            raise ValueError(f"{name} has incorrect repeat/fold counts")
        unique_keys = group_keys + ["repeat", "fold"]
        for keys, group in frame.groupby(group_keys, sort=False):
            contrast = keys[2]
            folds = range(1, 5) if contrast in WITHIN_CONTRASTS else (0,)
            expected_pairs = {
                (repeat, fold) for repeat in range(1, 21) for fold in folds
            }
            observed_pairs = set(
                group[["repeat", "fold"]].astype(int).itertuples(index=False, name=None)
            )
            if observed_pairs != expected_pairs:
                raise ValueError(f"{name} has incorrect repeat/fold keys for {keys}")
    elif method == RUNWISE_METHOD:
        _require_columns(frame, ("held_out_run",), name)
        if not (counts == 10).all():
            raise ValueError(f"{name} must contain 10 folds per result key")
        unique_keys = group_keys + ["held_out_run"]
        held_sets = frame.groupby(group_keys, sort=False)["held_out_run"].agg(
            lambda values: tuple(sorted(int(value) for value in values))
        )
        if any(values != tuple(range(1, 11)) for values in held_sets):
            raise ValueError(f"{name} lacks one or more held-out runs")
    else:
        raise ValueError(f"Unknown ERP method: {method}")
    if frame.duplicated(unique_keys).any():
        raise ValueError(f"{name} contains duplicate fold keys")

    expected_group_count = 3 * len(CONTRASTS) * len(ROI_ORDER)
    if len(counts) != expected_group_count:
        raise ValueError(
            f"{name} has {len(counts)} groups; expected {expected_group_count}"
        )


def validate_roi_count_rows(frame: pd.DataFrame, subject: int) -> None:
    name = f"{subject_label(subject)} ROI voxel counts"
    _require_columns(frame, ("subject", "pipeline", "roi", "analysis_voxels"), name)
    keys = ["subject", "pipeline", "roi"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{name} contains duplicate keys")
    expected = {
        (subject_label(subject), pipeline, roi)
        for pipeline in PIPELINES
        for roi in ROI_ORDER
    }
    observed = set(frame[keys].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(f"{name} does not contain exactly 3 x 17 rows")
    if not (pd.to_numeric(frame["analysis_voxels"], errors="coerce") >= 10).all():
        raise ValueError(f"{name} contains fewer than 10 voxels")


def dynamic_pipeline_labels(frame: pd.DataFrame) -> dict[str, str]:
    labels = {
        "historical_28sub_original": "Attached historical result",
        "legacy_lsa_8mm": "Legacy LS-A 8 mm",
        "glmsingle_typed_0mm": "GLMsingle Type-D 0 mm",
        "glmsingle_typed_3mm": "GLMsingle Type-D +3 mm",
    }
    output = {}
    for pipeline, group in frame.groupby("pipeline", sort=False):
        base = labels.get(str(pipeline), str(pipeline))
        if "subject" in group:
            n_subjects = int(group["subject"].nunique())
        elif "n_subjects" in group:
            values = set(pd.to_numeric(group["n_subjects"], errors="raise").astype(int))
            if len(values) != 1:
                raise ValueError(f"Pipeline {pipeline} has inconsistent n_subjects")
            n_subjects = values.pop()
        else:
            raise ValueError("Dynamic labels require subject or n_subjects")
        output[str(pipeline)] = f"{base} (n={n_subjects})"
    return output
