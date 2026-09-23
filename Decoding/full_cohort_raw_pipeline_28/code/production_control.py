#!/usr/bin/env python3
"""Shared fail-closed controls for the 28-subject production lifecycle.

This module contains no analysis code.  It validates immutable control files,
renders an allow-listed command contract, and verifies the append-only,
hash-chained stage ledger used by :mod:`audit_stage_runner`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = ROOT / "ANALYSIS_FREEZE.json"
DEFAULT_CONTRACT = ROOT / "config" / "stage_contracts.json"
DEFAULT_APPROVAL = ROOT / "runtime" / "production_approval.json"
DEFAULT_LOG_ROOT = ROOT / "logs" / "stage_attempts"
MIN_STORAGE_FREE_BYTES = 50 * 1024**3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ControlError(RuntimeError):
    """A production invariant failed; callers must stop without launching work."""


@dataclass(frozen=True)
class LedgerState:
    events: tuple[dict[str, Any], ...]
    head_sha256: str | None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ControlError(f"Not a regular file: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ControlError(f"Required control file is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ControlError(f"Invalid JSON control file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ControlError(f"Control file must contain a JSON object: {path}")
    return value


def validate_subject_lists(freeze: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    included = freeze.get("included_subjects")
    remaining = freeze.get("remaining_batch_subjects")
    if (
        not isinstance(included, list)
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in included)
        or len(included) != 28
        or len(set(included)) != 28
    ):
        raise ControlError("Freeze must contain exactly 28 unique integer included_subjects")
    if (
        not isinstance(remaining, list)
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in remaining)
        or len(remaining) != 25
        or len(set(remaining)) != 25
        or not set(remaining).issubset(set(included))
    ):
        raise ControlError(
            "Freeze must contain exactly 25 unique remaining_batch_subjects drawn from the cohort"
        )
    pilot = freeze.get("pilot_complete_subjects")
    if sorted(set(included) - set(remaining)) != sorted(pilot or []):
        raise ControlError("Included subjects must partition exactly into pilot and remaining sets")
    return list(included), list(remaining)


def read_freeze(path: Path, *, require_production: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    record = file_record(path)
    freeze = load_json(path)
    validate_subject_lists(freeze)
    analysis_id = freeze.get("analysis_id")
    if not isinstance(analysis_id, str) or not analysis_id:
        raise ControlError("Freeze analysis_id is missing")
    if require_production and freeze.get("freeze_status") != "frozen_for_production":
        raise ControlError(
            "Production requires freeze_status='frozen_for_production'; "
            f"found {freeze.get('freeze_status')!r}"
        )
    canonical_path_text = freeze.get("canonical_cohort_config")
    canonical_hash = freeze.get("canonical_cohort_config_sha256")
    if not isinstance(canonical_path_text, str) or not canonical_path_text:
        raise ControlError("Freeze canonical_cohort_config is missing")
    if not isinstance(canonical_hash, str) or not SHA256_RE.fullmatch(canonical_hash):
        raise ControlError("Freeze canonical_cohort_config_sha256 is invalid")
    canonical_record = file_record(Path(canonical_path_text))
    if canonical_record["sha256"] != canonical_hash:
        raise ControlError("Freeze canonical cohort configuration no longer matches its SHA-256")
    return freeze, record


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = file_record(path)
    contract = load_json(path)
    if contract.get("schema_version") != 1:
        raise ControlError("Unsupported stage-contract schema_version")
    stage_order = contract.get("stage_order")
    stages = contract.get("stages")
    if (
        not isinstance(stage_order, list)
        or not stage_order
        or len(stage_order) != len(set(stage_order))
        or not all(isinstance(value, str) and value for value in stage_order)
        or not isinstance(stages, dict)
        or set(stages) != set(stage_order)
    ):
        raise ControlError("stage_order and stages must name the same unique stages")
    storage = contract.get("storage_gate")
    if not isinstance(storage, dict):
        raise ControlError("Contract storage_gate is missing")
    minimum = storage.get("minimum_free_bytes")
    if not isinstance(minimum, int) or minimum < MIN_STORAGE_FREE_BYTES:
        raise ControlError(
            f"Contract storage minimum must be at least {MIN_STORAGE_FREE_BYTES} bytes (50 GiB)"
        )
    if not isinstance(storage.get("path"), str) or not storage["path"]:
        raise ControlError("Contract storage_gate.path is missing")
    pin_files = contract.get("pin_files")
    if not isinstance(pin_files, dict) or not pin_files:
        raise ControlError("Contract pin_files mapping is missing")
    for name, value in pin_files.items():
        if not isinstance(name, str) or not name or not isinstance(value, str) or not value:
            raise ControlError("Every pin_files entry must map a non-empty name to a path")
    for stage_name in stage_order:
        stage = stages[stage_name]
        if not isinstance(stage, dict):
            raise ControlError(f"Stage {stage_name} must be an object")
        state = stage.get("state")
        if state not in {"executable", "blocked"}:
            raise ControlError(f"Stage {stage_name} has invalid state {state!r}")
        argv = stage.get("command_argv")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(value, str) and value and "\x00" not in value for value in argv
        ):
            raise ControlError(f"Stage {stage_name} requires a non-empty string command_argv")
        required = stage.get("required_approved_files")
        if not isinstance(required, list) or not required or not all(
            isinstance(value, str) and value for value in required
        ):
            raise ControlError(f"Stage {stage_name} requires required_approved_files")
        unknown = set(required) - (
            {
                "freeze",
                "canonical_cohort_config",
                "cohort_config",
                "source_selection",
                "stage_contract",
            }
            | set(pin_files)
        )
        if unknown:
            raise ControlError(f"Stage {stage_name} names unknown approved files: {sorted(unknown)}")
        if state == "blocked" and not stage.get("blocked_reason"):
            raise ControlError(f"Blocked stage {stage_name} requires blocked_reason")
    return contract, record


def _assert_record_matches(name: str, approved: Mapping[str, Any], actual_path: Path) -> dict[str, Any]:
    if not isinstance(approved, dict):
        raise ControlError(f"Approval record {name!r} is malformed")
    actual = file_record(actual_path)
    try:
        approved_path = str(Path(str(approved["path"])).resolve(strict=True))
    except (KeyError, FileNotFoundError) as error:
        raise ControlError(f"Approved path for {name!r} is missing") from error
    if approved_path != actual["path"]:
        raise ControlError(
            f"Approved path mismatch for {name}: approved={approved_path}; actual={actual['path']}"
        )
    if approved.get("sha256") != actual["sha256"] or approved.get("size_bytes") != actual["size_bytes"]:
        raise ControlError(f"Approved identity changed for {name}: {actual['path']}")
    return actual


def validate_approval(
    approval_path: Path,
    freeze_path: Path,
    contract_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    approval_record = file_record(approval_path)
    approval = load_json(approval_path)
    if approval.get("schema_version") != 1 or approval.get("approval_status") != "approved_for_production":
        raise ControlError("Production approval is absent or is not approved_for_production")
    freeze, freeze_record = read_freeze(freeze_path, require_production=True)
    contract, contract_record = load_contract(contract_path)
    if approval.get("analysis_id") != freeze["analysis_id"]:
        raise ControlError("Approval analysis_id does not match the freeze")
    if contract.get("analysis_id") != freeze["analysis_id"]:
        raise ControlError("Stage-contract analysis_id does not match the freeze")
    _, remaining = validate_subject_lists(freeze)
    if approval.get("production_subjects") != remaining:
        raise ControlError("Approval production_subjects do not exactly match the frozen remaining cohort")
    approved_files = approval.get("approved_files")
    if not isinstance(approved_files, dict):
        raise ControlError("Approval approved_files mapping is missing")
    identities: dict[str, dict[str, Any]] = {}
    identities["freeze"] = _assert_record_matches("freeze", approved_files.get("freeze", {}), freeze_path)
    identities["stage_contract"] = _assert_record_matches(
        "stage_contract", approved_files.get("stage_contract", {}), contract_path
    )
    canonical_path = Path(str(freeze["canonical_cohort_config"]))
    identities["canonical_cohort_config"] = _assert_record_matches(
        "canonical_cohort_config",
        approved_files.get("canonical_cohort_config", {}),
        canonical_path,
    )
    if identities["canonical_cohort_config"]["sha256"] != freeze[
        "canonical_cohort_config_sha256"
    ]:
        raise ControlError("Approval canonical cohort config differs from the hash in the freeze")
    for fixed_name in ("cohort_config", "source_selection"):
        record = approved_files.get(fixed_name)
        if not isinstance(record, dict) or "path" not in record:
            raise ControlError(f"Approval is missing {fixed_name}")
        identities[fixed_name] = _assert_record_matches(
            fixed_name, record, Path(str(record["path"]))
        )
    for name, path_text in contract["pin_files"].items():
        identities[name] = _assert_record_matches(
            name, approved_files.get(name, {}), Path(path_text)
        )
    expected_names = {
        "freeze",
        "stage_contract",
        "canonical_cohort_config",
        "cohort_config",
        "source_selection",
        *contract["pin_files"],
    }
    if set(approved_files) != expected_names:
        raise ControlError(
            "Approval file set is not exact; "
            f"missing={sorted(expected_names - set(approved_files))}, "
            f"unexpected={sorted(set(approved_files) - expected_names)}"
        )
    return approval, approval_record, freeze, contract, identities


def approved_path(identities: Mapping[str, Mapping[str, Any]], name: str) -> str:
    try:
        return str(identities[name]["path"])
    except KeyError as error:
        raise ControlError(f"No approved path is available for {name!r}") from error


def build_control_identity(
    analysis_id: str,
    approval_record: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], str]:
    identity = {
        "analysis_id": analysis_id,
        "approval_sha256": str(approval_record["sha256"]),
        "freeze_sha256": str(identities["freeze"]["sha256"]),
        "stage_contract_sha256": str(identities["stage_contract"]["sha256"]),
        "source_selection_sha256": str(identities["source_selection"]["sha256"]),
        "cohort_config_sha256": str(identities["cohort_config"]["sha256"]),
        "canonical_cohort_config_sha256": str(
            identities["canonical_cohort_config"]["sha256"]
        ),
    }
    return identity, sha256_bytes(canonical_json_bytes(identity))


def _template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, format_spec, conversion in Formatter().parse(template):
        if field_name is None:
            continue
        if format_spec or conversion:
            raise ControlError(f"Format specifications/conversions are forbidden: {template!r}")
        if not field_name or "." in field_name or "[" in field_name:
            raise ControlError(f"Only simple named placeholders are allowed: {template!r}")
        fields.add(field_name)
    return fields


def render_command(
    contract: Mapping[str, Any],
    stage_name: str,
    subject: int,
    identities: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    stages = contract["stages"]
    if stage_name not in stages:
        raise ControlError(f"Unknown stage {stage_name!r}")
    stage = stages[stage_name]
    if stage["state"] != "executable":
        raise ControlError(
            f"Stage {stage_name!r} is blocked by contract: {stage.get('blocked_reason', 'unspecified')}"
        )
    root = str(ROOT.resolve())
    variables: dict[str, str] = {
        "root": root,
        "subject": str(subject),
        "subject02": f"{subject:02d}",
        "freeze": approved_path(identities, "freeze"),
        "cohort_config": approved_path(identities, "cohort_config"),
        "source_selection": approved_path(identities, "source_selection"),
        "stage_contract": approved_path(identities, "stage_contract"),
    }
    variables.update({name: approved_path(identities, name) for name in contract["pin_files"]})
    paths = contract.get("path_variables", {})
    if not isinstance(paths, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in paths.items()
    ):
        raise ControlError("path_variables must be a string mapping")
    # Path-variable templates may use only the immutable base variables.
    for name, template in paths.items():
        unknown = _template_fields(template) - set(variables)
        if unknown:
            raise ControlError(f"Path variable {name} uses unknown fields: {sorted(unknown)}")
        variables[name] = template.format_map(variables)
    rendered: list[str] = []
    for template in stage["command_argv"]:
        unknown = _template_fields(template) - set(variables)
        if unknown:
            raise ControlError(f"Stage {stage_name} uses unknown fields: {sorted(unknown)}")
        value = template.format_map(variables)
        if not value or "\x00" in value:
            raise ControlError(f"Stage {stage_name} rendered an invalid argv item")
        rendered.append(value)
    return rendered, variables


def validate_stage_files(
    stage: Mapping[str, Any], identities: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    records = []
    baseline = [
        "freeze",
        "canonical_cohort_config",
        "cohort_config",
        "source_selection",
        "stage_contract",
    ]
    names = list(dict.fromkeys([*baseline, *stage["required_approved_files"]]))
    for name in names:
        if name not in identities:
            raise ControlError(f"Required approved file {name!r} is unavailable")
        # Re-hash at attempt time, not merely when the approval was loaded.
        actual = file_record(Path(str(identities[name]["path"])))
        if actual != dict(identities[name]):
            raise ControlError(f"Approved file changed while preparing attempt: {name}")
        records.append({"name": name, **actual})
    return records


def event_hash(event_without_hash: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(event_without_hash))


def read_ledger(path: Path) -> LedgerState:
    if not path.exists():
        return LedgerState(events=(), head_sha256=None)
    events: list[dict[str, Any]] = []
    previous: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for expected_sequence, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ControlError(f"Ledger has a non-terminated final record: {path}")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ControlError(f"Ledger JSON is corrupt at line {expected_sequence}") from error
            if not isinstance(event, dict):
                raise ControlError(f"Ledger line {expected_sequence} is not an object")
            supplied_hash = event.pop("event_sha256", None)
            if event.get("sequence") != expected_sequence:
                raise ControlError(f"Ledger sequence mismatch at line {expected_sequence}")
            if event.get("previous_event_sha256") != previous:
                raise ControlError(f"Ledger hash-chain mismatch at line {expected_sequence}")
            calculated = event_hash(event)
            if supplied_hash != calculated:
                raise ControlError(f"Ledger event hash mismatch at line {expected_sequence}")
            event["event_sha256"] = supplied_hash
            events.append(event)
            previous = supplied_hash
    return LedgerState(events=tuple(events), head_sha256=previous)


def encode_ledger_event(state: LedgerState, payload: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"sequence", "previous_event_sha256", "event_sha256"} & set(payload)
    if forbidden:
        raise ControlError(f"Caller supplied reserved ledger keys: {sorted(forbidden)}")
    event = {
        "sequence": len(state.events) + 1,
        "previous_event_sha256": state.head_sha256,
        **dict(payload),
    }
    event["event_sha256"] = event_hash(event)
    return event


def latest_stage_completions(
    events: Iterable[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    latest: dict[tuple[int, str], Mapping[str, Any]] = {}
    for event in events:
        if event.get("event") == "completed":
            key = (int(event["subject"]), str(event["stage"]))
            latest[key] = event
    return latest


def validate_prior_stage(
    contract: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    subject: int,
    stage_name: str,
    identity_sha256: str,
) -> None:
    order = list(contract["stage_order"])
    index = order.index(stage_name)
    if index == 0:
        return
    prior = order[index - 1]
    completion = latest_stage_completions(events).get((subject, prior))
    if completion is None:
        raise ControlError(f"Sub{subject} stage {stage_name} requires prior stage {prior}")
    if completion.get("status") != "passed":
        raise ControlError(
            f"Sub{subject} stage {stage_name} requires a passed {prior}; "
            f"latest status={completion.get('status')!r}"
        )
    if completion.get("control_identity_sha256") != identity_sha256:
        raise ControlError(
            f"Sub{subject} prior stage {prior} was completed under a different control identity"
        )


def validate_source_selection_for_approval(path: Path, subjects: Sequence[int]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    required = {"asset_id", "asset_type", "subject", "approval_status", "source_sha256"}
    if not required.issubset(fields):
        raise ControlError(f"Source manifest lacks required columns: {sorted(required - fields)}")
    if not rows:
        raise ControlError("Source manifest is empty")
    if any(row["approval_status"] != "approved" for row in rows):
        raise ControlError("Every source-manifest row must be explicitly approved")
    if len({row["asset_id"] for row in rows}) != len(rows):
        raise ControlError("Source manifest asset_id values are not unique")
    if any(not SHA256_RE.fullmatch(row["source_sha256"]) for row in rows):
        raise ControlError("Every source-manifest row must contain a lowercase SHA-256")
    subject_set = {int(row["subject"]) for row in rows}
    if subject_set != set(subjects):
        raise ControlError(
            "Source manifest subject set is not the exact remaining cohort; "
            f"missing={sorted(set(subjects) - subject_set)}, "
            f"unexpected={sorted(subject_set - set(subjects))}"
        )
    bold_counts = {
        subject: sum(
            int(row["subject"]) == subject and row["asset_type"].lower() in {"bold", "func", "functional"}
            for row in rows
        )
        for subject in subjects
    }
    bad = {subject: count for subject, count in bold_counts.items() if count != 10}
    if bad:
        raise ControlError(f"Every production subject must have exactly 10 approved BOLD assets: {bad}")
    return {"row_count": len(rows), "bold_counts": bold_counts}


def ensure_exact_subjects(actual: Sequence[int], expected: Sequence[int]) -> None:
    if list(actual) != list(expected):
        raise ControlError(
            "Production subject order/set is not exact; "
            f"expected={list(expected)}, actual={list(actual)}"
        )
