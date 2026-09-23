#!/usr/bin/env python3
"""Promote an independently reviewed, hash-frozen source draft to production.

The promotion is deliberately separate from discovery and hashing. It refuses
to change any selected path or metadata field, verifies the pinned independent
review inputs, re-hashes every selected source, and changes only
``approval_status`` before publishing the production manifest atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from cohortlib import (
    SELECTION_COLUMNS,
    audit_selection,
    load_config,
    read_selection,
    sha256_file,
    write_tsv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "cohort.json"
DEFAULT_REVIEW_DRAFT = ROOT / "audit" / "source_inventory" / "review_draft_selection.tsv"
DEFAULT_HASHED_DRAFT = (
    ROOT / "audit" / "source_inventory" / "hashed_review_draft_selection.tsv"
)
DEFAULT_REVIEW_RECORD = (
    ROOT / "audit" / "source_inventory" / "independent_selection_review_20260729.json"
)
DEFAULT_OUTPUT = ROOT / "manifests" / "source_selection.tsv"
DEFAULT_AUDIT_OUTPUT = (
    ROOT / "audit" / "source_inventory" / "source_manifest_promotion_20260729.json"
)

HASH_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_ARTIFACTS = {
    "config/cohort.json",
    "audit/source_inventory/review_draft_selection.tsv",
    "audit/source_inventory/manual_resolution_decisions.tsv",
    "audit/source_inventory/gre_pair_audit.tsv",
    "audit/source_inventory/onset_audit.tsv",
    "audit/source_inventory/review_validation/selection_validation_summary.json",
    "audit/source_inventory/provenance_repair_20260729.json",
}
EXPECTED_COUNTS = {
    "subjects": 25,
    "assets": 394,
    "bold": 250,
    "t1w": 48,
    "fmap": 96,
    "machine_unique_candidates": 312,
    "manually_reviewed_candidates": 82,
    "gre_sessions": 20,
    "bold_onset_files": 250,
}
REQUIRED_FINDINGS = {
    "all_required_assets_have_one_exact_selected_path",
    "all_selected_paths_are_in_the_configured_same_session",
    "no_moco_series_selected",
    "no_cross_session_fieldmap_borrowing",
    "scanner_run_labels_match_the_frozen_logical_run_mapping",
    "all_manual_resolution_decisions_match_the_selected_rows",
    "all_gre_component_triples_pass_echo_and_series_pairing_checks",
    "all_bold_rows_have_the_exact_config_derived_onset_path",
    "sidecar_and_onset_identities_are_frozen",
}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".staging")
    if staging.exists():
        raise RuntimeError(f"Refusing abandoned audit staging file: {staging}")
    staging.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(staging, path)


def validate_review_record(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("status") != "passed":
        raise RuntimeError("Independent review record is not schema v1 with passed status")
    if record.get("expected_counts") != EXPECTED_COUNTS:
        raise RuntimeError("Independent review expected counts differ from the frozen cohort")
    findings = record.get("findings")
    if not isinstance(findings, dict) or set(findings) != REQUIRED_FINDINGS:
        raise RuntimeError("Independent review finding set is incomplete or unexpected")
    if any(findings[key] is not True for key in REQUIRED_FINDINGS):
        raise RuntimeError("Independent review contains a non-passing finding")

    artifacts = record.get("reviewed_artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("Independent review does not contain reviewed_artifacts")
    by_path = {str(item.get("path", "")).replace("\\", "/"): item for item in artifacts}
    if set(by_path) != EXPECTED_ARTIFACTS or len(artifacts) != len(EXPECTED_ARTIFACTS):
        raise RuntimeError("Independent review artifact set is incomplete, duplicated, or unexpected")
    for relative, item in by_path.items():
        artifact = ROOT.joinpath(*Path(relative).parts)
        if not artifact.is_file():
            raise RuntimeError(f"Reviewed artifact is missing: {artifact}")
        expected_size = int(item.get("size_bytes", -1))
        expected_hash = str(item.get("sha256", "")).casefold()
        if artifact.stat().st_size != expected_size:
            raise RuntimeError(f"Reviewed artifact size changed: {artifact}")
        if not HASH_RE.fullmatch(expected_hash) or sha256_file(artifact) != expected_hash:
            raise RuntimeError(f"Reviewed artifact SHA-256 changed: {artifact}")
    return record


def validate_draft_equivalence(
    reviewed_rows: list[dict[str, str]], hashed_rows: list[dict[str, str]]
) -> None:
    reviewed_by_id = {row["asset_id"]: row for row in reviewed_rows}
    hashed_by_id = {row["asset_id"]: row for row in hashed_rows}
    if len(reviewed_by_id) != len(reviewed_rows) or len(hashed_by_id) != len(hashed_rows):
        raise RuntimeError("Duplicate asset_id in reviewed or hash-frozen draft")
    if set(reviewed_by_id) != set(hashed_by_id):
        raise RuntimeError("Asset IDs differ between independently reviewed and hash-frozen drafts")
    for asset_id, reviewed in reviewed_by_id.items():
        hashed = hashed_by_id[asset_id]
        for field in SELECTION_COLUMNS:
            if field == "source_sha256":
                continue
            if reviewed[field] != hashed[field]:
                raise RuntimeError(
                    f"Field {field} changed after independent review for {asset_id}: "
                    f"{reviewed[field]!r} != {hashed[field]!r}"
                )


def validate_frozen_rows(config: dict, rows: list[dict[str, str]]) -> dict:
    remaining = list(map(int, config["cohort"]["remaining_subjects"]))
    if sorted({int(row["subject"]) for row in rows}) != remaining:
        raise RuntimeError("Hash-frozen draft subjects differ from the exact remaining cohort")
    status_counts = Counter(row["approval_status"] for row in rows)
    if status_counts != Counter({"proposed_unique": 312, "reviewed_candidate": 82}):
        raise RuntimeError(f"Unexpected pre-approval status counts: {dict(status_counts)}")
    type_counts = Counter(row["asset_type"] for row in rows)
    if type_counts != Counter({"bold": 250, "fmap": 96, "t1w": 48}):
        raise RuntimeError(f"Unexpected asset counts: {dict(type_counts)}")
    if len(rows) != 394:
        raise RuntimeError(f"Expected 394 assets; found {len(rows)}")
    for row in rows:
        if not HASH_RE.fullmatch(row["source_sha256"].casefold()):
            raise RuntimeError(f"Missing/invalid NIfTI SHA-256 for {row['asset_id']}")
        if not HASH_RE.fullmatch(row["source_json_sha256"].casefold()):
            raise RuntimeError(f"Missing/invalid JSON SHA-256 for {row['asset_id']}")
        if row["asset_type"] == "bold":
            if not HASH_RE.fullmatch(row["onset_mat_sha256"].casefold()):
                raise RuntimeError(f"Missing/invalid onset SHA-256 for {row['asset_id']}")
        elif row["onset_mat_relpath"] or row["onset_mat_size_bytes"] or row["onset_mat_sha256"]:
            raise RuntimeError(f"Unexpected onset identity on non-BOLD asset {row['asset_id']}")
    return {
        "rows": len(rows),
        "subjects": len(remaining),
        "asset_type_counts": dict(sorted(type_counts.items())),
        "preapproval_status_counts": dict(sorted(status_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--review-draft", type=Path, default=DEFAULT_REVIEW_DRAFT)
    parser.add_argument("--hashed-draft", type=Path, default=DEFAULT_HASHED_DRAFT)
    parser.add_argument("--review-record", type=Path, default=DEFAULT_REVIEW_RECORD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--stimuli-csv", type=Path)
    args = parser.parse_args()

    placeholder_hash = None
    if args.output.exists():
        existing_rows = read_selection(args.output)
        if existing_rows:
            raise RuntimeError("Refusing to overwrite a nonempty production manifest")
        placeholder_hash = sha256_file(args.output)
    if args.audit_output.exists():
        raise RuntimeError("Refusing to overwrite an existing promotion audit")
    manifest_staging = args.output.with_name(args.output.name + ".staging")
    if manifest_staging.exists():
        raise RuntimeError(f"Refusing abandoned manifest staging file: {manifest_staging}")

    config = load_config(args.config)
    review_record = validate_review_record(args.review_record)
    reviewed_rows = read_selection(args.review_draft)
    hashed_rows = read_selection(args.hashed_draft)
    validate_draft_equivalence(reviewed_rows, hashed_rows)
    frozen_summary = validate_frozen_rows(config, hashed_rows)

    raw_root = args.raw_root or Path(config["paths"]["raw_root"])
    log_root = args.log_root or Path(config["paths"]["log_root"])
    stimuli_csv = args.stimuli_csv or Path(config["paths"]["stimuli_csv"])
    remaining = list(map(int, config["cohort"]["remaining_subjects"]))
    _, issues = audit_selection(
        config,
        args.hashed_draft,
        raw_root,
        log_root,
        stimuli_csv,
        remaining,
        require_approved=False,
        check_onsets=True,
        verify_hashes=True,
    )
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        preview = "; ".join(
            f"{issue.code}:{issue.asset_id}:{issue.message}" for issue in errors[:10]
        )
        raise RuntimeError(f"Strict hash/onset audit failed with {len(errors)} errors: {preview}")

    promoted_rows = [dict(row, approval_status="approved") for row in hashed_rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(manifest_staging, promoted_rows, SELECTION_COLUMNS)
    _, promoted_issues = audit_selection(
        config,
        manifest_staging,
        raw_root,
        log_root,
        stimuli_csv,
        remaining,
        require_approved=True,
        check_onsets=False,
        verify_hashes=False,
    )
    promoted_errors = [issue for issue in promoted_issues if issue.severity == "error"]
    if promoted_errors:
        raise RuntimeError(f"Promoted manifest structural audit failed with {len(promoted_errors)} errors")

    source_hash = sha256_file(args.hashed_draft)
    output_hash = sha256_file(manifest_staging)
    audit_payload = {
        "schema_version": 1,
        "status": "passed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "action": "promoted independently reviewed hash-frozen source rows",
        "only_intended_field_change": "approval_status -> approved",
        "review_id": review_record["review_id"],
        "review_record": str(args.review_record),
        "review_record_sha256": sha256_file(args.review_record),
        "reviewed_draft": str(args.review_draft),
        "reviewed_draft_sha256": sha256_file(args.review_draft),
        "hashed_draft": str(args.hashed_draft),
        "hashed_draft_sha256": source_hash,
        "production_manifest": str(args.output),
        "production_manifest_sha256": output_hash,
        "replaced_header_only_placeholder_sha256": placeholder_hash,
        "strict_current_byte_hash_verification": "passed",
        "strict_onset_verification": "passed",
        "warnings": len([issue for issue in issues if issue.severity == "warning"]),
        **frozen_summary,
    }
    os.replace(manifest_staging, args.output)
    try:
        _atomic_json(args.audit_output, audit_payload)
    except Exception:
        # The manifest itself remains self-validating and immutable; make the
        # missing audit conspicuous instead of silently deleting approved work.
        raise RuntimeError(
            f"Production manifest was published at {args.output}, but promotion audit publication failed"
        )
    print(json.dumps(audit_payload, indent=2))


if __name__ == "__main__":
    main()
