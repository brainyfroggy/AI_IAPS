#!/usr/bin/env python3
"""Inventory raw candidates and validate an explicit full-cohort selection.

The discovery report is advisory. It never grants production approval. The
builder accepts only exact paths from manifests/source_selection.tsv whose
approval_status is ``approved`` and which pass every gate again.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from cohortlib import (
    SELECTION_COLUMNS,
    Issue,
    audit_all_onsets,
    audit_selection,
    classify_source,
    is_norm_gre,
    iter_expected_assets,
    iter_expected_runs,
    load_config,
    metadata_run_label,
    read_selection,
    read_json,
    sha256_file,
    sidecar_for_nifti,
    write_tsv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "cohort.json"
DEFAULT_SELECTION = ROOT / "manifests" / "source_selection.tsv"
DEFAULT_REPORT_DIR = ROOT / "audit" / "source_inventory"

CANDIDATE_COLUMNS = [
    "subject",
    "session",
    "source_dir",
    "source_relpath",
    "source_size_bytes",
    "sidecar_relpath",
    "sidecar_size_bytes",
    "sidecar_sha256",
    "asset_type",
    "fmap_role",
    "metadata_run_label",
    "series_number",
    "series_description",
    "protocol_name",
    "echo_time",
    "phase_encoding_direction",
    "is_norm_gre",
    "candidate_status",
    "exclusion_reason",
]


def _configured_paths(config: Mapping, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    raw_root = args.raw_root or Path(config["paths"]["raw_root"])
    log_root = args.log_root or Path(config["paths"]["log_root"])
    stimuli_csv = args.stimuli_csv or Path(config["paths"]["stimuli_csv"])
    return raw_root, log_root, stimuli_csv


def _subjects_arg(values: list[int] | None) -> Iterable[int] | None:
    return values if values else None


def discover_candidates(
    config: Mapping,
    raw_root: Path,
    subjects: Iterable[int] | None,
) -> list[dict[str, object]]:
    sessions: dict[tuple[int, int], str] = {}
    for run in iter_expected_runs(config, subjects):
        sessions[(run.subject, run.session)] = run.source_dir
    rows: list[dict[str, object]] = []
    for (subject, session), source_dir in sorted(sessions.items()):
        acquisition_root = raw_root.joinpath(*PurePosixPath(source_dir).parts)
        if not acquisition_root.is_dir():
            rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "source_dir": source_dir,
                    "source_relpath": "",
                    "source_size_bytes": "",
                    "sidecar_relpath": "",
                    "sidecar_size_bytes": "",
                    "sidecar_sha256": "",
                    "asset_type": "",
                    "fmap_role": "",
                    "metadata_run_label": "",
                    "series_number": "",
                    "series_description": "",
                    "protocol_name": "",
                    "echo_time": "",
                    "phase_encoding_direction": "",
                    "is_norm_gre": "",
                    "candidate_status": "missing_acquisition_directory",
                    "exclusion_reason": str(acquisition_root),
                }
            )
            continue
        # Recursive enumeration is allowed only here, to expose every candidate.
        # Production selection below is always an exact manifest path.
        for source in sorted(acquisition_root.rglob("*.nii.gz")):
            sidecar = sidecar_for_nifti(source)
            try:
                rel = source.relative_to(raw_root).as_posix()
            except ValueError:
                continue
            if not sidecar.is_file():
                rows.append(
                    {
                        "subject": subject,
                        "session": session,
                        "source_dir": source_dir,
                        "source_relpath": rel,
                        "source_size_bytes": source.stat().st_size,
                        "sidecar_relpath": "",
                        "sidecar_size_bytes": "",
                        "sidecar_sha256": "",
                        "asset_type": "",
                        "fmap_role": "",
                        "metadata_run_label": "",
                        "series_number": "",
                        "series_description": "",
                        "protocol_name": "",
                        "echo_time": "",
                        "phase_encoding_direction": "",
                        "is_norm_gre": "",
                        "candidate_status": "rejected",
                        "exclusion_reason": "missing_json_sidecar",
                    }
                )
                continue
            try:
                metadata = read_json(sidecar)
                asset_type, fmap_role, exclusion = classify_source(source, metadata)
                run_label = metadata_run_label(metadata)
            except Exception as exc:  # noqa: BLE001 - surface malformed metadata in report
                metadata = {}
                asset_type, fmap_role, exclusion, run_label = "", "", f"invalid_json:{exc}", None
            required = asset_type in {"bold", "t1w", "fmap"}
            status = "eligible" if required and not exclusion else "rejected"
            rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "source_dir": source_dir,
                    "source_relpath": rel,
                    "source_size_bytes": source.stat().st_size,
                    "sidecar_relpath": sidecar.relative_to(raw_root).as_posix(),
                    "sidecar_size_bytes": sidecar.stat().st_size,
                    "sidecar_sha256": sha256_file(sidecar),
                    "asset_type": asset_type,
                    "fmap_role": fmap_role,
                    "metadata_run_label": run_label if run_label is not None else "",
                    "series_number": metadata.get("SeriesNumber", ""),
                    "series_description": metadata.get("SeriesDescription", ""),
                    "protocol_name": metadata.get("ProtocolName", ""),
                    "echo_time": metadata.get("EchoTime", ""),
                    "phase_encoding_direction": metadata.get("PhaseEncodingDirection", ""),
                    "is_norm_gre": is_norm_gre(metadata),
                    "candidate_status": status,
                    "exclusion_reason": exclusion,
                }
            )
    return rows


def propose_selection(
    config: Mapping,
    candidates: list[dict],
    log_root: Path,
    subjects: Iterable[int] | None,
) -> list[dict]:
    eligible = [row for row in candidates if row["candidate_status"] == "eligible"]
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for row in eligible:
        key = (
            int(row["subject"]),
            int(row["session"]),
            row["asset_type"],
            row["fmap_role"],
            int(row["metadata_run_label"]) if row["metadata_run_label"] != "" else None,
        )
        by_key[key].append(row)

    proposals: list[dict[str, object]] = []
    for asset in iter_expected_assets(config, subjects):
        if asset.asset_type == "bold":
            matches = by_key.get(
                (
                    asset.subject,
                    asset.session,
                    "bold",
                    "",
                    asset.source_run_label,
                ),
                [],
            )
        elif asset.asset_type == "t1w":
            matches = by_key.get((asset.subject, asset.session, "t1w", "", None), [])
        else:
            matches = by_key.get(
                (asset.subject, asset.session, "fmap", asset.fmap_role, None),
                [],
            )
            if asset.sdc_mode == "gre" and asset.fmap_role.startswith("magnitude"):
                norm = [row for row in matches if row["is_norm_gre"] is True]
                if norm:
                    matches = norm
        exact = matches[0] if len(matches) == 1 else None
        if asset.sdc_mode == "gre" and asset.asset_type == "fmap":
            # Even a unique component needs review of dcm2niix/series provenance.
            status = "review_required_gre_component" if exact else (
                "unresolved_missing" if not matches else "unresolved_ambiguous"
            )
        else:
            status = "proposed_unique" if exact else (
                "unresolved_missing" if not matches else "unresolved_ambiguous"
            )
        candidate_paths = " | ".join(str(row["source_relpath"]) for row in matches)
        notes = f"candidate_count={len(matches)}"
        if candidate_paths:
            notes += f"; candidates={candidate_paths}"
        onset_path = (
            log_root.joinpath(*PurePosixPath(asset.log_relpath).parts)
            if asset.asset_type == "bold"
            else None
        )
        onset_size = onset_path.stat().st_size if onset_path is not None and onset_path.is_file() else ""
        onset_hash = sha256_file(onset_path) if onset_path is not None and onset_path.is_file() else ""
        proposals.append(
            {
                "asset_id": asset.asset_id,
                "asset_type": asset.asset_type,
                "subject": asset.subject,
                "session": asset.session,
                "experimental_run": asset.experimental_run if asset.experimental_run is not None else "",
                "source_run_label": asset.source_run_label if asset.source_run_label is not None else "",
                "source_relpath": exact["source_relpath"] if exact else "",
                "source_json_relpath": exact["sidecar_relpath"] if exact else "",
                "log_relpath": asset.log_relpath,
                "onset_mat_relpath": asset.log_relpath,
                "sdc_mode": asset.sdc_mode,
                "fmap_role": asset.fmap_role,
                "approval_status": status,
                "source_size_bytes": exact["source_size_bytes"] if exact else "",
                "source_sha256": "",
                "source_json_size_bytes": exact["sidecar_size_bytes"] if exact else "",
                "source_json_sha256": exact["sidecar_sha256"] if exact else "",
                "onset_mat_size_bytes": onset_size,
                "onset_mat_sha256": onset_hash,
                "notes": notes,
            }
        )
    return proposals


def write_summary(
    report_dir: Path,
    config: Mapping,
    candidates: list[dict],
    proposals: list[dict],
    onset_rows: list[dict],
) -> dict[str, object]:
    summary = {
        "cohort_remaining_subjects": config["cohort"]["remaining_subjects"],
        "expected_bold_runs": sum(1 for row in proposals if row["asset_type"] == "bold"),
        "expected_assets": len(proposals),
        "candidate_rows": len(candidates),
        "candidate_status_counts": dict(Counter(str(row["candidate_status"]) for row in candidates)),
        "proposal_status_counts": dict(Counter(str(row["approval_status"]) for row in proposals)),
        "onset_status_counts": dict(Counter(str(row["status"]) for row in onset_rows)),
        "production_ready": all(row["approval_status"] == "approved" for row in proposals)
        and all(row["status"] == "passed" for row in onset_rows),
        "note": "Discovery/proposal never grants approval; production_ready remains false until a reviewed manifest is approved and re-audited.",
    }
    (report_dir / "discovery_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def command_discover(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    raw_root, log_root, stimuli_csv = _configured_paths(config, args)
    subjects = _subjects_arg(args.subject)
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    candidates = discover_candidates(config, raw_root, subjects)
    proposals = propose_selection(config, candidates, log_root, subjects)
    onset_rows = audit_all_onsets(config, log_root, stimuli_csv, subjects)
    write_tsv(report_dir / "candidate_inventory.tsv", candidates, CANDIDATE_COLUMNS)
    write_tsv(report_dir / "proposed_source_selection.tsv", proposals, SELECTION_COLUMNS)
    write_tsv(report_dir / "onset_audit.tsv", onset_rows)
    summary = write_summary(report_dir, config, candidates, proposals, onset_rows)
    print(json.dumps(summary, indent=2))
    return 0 if all(row["status"] == "passed" for row in onset_rows) else 2


def command_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    raw_root, log_root, stimuli_csv = _configured_paths(config, args)
    rows, issues = audit_selection(
        config,
        args.selection,
        raw_root,
        log_root,
        stimuli_csv,
        _subjects_arg(args.subject),
        require_approved=not args.allow_proposed,
        check_onsets=not args.skip_onsets,
        verify_hashes=args.verify_hashes,
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(
        args.report_dir / "selection_validation_issues.tsv",
        [issue.as_dict() for issue in issues],
        ["severity", "code", "asset_id", "message"],
    )
    counts = Counter(issue.severity for issue in issues)
    summary = {
        "selection": str(args.selection),
        "rows_in_scope": len(rows),
        "errors": counts.get("error", 0),
        "warnings": counts.get("warning", 0),
        "status": "passed" if counts.get("error", 0) == 0 else "failed",
        "verify_hashes": args.verify_hashes,
    }
    (args.report_dir / "selection_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    for issue in issues[:50]:
        print(f"{issue.severity.upper()} {issue.code} {issue.asset_id}: {issue.message}", file=sys.stderr)
    if len(issues) > 50:
        print(f"... {len(issues) - 50} additional issues in TSV", file=sys.stderr)
    return 0 if counts.get("error", 0) == 0 else 2


def command_onsets(args: argparse.Namespace) -> int:
    """Refresh only the onset gate without repeating the slow raw-file inventory."""
    config = load_config(args.config)
    _, log_root, stimuli_csv = _configured_paths(config, args)
    onset_rows = audit_all_onsets(config, log_root, stimuli_csv, _subjects_arg(args.subject))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.report_dir / "onset_audit.tsv", onset_rows)
    counts = Counter(str(row["status"]) for row in onset_rows)
    summary = {
        "expected_runs": len(onset_rows),
        "status_counts": dict(counts),
        "status": "passed" if counts.get("failed", 0) == 0 else "failed",
    }
    (args.report_dir / "onset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    discovery_summary_path = args.report_dir / "discovery_summary.json"
    if discovery_summary_path.is_file():
        discovery_summary = json.loads(discovery_summary_path.read_text(encoding="utf-8"))
        discovery_summary["onset_status_counts"] = dict(counts)
        # Candidate proposals never become approved merely because onsets pass.
        discovery_summary["production_ready"] = False
        discovery_summary_path.write_text(
            json.dumps(discovery_summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))
    return 0 if counts.get("failed", 0) == 0 else 2


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def command_review_draft(args: argparse.Namespace) -> int:
    """Merge documented ambiguity decisions and audit proposed GRE triplets.

    Output remains deliberately unapproved and cannot be consumed by the BIDS
    builder until an auditor promotes every row into the production manifest.
    """
    config = load_config(args.config)
    proposal_path = args.report_dir / "proposed_source_selection.tsv"
    candidates_path = args.report_dir / "candidate_inventory.tsv"
    decisions_path = args.report_dir / "manual_resolution_decisions.tsv"
    proposals = read_selection(proposal_path)
    candidates = _read_tsv(candidates_path)
    decisions = {row["asset_id"]: row for row in _read_tsv(decisions_path)}
    candidates_by_path = {row["source_relpath"]: row for row in candidates}
    proposal_by_id = {row["asset_id"]: row for row in proposals}

    decision_errors: list[str] = []
    for asset_id, decision in decisions.items():
        if asset_id not in proposal_by_id:
            decision_errors.append(f"Decision refers to unknown asset {asset_id}")
            continue
        row = proposal_by_id[asset_id]
        candidate = candidates_by_path.get(decision["selected_source_relpath"])
        if candidate is None or candidate["candidate_status"] != "eligible":
            decision_errors.append(f"Decision for {asset_id} is not an eligible inventoried candidate")
            continue
        if candidate["source_size_bytes"] != decision["source_size_bytes"]:
            decision_errors.append(f"Decision size mismatch for {asset_id}")
            continue
        row["source_relpath"] = decision["selected_source_relpath"]
        row["source_size_bytes"] = decision["source_size_bytes"]
        row["source_json_relpath"] = candidate["sidecar_relpath"]
        row["source_json_size_bytes"] = candidate["sidecar_size_bytes"]
        row["source_json_sha256"] = candidate["sidecar_sha256"]
        row["approval_status"] = "reviewed_candidate"
        row["notes"] += "; manual_resolution_decisions.tsv"

    gre_rows: list[dict[str, object]] = []
    gre_groups: dict[tuple[int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in proposals:
        if row["sdc_mode"] == "gre" and row["asset_type"] == "fmap":
            gre_groups[(int(row["subject"]), int(row["session"]))][row["fmap_role"]] = row
    for (subject, session), roles in sorted(gre_groups.items()):
        messages: list[str] = []
        if set(roles) != {"magnitude1", "magnitude2", "phasediff"}:
            messages.append(f"component roles={sorted(roles)}")
            components = {}
        else:
            components = {
                role: candidates_by_path.get(row["source_relpath"])
                for role, row in roles.items()
            }
            if any(value is None for value in components.values()):
                messages.append("one or more proposed components are absent from candidate inventory")
        if components and not messages:
            mag1 = components["magnitude1"]
            mag2 = components["magnitude2"]
            phase = components["phasediff"]
            if mag1["is_norm_gre"] != "True" or mag2["is_norm_gre"] != "True":
                messages.append("magnitude pair is not the NORM reconstruction")
            if mag1["series_number"] != mag2["series_number"]:
                messages.append("magnitude1/2 do not share a series number")
            try:
                mag_series = int(mag1["series_number"])
                phase_series = int(phase["series_number"])
                echo1 = float(mag1["echo_time"])
                echo2 = float(mag2["echo_time"])
                phase_echo = float(phase["echo_time"])
            except (TypeError, ValueError) as exc:
                messages.append(f"invalid GRE series/echo metadata: {exc}")
                mag_series = phase_series = -1
                echo1 = echo2 = phase_echo = math.nan
            else:
                if phase_series != mag_series + 1:
                    messages.append("phase series is not immediately after NORM magnitude series")
                if not (0 < echo1 < echo2) or not math.isclose(phase_echo, echo2):
                    messages.append("echo ordering is not magnitude1 < magnitude2 = phase")
            timestamps = []
            for component in components.values():
                match = re.search(r"20\d{12}", component["source_relpath"])
                timestamps.append(match.group(0) if match else "")
            if len(set(timestamps)) != 1 or not timestamps[0]:
                messages.append("GRE components do not share one acquisition timestamp")
        else:
            mag1 = mag2 = phase = {}
            mag_series = phase_series = -1
            echo1 = echo2 = phase_echo = math.nan
        status = "passed" if not messages else "failed"
        gre_rows.append(
            {
                "subject": subject,
                "session": session,
                "status": status,
                "magnitude1_relpath": mag1.get("source_relpath", ""),
                "magnitude2_relpath": mag2.get("source_relpath", ""),
                "phasediff_relpath": phase.get("source_relpath", ""),
                "magnitude_series": mag_series if mag_series >= 0 else "",
                "phase_series": phase_series if phase_series >= 0 else "",
                "echo_time1": echo1 if math.isfinite(echo1) else "",
                "echo_time2": echo2 if math.isfinite(echo2) else "",
                "phase_echo_time": phase_echo if math.isfinite(phase_echo) else "",
                "message": "; ".join(messages),
            }
        )
        if status == "passed":
            for row in roles.values():
                row["approval_status"] = "reviewed_candidate"
                row["notes"] += "; GRE NORM/series/echo/timestamp triplet audit passed"

    write_tsv(args.report_dir / "gre_pair_audit.tsv", gre_rows)
    write_tsv(args.report_dir / "review_draft_selection.tsv", proposals, SELECTION_COLUMNS)
    blank_sources = [row["asset_id"] for row in proposals if not row["source_relpath"]]
    status_counts = Counter(row["approval_status"] for row in proposals)
    summary = {
        "rows": len(proposals),
        "status_counts": dict(status_counts),
        "manual_decisions": len(decisions),
        "manual_decision_errors": decision_errors,
        "gre_sessions": len(gre_rows),
        "gre_sessions_passed": sum(row["status"] == "passed" for row in gre_rows),
        "blank_source_assets": blank_sources,
        "approved_rows": status_counts.get("approved", 0),
        "selection_schema_columns": SELECTION_COLUMNS,
        "source_json_identities_frozen": sum(
            bool(row["source_json_relpath"])
            and bool(row["source_json_size_bytes"])
            and bool(row["source_json_sha256"])
            for row in proposals
        ),
        "bold_onset_mat_identities_frozen": sum(
            row["asset_type"] == "bold"
            and bool(row["onset_mat_relpath"])
            and bool(row["onset_mat_size_bytes"])
            and bool(row["onset_mat_sha256"])
            for row in proposals
        ),
        "nifti_hashes_frozen": sum(bool(row["source_sha256"]) for row in proposals),
        "stimuli_600trials_csv_identity": config["input_identities"][
            "stimuli_600trials_csv"
        ],
        "production_ready": False,
        "note": "This is a review draft, not the production source_selection.tsv.",
    }
    (args.report_dir / "review_draft_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if not decision_errors and not blank_sources and all(
        row["status"] == "passed" for row in gre_rows
    ) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--stimuli-csv", type=Path)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--subject", type=int, action="append")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("discover")
    subparsers.add_parser("onsets")
    subparsers.add_parser("review-draft")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    validate.add_argument("--verify-hashes", action="store_true")
    validate.add_argument("--skip-onsets", action="store_true")
    validate.add_argument(
        "--allow-proposed",
        action="store_true",
        help="Development only: do not require approval_status=approved (builder never uses this).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "discover":
        raise SystemExit(command_discover(args))
    if args.command == "onsets":
        raise SystemExit(command_onsets(args))
    if args.command == "review-draft":
        raise SystemExit(command_review_draft(args))
    raise SystemExit(command_validate(args))


if __name__ == "__main__":
    main()
