#!/usr/bin/env python3
"""Onboard Subject 34 as the 30th cohort member, following the exact same procedure used for
Sub33 in onboard_sub33.py (see that file's module docstring for the full rationale: it bypasses
full_cohort_raw_pipeline_28/code/cohortlib.py's load_config() gate, which is hard-pinned to the
original 25-subject config and would reject any new subject, by building a minimal Sub34-only
config view and calling cohortlib's pure functions directly).

Differences from Sub33's onboarding:
  - BOTH sessions arrived as raw DICOM here (Sub33 only had session 2 raw); both were converted
    with dcm2niix v1.0.20260724 (same version, same pilot venv) into
    LAB_IAPS_AI/rawfMRI/IAPS-DEV/Dev_034/ and .../Dev_034_02/, gzip-compressed (-z i, internal
    zlib -- no pigz installed) to match every other subject's .nii.gz convention, with output
    filenames renamed from dcm2niix's default "DICOM_<protocol>_<time>_<series>" (since %f
    resolved to the DICOM/ subfolder's own name) to "Dev_034_<protocol>_<time>_<series>" to match
    the Dev_033/Dev_033_2 naming convention. Renaming only touches the filename prefix, not file
    content, so this is safe -- classify_source() reads SeriesDescription/ProtocolName from the
    JSON sidecar content, not the filename's subject-prefix.
  - Session 1 (Dev_034) DICOM had two real-world wrinkles the review stage must resolve, not
    silently pick: (1) RUN1 has 5 series instead of the usual 2 (real + MoCo companion) -- two
    218-volume series (8, 9), two truncated 18-volume series (6, 7, with a dcm2niix partial-volume
    warning), and a 2-volume series (1006), consistent with a false start / restart during
    scanning; (2) two GRE fieldmap magnitude pairs (series 20 e1/e2 and 21 e1/e2) but only one
    phasediff (series 22). The stage_review GRE-triplet audit (phase_series == mag_series + 1)
    should select series 21 as the correct magnitude pair; the false-start RUN1 series must be
    resolved via manual_resolution_decisions.tsv before approval.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

FULL_COHORT_CODE = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/full_cohort_raw_pipeline_28/code")
sys.path.insert(0, str(FULL_COHORT_CODE))

from cohortlib import (  # noqa: E402
    SELECTION_COLUMNS,
    audit_selection,
    sha256_file,
    write_tsv,
)
import build_full_cohort_bids as bids_builder  # noqa: E402
import audit_full_cohort_sources as audit_tool  # noqa: E402

NEW_PIPELINE_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")
COHORT_28_CONFIG = NEW_PIPELINE_ROOT / "config" / "cohort_28.json"
RAW_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/LAB_IAPS_AI/rawfMRI/IAPS-DEV")
LOG_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/LAB_IAPS_AI/DataRecording")
STIMULI_CSV = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/stimuli_600trials.csv")
UNIFIED_BIDS_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/LAB_IAPS_AI/bids_unified_glmsingle_28")

REPORT_DIR = NEW_PIPELINE_ROOT / "audit" / "sub34_source_inventory"
SELECTION_PATH = NEW_PIPELINE_ROOT / "manifests" / "sub34_source_selection.tsv"
STAGING_BIDS_OUTPUT = NEW_PIPELINE_ROOT / "bids_sub34_staging"

SUBJECT = 34


def sub34_config() -> dict:
    full = json.loads(COHORT_28_CONFIG.read_text(encoding="utf-8"))
    entry = full["subjects"]["34"]
    return {
        "schema_version": 2,
        "cohort": {"remaining_subjects": [SUBJECT]},
        "paths": {
            "raw_root": str(RAW_ROOT),
            "log_root": str(LOG_ROOT),
            "stimuli_csv": str(STIMULI_CSV),
        },
        "input_identities": full["input_identities"],
        "subjects": {"34": entry},
    }


def stage_discover(config: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = audit_tool.discover_candidates(config, RAW_ROOT, [SUBJECT])
    proposals = audit_tool.propose_selection(config, candidates, LOG_ROOT, [SUBJECT])
    onset_rows = audit_tool.audit_all_onsets(config, LOG_ROOT, STIMULI_CSV, [SUBJECT])
    write_tsv(REPORT_DIR / "candidate_inventory.tsv", candidates, audit_tool.CANDIDATE_COLUMNS)
    write_tsv(REPORT_DIR / "proposed_source_selection.tsv", proposals, SELECTION_COLUMNS)
    write_tsv(REPORT_DIR / "onset_audit.tsv", onset_rows)
    status_counts: dict[str, int] = {}
    for row in proposals:
        status_counts[row["approval_status"]] = status_counts.get(row["approval_status"], 0) + 1
    print(f"candidates={len(candidates)} proposals={len(proposals)} status_counts={status_counts}")
    onset_status = {row["status"] for row in onset_rows}
    print(f"onset rows={len(onset_rows)} statuses={onset_status}")
    for row in proposals:
        if row["approval_status"] not in {"proposed_unique", "review_required_gre_component"}:
            print(f"  UNRESOLVED {row['asset_id']}: {row['approval_status']} ({row['notes']})")


def stage_review(config: dict) -> None:
    proposal_path = REPORT_DIR / "proposed_source_selection.tsv"
    candidates_path = REPORT_DIR / "candidate_inventory.tsv"
    decisions_path = REPORT_DIR / "manual_resolution_decisions.tsv"
    if not decisions_path.exists():
        write_tsv(decisions_path, [], ["asset_id", "selected_source_relpath", "source_size_bytes"])

    proposals = audit_tool.read_selection(proposal_path)
    candidates = audit_tool._read_tsv(candidates_path)
    decisions = {row["asset_id"]: row for row in audit_tool._read_tsv(decisions_path)}
    candidates_by_path = {row["source_relpath"]: row for row in candidates}
    proposal_by_id = {row["asset_id"]: row for row in proposals}

    decision_errors = []
    for asset_id, decision in decisions.items():
        if asset_id not in proposal_by_id:
            decision_errors.append(f"Decision refers to unknown asset {asset_id}")
            continue
        selected = decision["selected_source_relpath"]
        candidate = candidates_by_path.get(selected)
        if candidate is None:
            decision_errors.append(f"Decision for {asset_id} selects unknown candidate {selected}")
            continue
        row = proposal_by_id[asset_id]
        row["source_relpath"] = candidate["source_relpath"]
        row["source_json_relpath"] = candidate["sidecar_relpath"]
        row["source_size_bytes"] = candidate["source_size_bytes"]
        row["source_json_size_bytes"] = candidate["sidecar_size_bytes"]
        row["approval_status"] = "reviewed_candidate"
        row["notes"] = (row.get("notes", "") + f"; manually resolved via manual_resolution_decisions.tsv: "
                        f"selected {selected} (matches modal 218-volume run length; the other candidates "
                        f"were a 2-volume and an 18-volume truncated acquisition from an apparent scan "
                        f"restart, confirmed by dcm2niix's partial-volume warning on the 18-volume series)")

    import math
    import re
    from collections import defaultdict

    gre_rows = []
    gre_groups: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
    for row in proposals:
        if row["sdc_mode"] == "gre" and row["asset_type"] == "fmap":
            gre_groups[(int(row["subject"]), int(row["session"]))][row["fmap_role"]] = row
    for (subject, session), roles in sorted(gre_groups.items()):
        messages = []
        if set(roles) != {"magnitude1", "magnitude2", "phasediff"}:
            messages.append(f"component roles={sorted(roles)}")
            components = {}
        else:
            components = {role: candidates_by_path.get(row["source_relpath"]) for role, row in roles.items()}
            if any(value is None for value in components.values()):
                messages.append("one or more proposed components are absent from candidate inventory")
        if components and not messages:
            mag1, mag2, phase = components["magnitude1"], components["magnitude2"], components["phasediff"]
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
        status = "passed" if not messages else "failed"
        gre_rows.append({"subject": subject, "session": session, "status": status, "message": "; ".join(messages)})
        if status == "passed":
            for row in roles.values():
                row["approval_status"] = "reviewed_candidate"
                row["notes"] += "; GRE NORM/series/echo/timestamp triplet audit passed"

    for row in proposals:
        if row["asset_type"] in {"bold", "t1w"} and row["approval_status"] == "proposed_unique":
            row["approval_status"] = "reviewed_candidate"

    write_tsv(REPORT_DIR / "gre_pair_audit.tsv", gre_rows)
    write_tsv(REPORT_DIR / "review_draft_selection.tsv", proposals, SELECTION_COLUMNS)
    print(f"gre_sessions={len(gre_rows)} passed={sum(r['status'] == 'passed' for r in gre_rows)}")
    for row in gre_rows:
        if row["status"] != "passed":
            print(f"  GRE FAIL sub-{row['subject']:02d} ses-{row['session']:02d}: {row['message']}")
    blanks = [row["asset_id"] for row in proposals if not row["source_relpath"]]
    unresolved = [row["asset_id"] for row in proposals if row["approval_status"] not in {"reviewed_candidate"}]
    print(f"blank_sources={blanks}")
    print(f"still_unresolved={unresolved}")
    if decision_errors:
        print(f"decision_errors={decision_errors}")


def stage_approve() -> None:
    draft_path = REPORT_DIR / "review_draft_selection.tsv"
    from cohortlib import read_selection

    rows = read_selection(draft_path)
    approved = []
    for row in rows:
        if row["approval_status"] != "reviewed_candidate":
            raise RuntimeError(f"Refusing to approve non-reviewed row {row['asset_id']}: {row['approval_status']}")
        source = RAW_ROOT / row["source_relpath"]
        sidecar = RAW_ROOT / row["source_json_relpath"]
        row = dict(row)
        row["source_sha256"] = sha256_file(source)
        row["source_json_sha256"] = sha256_file(sidecar)
        row["approval_status"] = "approved"
        approved.append(row)
    write_tsv(SELECTION_PATH, approved, SELECTION_COLUMNS)
    print(f"wrote {len(approved)} approved rows to {SELECTION_PATH}")


def stage_validate(config: dict) -> None:
    rows, issues = audit_selection(
        config,
        SELECTION_PATH,
        RAW_ROOT,
        LOG_ROOT,
        STIMULI_CSV,
        [SUBJECT],
        require_approved=True,
        check_onsets=True,
        verify_hashes=True,
    )
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    print(f"rows={len(rows)} errors={len(errors)} warnings={len(warnings)}")
    for issue in issues:
        print(f"  {issue.severity.upper()} {issue.code} {issue.asset_id}: {issue.message}")
    if errors:
        raise SystemExit(f"VALIDATE_FAIL: {len(errors)} errors")
    print("VALIDATE_PASS")


def stage_build(config: dict) -> None:
    rows, issues = audit_selection(
        config,
        SELECTION_PATH,
        RAW_ROOT,
        LOG_ROOT,
        STIMULI_CSV,
        [SUBJECT],
        require_approved=True,
        check_onsets=True,
        verify_hashes=True,
    )
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise SystemExit(f"BUILD_ABORTED: {len(errors)} audit errors; see validate stage")
    bids_builder.build_bids(
        config,
        rows,
        RAW_ROOT,
        LOG_ROOT,
        STIMULI_CSV,
        STAGING_BIDS_OUTPUT,
        [SUBJECT],
        link_niftis=False,
        config_path=COHORT_28_CONFIG,
        selection_path=SELECTION_PATH,
    )
    print(f"BUILD_PASS: {STAGING_BIDS_OUTPUT}")


def stage_merge() -> None:
    src = STAGING_BIDS_OUTPUT / "sub-34"
    dst = UNIFIED_BIDS_ROOT / "sub-34"
    if not src.is_dir():
        raise SystemExit(f"Nothing to merge: {src} does not exist (run --stage build first)")
    if dst.exists():
        raise SystemExit(f"Refusing to overwrite existing {dst}")
    shutil.copytree(src, dst, copy_function=shutil.copyfile)
    print(f"MERGE_PASS: {src} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["discover", "review", "approve", "validate", "build", "merge"])
    args = ap.parse_args()
    config = sub34_config()
    if args.stage == "discover":
        stage_discover(config)
    elif args.stage == "review":
        stage_review(config)
    elif args.stage == "approve":
        stage_approve()
    elif args.stage == "validate":
        stage_validate(config)
    elif args.stage == "build":
        stage_build(config)
    elif args.stage == "merge":
        stage_merge()


if __name__ == "__main__":
    main()
