#!/usr/bin/env python3
"""Create the immutable production approval after every launch gate is frozen.

The output is created exclusively and is never overwritten.  This command is
expected to fail during prelaunch: it requires a production-status freeze, an
exact 25-subject approved source manifest with ten BOLD runs per subject, a
fully executable stage contract, and all referenced code/tool files present.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from production_control import (
    DEFAULT_APPROVAL,
    DEFAULT_CONTRACT,
    DEFAULT_FREEZE,
    ControlError,
    file_record,
    load_contract,
    read_freeze,
    validate_source_selection_for_approval,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT_CONFIG = ROOT / "config" / "cohort.json"
DEFAULT_SOURCE_SELECTION = ROOT / "manifests" / "source_selection.tsv"


def write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_approval(args: argparse.Namespace) -> dict:
    freeze, freeze_record = read_freeze(args.freeze, require_production=True)
    contract, contract_record = load_contract(args.contract)
    if contract.get("analysis_id") != freeze["analysis_id"]:
        raise ControlError("Stage contract and freeze analysis_id differ")
    blocked = {
        name: stage.get("blocked_reason", "unspecified")
        for name, stage in contract["stages"].items()
        if stage["state"] != "executable"
    }
    if blocked:
        raise ControlError(f"Cannot approve while stage contracts are blocked: {blocked}")
    remaining = list(freeze["remaining_batch_subjects"])
    manifest_summary = validate_source_selection_for_approval(
        args.source_selection, remaining
    )
    cohort_record = file_record(args.cohort_config)
    source_record = file_record(args.source_selection)
    approved_files = {
        "freeze": freeze_record,
        "stage_contract": contract_record,
        "canonical_cohort_config": file_record(
            Path(str(freeze["canonical_cohort_config"]))
        ),
        "cohort_config": cohort_record,
        "source_selection": source_record,
    }
    for name, path_text in contract["pin_files"].items():
        approved_files[name] = file_record(Path(path_text))
    return {
        "schema_version": 1,
        "approval_status": "approved_for_production",
        "analysis_id": freeze["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "production_subjects": remaining,
        "stage_order": contract["stage_order"],
        "source_manifest_summary": manifest_summary,
        "approved_files": approved_files,
        "policy": {
            "overwrite_forbidden": True,
            "every_stage_executable_at_approval": True,
            "subject_set_and_order_exact": True,
            "source_rows_explicitly_approved_and_hashed": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--cohort-config", type=Path, default=DEFAULT_COHORT_CONFIG)
    parser.add_argument("--source-selection", type=Path, default=DEFAULT_SOURCE_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        approval = build_approval(args)
        if args.dry_run:
            print(json.dumps(approval, indent=2, sort_keys=True))
            return 0
        write_json_exclusive(args.output, approval)
        print(f"PRODUCTION_APPROVAL_CREATED: {args.output.resolve()}")
        return 0
    except (ControlError, FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
