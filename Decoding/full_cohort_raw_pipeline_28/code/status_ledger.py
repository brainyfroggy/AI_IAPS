#!/usr/bin/env python3
"""Verify the immutable stage ledger and report every production subject."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from audit_stage_runner import verify_attempt_evidence, verify_event_mirror
from production_control import (
    DEFAULT_APPROVAL,
    DEFAULT_CONTRACT,
    DEFAULT_FREEZE,
    DEFAULT_LOG_ROOT,
    ControlError,
    build_control_identity,
    latest_stage_completions,
    read_ledger,
    validate_approval,
)


def status_rows(
    approval: dict,
    contract: dict,
    events: tuple[dict, ...],
    control_identity_sha256: str,
) -> list[dict[str, str]]:
    completions = latest_stage_completions(events)
    rows: list[dict[str, str]] = []
    for subject in approval["production_subjects"]:
        row: dict[str, str] = {"subject": str(subject)}
        reached_gap = False
        for stage in contract["stage_order"]:
            event = completions.get((subject, stage))
            if event is None:
                row[stage] = "pending"
                reached_gap = True
            else:
                if event.get("control_identity_sha256") != control_identity_sha256:
                    raise ControlError(
                        f"Sub{subject} stage {stage} belongs to a different control identity"
                    )
                row[stage] = str(event["status"])
                if reached_gap:
                    raise ControlError(
                        f"Sub{subject} has stage {stage} history after an earlier missing stage"
                    )
                if event["status"] != "passed":
                    reached_gap = True
        row["overall_status"] = (
            "complete"
            if all(row[name] == "passed" for name in contract["stage_order"])
            else "incomplete"
        )
        rows.append(row)
    known_subjects = {int(event["subject"]) for event in events if "subject" in event}
    unexpected = known_subjects - set(approval["production_subjects"])
    if unexpected:
        raise ControlError(f"Ledger contains unexpected subjects: {sorted(unexpected)}")
    return rows


def atomic_write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        approval, approval_record, freeze, contract, identities = validate_approval(
            args.approval, args.freeze, args.contract
        )
        _, identity_sha256 = build_control_identity(
            freeze["analysis_id"], approval_record, identities
        )
        state = read_ledger(args.log_root.resolve() / "ledger.jsonl")
        verify_event_mirror(args.log_root.resolve(), state)
        verify_attempt_evidence(state)
        rows = status_rows(approval, contract, state.events, identity_sha256)
        if len(rows) != 25:
            raise ControlError(f"Status report must contain all 25 subjects; found {len(rows)}")
        fields = ["subject", *contract["stage_order"], "overall_status"]
        if args.output:
            atomic_write_tsv(args.output, rows, fields)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return 0
    except (ControlError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
