#!/usr/bin/env python3
"""Sequentially orchestrate the exact frozen 25-subject production cohort.

Before executing anything, the orchestrator validates all subjects, all stage
contracts, and the production approval.  It then visits every subject in the
frozen order and every lifecycle stage in order.  Passed stages are announced
explicitly; failed/blocked stages stop the batch.  There is no skip option.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from audit_stage_runner import verify_attempt_evidence, verify_event_mirror
from production_control import (
    DEFAULT_APPROVAL,
    DEFAULT_CONTRACT,
    DEFAULT_FREEZE,
    DEFAULT_LOG_ROOT,
    ControlError,
    build_control_identity,
    ensure_exact_subjects,
    latest_stage_completions,
    read_ledger,
    validate_approval,
)


RUNNER = Path(__file__).resolve().with_name("audit_stage_runner.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        help="Must reproduce the exact frozen 25-subject list and order; omission is forbidden",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-failed", action="store_true")
    parser.add_argument(
        "--ack-analysis-id",
        help="Required with --execute and must equal the approved analysis_id",
    )
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
        subjects = args.subjects or list(approval["production_subjects"])
        ensure_exact_subjects(subjects, approval["production_subjects"])
        blocked = {
            name: stage["blocked_reason"]
            for name, stage in contract["stages"].items()
            if stage["state"] != "executable"
        }
        if blocked:
            raise ControlError(
                "The full lifecycle is not executable; nothing was launched. "
                f"Blocked stages: {blocked}"
            )
        if args.execute and args.ack_analysis_id != approval["analysis_id"]:
            raise ControlError(
                "--execute requires --ack-analysis-id equal to the approved analysis_id"
            )

        log_root = args.log_root.resolve()
        state = read_ledger(log_root / "ledger.jsonl")
        verify_event_mirror(log_root, state)
        verify_attempt_evidence(state)
        completions = latest_stage_completions(state.events)
        mismatched = [
            (subject, stage)
            for (subject, stage), event in completions.items()
            if event.get("control_identity_sha256") != identity_sha256
        ]
        if mismatched:
            raise ControlError(
                f"Ledger contains completions from a different control identity: {mismatched}"
            )
        plan = [(subject, stage) for subject in subjects for stage in contract["stage_order"]]
        print(
            f"CONTROL_PASS analysis={approval['analysis_id']} subjects={len(subjects)} "
            f"stages={len(contract['stage_order'])} attempts={len(plan)}"
        )
        if not args.execute:
            for subject, stage in plan:
                status = completions.get((subject, stage), {}).get("status", "pending")
                print(f"PLAN Sub{subject:02d} {stage}: {status}")
            print("PLAN_ONLY: pass --execute with the exact --ack-analysis-id to launch")
            return 0

        for subject, stage in plan:
            # Refresh after every long-running stage so externally changed or corrupt
            # history cannot be missed.
            state = read_ledger(log_root / "ledger.jsonl")
            verify_event_mirror(log_root, state)
            verify_attempt_evidence(state)
            completion = latest_stage_completions(state.events).get((subject, stage))
            if completion and completion.get("control_identity_sha256") != identity_sha256:
                raise ControlError(
                    f"Sub{subject} {stage} completion belongs to a different control identity"
                )
            if completion and completion.get("status") == "passed":
                print(
                    f"VERIFIED-COMPLETE Sub{subject:02d} {stage} "
                    f"attempt={completion.get('attempt_id')}",
                    flush=True,
                )
                continue
            command = [
                sys.executable,
                str(RUNNER),
                "--subject",
                str(subject),
                "--stage",
                stage,
                "--freeze",
                str(args.freeze),
                "--contract",
                str(args.contract),
                "--approval",
                str(args.approval),
                "--log-root",
                str(args.log_root),
            ]
            if completion is not None:
                if not args.resume_failed:
                    raise ControlError(
                        f"Sub{subject} {stage} previously {completion.get('status')}; "
                        "audit it and pass --resume-failed"
                    )
                command.append("--resume")
            print(f"LAUNCH Sub{subject:02d} {stage}", flush=True)
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise ControlError(
                    f"Batch stopped at Sub{subject:02d} {stage}; exit={result.returncode}"
                )
        print("FULL_BATCH_COMPLETE: all 25 subjects passed every lifecycle stage")
        return 0
    except (ControlError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
