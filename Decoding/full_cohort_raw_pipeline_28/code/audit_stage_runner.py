#!/usr/bin/env python3
"""Run one exact, approved production stage with immutable audit evidence.

The caller supplies only a subject and stage.  The executable argv is rendered
from a hash-approved contract; arbitrary commands are deliberately impossible.
Every launch is bound to the exact freeze, source manifest, code/config hashes,
current contract-compliant local-storage evidence, prior-stage success, and a hash-chained
append-only ledger.  Resume is allowed only after a failed attempt with the
same command and control signature.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from production_control import (
    DEFAULT_APPROVAL,
    DEFAULT_CONTRACT,
    DEFAULT_FREEZE,
    DEFAULT_LOG_ROOT,
    ControlError,
    LedgerState,
    build_control_identity,
    canonical_json_bytes,
    encode_ledger_event,
    file_record,
    latest_stage_completions,
    read_ledger,
    render_command,
    sha256_bytes,
    sha256_file,
    validate_approval,
    validate_prior_stage,
    validate_stage_files,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def command_display(argv: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(argv))
    return shlex.join(argv)


def write_json_exclusive(path: Path, value: Any) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_jsonl(path: Path, value: Any) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if len(encoded) > 64 * 1024:
        raise ControlError("Ledger record is unexpectedly large")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def acquire_lock(path: Path, value: Mapping[str, Any]) -> None:
    try:
        write_json_exclusive(path, value)
    except FileExistsError as error:
        try:
            existing = path.read_text(encoding="utf-8")[:1000]
        except OSError:
            existing = "<unreadable>"
        raise ControlError(
            f"Lock already exists: {path}. Stale locks are never removed automatically. "
            f"Audit it manually. Existing record: {existing}"
        ) from error


def remove_own_lock(path: Path, token: str) -> None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlError(f"Cannot verify owned lock before release: {path}") from error
    if record.get("token") != token or record.get("pid") != os.getpid():
        raise ControlError(f"Refusing to remove a lock not owned by this process: {path}")
    path.unlink()


def verify_event_mirror(log_root: Path, state: LedgerState) -> None:
    event_root = log_root / "status_history" / "events"
    paths = sorted(event_root.glob("*.json")) if event_root.exists() else []
    if len(paths) != len(state.events):
        raise ControlError(
            "Ledger/event-mirror count mismatch; manual audit required: "
            f"ledger={len(state.events)}, mirrors={len(paths)}"
        )
    for sequence, (path, event) in enumerate(zip(paths, state.events), start=1):
        expected_name = f"{sequence:08d}_{event['event_sha256']}.json"
        if path.name != expected_name:
            raise ControlError(f"Unexpected event-mirror name: {path.name}; expected {expected_name}")
        try:
            mirrored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ControlError(f"Cannot verify event mirror: {path}") from error
        if mirrored != event:
            raise ControlError(f"Event mirror differs from ledger: {path}")


def verify_attempt_evidence(state: LedgerState) -> None:
    """Cross-check every completion event against immutable attempt artifacts."""
    started = {
        str(event["attempt_id"]): event
        for event in state.events
        if event.get("event") == "started"
    }
    completed_ids: set[str] = set()
    for event in state.events:
        if event.get("event") != "completed":
            continue
        attempt_id = str(event.get("attempt_id"))
        if attempt_id in completed_ids:
            raise ControlError(f"Duplicate completion event for attempt {attempt_id}")
        completed_ids.add(attempt_id)
        start = started.get(attempt_id)
        if start is None:
            raise ControlError(f"Completion has no start event: {attempt_id}")
        if event.get("started_event_sha256") != start.get("event_sha256"):
            raise ControlError(f"Completion/start event identity mismatch: {attempt_id}")
        for key in (
            "subject",
            "stage",
            "analysis_id",
            "attempt_signature_sha256",
            "control_identity_sha256",
            "attempt_dir",
        ):
            if event.get(key) != start.get(key):
                raise ControlError(f"Start/completion {key} mismatch: {attempt_id}")
        attempt_dir = Path(str(event["attempt_dir"]))
        attempt_path = attempt_dir / "attempt.json"
        result_path = attempt_dir / "result.json"
        if not attempt_path.is_file() or not result_path.is_file():
            raise ControlError(f"Attempt evidence is incomplete: {attempt_dir}")
        if sha256_file(result_path) != event.get("result_sha256"):
            raise ControlError(f"Result hash differs from completion event: {result_path}")
        try:
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ControlError(f"Cannot parse attempt evidence: {attempt_dir}") from error
        for key in (
            "attempt_id",
            "subject",
            "stage",
            "attempt_signature_sha256",
            "control_identity_sha256",
        ):
            if result.get(key) != event.get(key) or attempt.get(key) != event.get(key):
                raise ControlError(f"Attempt/result/ledger {key} mismatch: {attempt_id}")
        if result.get("status") != event.get("status") or result.get("exit_code") != event.get(
            "exit_code"
        ):
            raise ControlError(f"Result status differs from ledger: {attempt_id}")
        for stream_name in ("stdout", "stderr"):
            approved_stream = result.get(stream_name)
            if not isinstance(approved_stream, dict) or "path" not in approved_stream:
                raise ControlError(f"Result lacks {stream_name} file record: {attempt_id}")
            if file_record(Path(str(approved_stream["path"]))) != approved_stream:
                raise ControlError(f"{stream_name} changed after completion: {attempt_id}")


def append_history_event(
    log_root: Path,
    payload: Mapping[str, Any],
    *,
    expected_head: str | None = None,
) -> dict[str, Any]:
    log_root.mkdir(parents=True, exist_ok=True)
    ledger_lock = log_root / ".ledger.lock"
    lock_token = uuid.uuid4().hex
    acquire_lock(
        ledger_lock,
        {"token": lock_token, "pid": os.getpid(), "acquired_at": utc_now()},
    )
    try:
        ledger_path = log_root / "ledger.jsonl"
        state = read_ledger(ledger_path)
        verify_event_mirror(log_root, state)
        if expected_head is not None and state.head_sha256 != expected_head:
            raise ControlError("Ledger changed concurrently before append")
        event = encode_ledger_event(state, payload)
        event_root = log_root / "status_history" / "events"
        event_root.mkdir(parents=True, exist_ok=True)
        mirror = event_root / f"{event['sequence']:08d}_{event['event_sha256']}.json"
        write_json_exclusive(mirror, event)
        append_jsonl(ledger_path, event)
        return event
    finally:
        remove_own_lock(ledger_lock, lock_token)


def storage_snapshot(contract: Mapping[str, Any]) -> dict[str, Any]:
    gate = contract["storage_gate"]
    target = Path(str(gate["path"]))
    try:
        usage = shutil.disk_usage(target)
    except OSError as error:
        raise ControlError(f"Cannot measure storage gate path {target}: {error}") from error
    minimum = int(gate["minimum_free_bytes"])
    return {
        "checked_at": utc_now(),
        "path": str(target),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "minimum_free_bytes": minimum,
        "passed": int(usage.free) >= minimum,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retry the latest failed stage only if its command/control signature is unchanged",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all controls and print the exact attempt without writing or launching",
    )
    args = parser.parse_args(argv)
    for name in ("freeze", "contract", "approval", "log_root"):
        value = getattr(args, name)
        if "\x00" in str(value):
            parser.error(f"--{name.replace('_', '-')} cannot contain NUL")
    return args


def _current_stage_event(
    events: Sequence[Mapping[str, Any]], subject: int, stage: str
) -> Mapping[str, Any] | None:
    return latest_stage_completions(events).get((subject, stage))


def _incomplete_attempts(
    events: Sequence[Mapping[str, Any]], subject: int, stage: str
) -> list[str]:
    started = {
        str(event["attempt_id"])
        for event in events
        if event.get("event") == "started"
        and event.get("subject") == subject
        and event.get("stage") == stage
    }
    completed = {
        str(event["attempt_id"])
        for event in events
        if event.get("event") == "completed"
        and event.get("subject") == subject
        and event.get("stage") == stage
    }
    return sorted(started - completed)


def execute(args: argparse.Namespace) -> int:
    approval, approval_record, freeze, contract, identities = validate_approval(
        args.approval, args.freeze, args.contract
    )
    if args.subject not in approval["production_subjects"]:
        raise ControlError(
            f"Sub{args.subject} is not in the exact approved 25-subject production batch"
        )
    if args.stage not in contract["stage_order"]:
        raise ControlError(f"Unknown stage {args.stage!r}")

    stage_contract = contract["stages"][args.stage]
    command, variables = render_command(contract, args.stage, args.subject, identities)
    config_records = validate_stage_files(stage_contract, identities)
    executable = Path(command[0])
    if executable.is_absolute() and not executable.is_file():
        raise ControlError(f"Contract executable does not exist: {executable}")

    control_identity, control_identity_sha256 = build_control_identity(
        freeze["analysis_id"], approval_record, identities
    )
    signature_payload = {
        "schema_version": 1,
        "analysis_id": freeze["analysis_id"],
        "subject": args.subject,
        "stage": args.stage,
        "command_argv": command,
        "control_identity_sha256": control_identity_sha256,
        "required_approved_files": config_records,
        "stage_contract": stage_contract,
    }
    attempt_signature = sha256_bytes(canonical_json_bytes(signature_payload))

    log_root = args.log_root.resolve()
    ledger_path = log_root / "ledger.jsonl"
    state = read_ledger(ledger_path)
    verify_event_mirror(log_root, state)
    verify_attempt_evidence(state)
    validate_prior_stage(
        contract,
        state.events,
        args.subject,
        args.stage,
        control_identity_sha256,
    )
    incomplete = _incomplete_attempts(state.events, args.subject, args.stage)
    if incomplete:
        raise ControlError(
            "Stage has started attempts without completion evidence; manual audit is required: "
            f"{incomplete}"
        )
    previous = _current_stage_event(state.events, args.subject, args.stage)
    if previous is not None and previous.get("status") == "passed":
        raise ControlError(
            f"Sub{args.subject} stage {args.stage} already passed in attempt "
            f"{previous.get('attempt_id')}; duplicate execution is forbidden"
        )
    if previous is None and args.resume:
        raise ControlError("--resume was supplied but no completed failed attempt exists")
    if previous is not None and previous.get("status") != "passed":
        if not args.resume:
            raise ControlError(
                "Latest attempt did not pass; an explicit --resume is required after audit"
            )
        if previous.get("attempt_signature_sha256") != attempt_signature:
            raise ControlError(
                "Resume refused: command/config/control signature changed since the failed attempt"
            )

    storage = storage_snapshot(contract)
    attempt_id = f"{timestamp_slug()}_{uuid.uuid4().hex}"
    base = log_root / "subjects" / f"Sub{args.subject}" / "stages" / args.stage
    attempt_dir = base / "attempts" / attempt_id
    runner_record = file_record(Path(__file__))
    plan = {
        "schema_version": 2,
        "attempt_id": attempt_id,
        "attempt_signature_sha256": attempt_signature,
        "analysis_id": freeze["analysis_id"],
        "subject": args.subject,
        "stage": args.stage,
        "created_at": utc_now(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd().resolve()),
        "command_argv": command,
        "command_display": command_display(command),
        "rendered_variables": variables,
        "control_identity": control_identity,
        "control_identity_sha256": control_identity_sha256,
        "approval": approval_record,
        "required_approved_files": config_records,
        "runner": runner_record,
        "storage_gate": storage,
        "resume": bool(args.resume),
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
    }
    if args.dry_run:
        print(json.dumps({**plan, "dry_run": True}, indent=2, sort_keys=True))
        return 0 if storage["passed"] else 2

    subject_lock = log_root / "subjects" / f"Sub{args.subject}" / ".lifecycle.lock"
    subject_lock.parent.mkdir(parents=True, exist_ok=True)
    lock_token = uuid.uuid4().hex
    lock_record = {
        "token": lock_token,
        "attempt_id": attempt_id,
        "pid": os.getpid(),
        "subject": args.subject,
        "stage": args.stage,
        "acquired_at": utc_now(),
    }
    acquire_lock(subject_lock, lock_record)
    try:
        # Revalidate the ledger after lock acquisition to close the subject race.
        current_state = read_ledger(ledger_path)
        verify_event_mirror(log_root, current_state)
        verify_attempt_evidence(current_state)
        if current_state.head_sha256 != state.head_sha256:
            # Other subjects may append safely, but re-check this subject's lifecycle.
            validate_prior_stage(
                contract,
                current_state.events,
                args.subject,
                args.stage,
                control_identity_sha256,
            )
            current_incomplete = _incomplete_attempts(
                current_state.events, args.subject, args.stage
            )
            if current_incomplete:
                raise ControlError(
                    "Stage acquired incomplete history while waiting for the lifecycle lock: "
                    f"{current_incomplete}"
                )
            current_previous = _current_stage_event(
                current_state.events, args.subject, args.stage
            )
            if current_previous != previous:
                raise ControlError("Subject stage history changed concurrently")

        attempt_dir.mkdir(parents=True, exist_ok=False)
        write_json_exclusive(attempt_dir / "attempt.json", plan)
        started = append_history_event(
            log_root,
            {
                "event": "started",
                "attempt_id": attempt_id,
                "attempt_signature_sha256": attempt_signature,
                "control_identity_sha256": control_identity_sha256,
                "analysis_id": freeze["analysis_id"],
                "subject": args.subject,
                "stage": args.stage,
                "timestamp": utc_now(),
                "attempt_dir": str(attempt_dir),
                "storage_free_bytes": storage["free_bytes"],
                "storage_minimum_free_bytes": storage["minimum_free_bytes"],
            },
        )

        started_at = utc_now()
        interrupted = False
        exit_code: int | None = None
        spawn_error: str | None = None
        status = "blocked"
        if storage["passed"]:
            with (attempt_dir / "stdout.log").open("xb") as stdout_handle, (
                attempt_dir / "stderr.log"
            ).open("xb") as stderr_handle:
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        shell=False,
                    )
                    try:
                        exit_code = process.wait()
                    except KeyboardInterrupt:
                        interrupted = True
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        exit_code = 130
                except OSError as error:
                    spawn_error = f"{type(error).__name__}: {error}"
                    exit_code = 127
                    stderr_handle.write((spawn_error + "\n").encode("utf-8"))
            status = "passed" if exit_code == 0 else "failed"
        else:
            exit_code = 2
            message = (
                f"STORAGE GATE FAILED: {storage['free_bytes']} bytes free at {storage['path']}; "
                f"minimum={storage['minimum_free_bytes']} bytes\n"
            )
            (attempt_dir / "stdout.log").write_bytes(b"")
            (attempt_dir / "stderr.log").write_text(message, encoding="utf-8")

        completed_at = utc_now()
        stdout_record = file_record(attempt_dir / "stdout.log")
        stderr_record = file_record(attempt_dir / "stderr.log")
        result = {
            "schema_version": 2,
            "attempt_id": attempt_id,
            "attempt_signature_sha256": attempt_signature,
            "control_identity_sha256": control_identity_sha256,
            "subject": args.subject,
            "stage": args.stage,
            "started_at": started_at,
            "completed_at": completed_at,
            "exit_code": exit_code,
            "status": status,
            "interrupted": interrupted,
            "spawn_error": spawn_error,
            "storage_gate_passed": storage["passed"],
            "stdout": stdout_record,
            "stderr": stderr_record,
        }
        write_json_exclusive(attempt_dir / "result.json", result)
        append_history_event(
            log_root,
            {
                "event": "completed",
                "attempt_id": attempt_id,
                "attempt_signature_sha256": attempt_signature,
                "control_identity_sha256": control_identity_sha256,
                "analysis_id": freeze["analysis_id"],
                "subject": args.subject,
                "stage": args.stage,
                "timestamp": completed_at,
                "exit_code": exit_code,
                "status": status,
                "attempt_dir": str(attempt_dir),
                "started_event_sha256": started["event_sha256"],
                "result_sha256": sha256_file(attempt_dir / "result.json"),
            },
        )
        return int(exit_code if exit_code is not None else 127)
    finally:
        remove_own_lock(subject_lock, lock_token)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return execute(parse_args(argv))
    except (ControlError, ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
