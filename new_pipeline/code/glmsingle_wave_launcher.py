#!/usr/bin/env python3
"""Real, enforced-concurrency GLMsingle launcher for the new_pipeline wave.

Resolves a gap discovered 2026-08-06: wave_orchestrator.py's process_subject()
only ever CONSTRUCTS the GLMsingle command (see launch_glmsingle_command()) and
never executes it -- every real GLMsingle run this wave was launched by hand,
ad hoc, with no shared coordination. That let 3 real GLMsingle jobs run
concurrently on top of an already-running 5-way fMRIPrep batch with nothing
enforcing the design's concurrency cap, and all 3 died silently (free memory
observed down to 2.3GB, load average ~26). Kernel-level OOM-killer evidence
was checked for and NOT found (cgroup memory.events oom_kill=0 across
docker/user.slice/system.slice/user-1000.slice; dmesg has zero OOM messages
across the full WSL2 uptime) -- the exact mechanism isn't fully confirmed
(one live hypothesis: an operator-side TaskStop call against an unrelated
monitoring task took down a shared Windows job object that sibling background
launches also belonged to), but reducing concurrency and adding a real,
enforced pool is a sound mitigation regardless of which explanation is
correct, per the user's considered decision.

This script provides that real pool: a bounded ThreadPoolExecutor, sized from
cohort_waves.json's concurrency.glmsingle.workers (3, lowered from 5 on
2026-08-06), where each worker blocks on a real subprocess.run() of
run_glmsingle_wave.py for one subject via the WSL venv until that subject's
GLMsingle run actually finishes -- so at most N are ever running at once,
enforced by Python's own executor, not by hoping nobody launches an extra one
by hand.

Fail-closed: refuses to launch a subject whose fMRIPrep final_root is missing,
or whose GLMsingle output directory already exists (never silently overwrites
a completed or partial run -- clear a stale directory explicitly first, same
as every other stage in this project). Appends to the same JSONL ledger
wave_orchestrator.py uses, under a "glmsingle" stage, so a subject's full
history reads from one place.

Deliberately does not use this session's Monitor+TaskStop tool pattern
internally or expect the caller to either -- verify progress via direct,
repeated process/ps checks, consistent with the mitigation above.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

NEW_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_AI_IAPS_ROOT = NEW_PIPELINE_ROOT.parent

DEFAULT_COHORT_CONFIG = NEW_PIPELINE_ROOT / "config" / "cohort_28.json"
DEFAULT_WAVES_CONFIG = NEW_PIPELINE_ROOT / "config" / "cohort_waves.json"
DEFAULT_GLMSINGLE_SCRIPT = NEW_PIPELINE_ROOT / "code" / "run_glmsingle_wave.py"
DEFAULT_VENV_PYTHON = "/home/yujun/.cache/ai_iaps_pilot_venv/bin/python"
DEFAULT_WSL_DISTRIBUTION = "Ubuntu-22.04"
DEFAULT_WSL_EXECUTABLE = r"C:\Windows\System32\wsl.exe"

LAB_IAPS_AI_ROOT = Path("N:/Experimental_Data/yujunchen/projects/LAB_IAPS_AI")
DEFAULT_FMRIPREP_DERIVATIVES_ROOT = LAB_IAPS_AI_ROOT / "fmriprep_derivatives" / "unified_glmsingle_28"
DEFAULT_PER_SUBJECT_BIDS_ROOT = LAB_IAPS_AI_ROOT / "bids_unified_glmsingle_28_per_subject"
DEFAULT_GLMSINGLE_OUTPUT_ROOT = NEW_PIPELINE_ROOT / "glmsingle"

LEDGER_PATH = NEW_PIPELINE_ROOT / "logs" / "wave_orchestrator_ledger.jsonl"
LOG_DIR = NEW_PIPELINE_ROOT / "logs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ledger_append(event: Mapping) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = dict(event)
    record.setdefault("ts", utc_now())
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")


def windows_to_wsl(path: Path) -> str:
    # Path.resolve() can reveal the real UNC path behind a mapped drive
    # letter for network shares (observed for N: this session) -- handle
    # both forms, same fix already present in fmriprep_sdc_workflow_v6.py's
    # own windows_to_wsl().
    absolute = Path(os.path.abspath(path))
    if absolute.drive.startswith("\\\\"):
        n_target = Path("N:\\").resolve(strict=False)
        try:
            relative = absolute.relative_to(n_target)
        except ValueError as error:
            raise ValueError(f"Cannot map non-N UNC path to WSL: {absolute}") from error
        tail = relative.as_posix().lstrip("/")
        return f"/mnt/n/{tail}"
    drive = absolute.drive.rstrip(":").lower()
    if len(drive) != 1 or not drive.isalpha():
        raise ValueError(f"Cannot map path to WSL: {absolute}")
    tail = absolute.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def load_glmsingle_worker_cap(waves_config: Path) -> int:
    cfg = json.loads(waves_config.read_text(encoding="utf-8"))
    return int(cfg["concurrency"]["glmsingle"]["workers"])


def build_command(
    subject: int,
    cohort_config: Path,
    fmriprep_root: Path,
    bids_root: Path,
    output_root: Path,
    branch: str,
) -> list[str]:
    output = output_root / f"sub-{subject:02d}"
    return [
        DEFAULT_WSL_EXECUTABLE,
        "-d",
        DEFAULT_WSL_DISTRIBUTION,
        "--",
        DEFAULT_VENV_PYTHON,
        windows_to_wsl(DEFAULT_GLMSINGLE_SCRIPT),
        "--cohort-config",
        windows_to_wsl(cohort_config),
        "--subject",
        str(subject),
        "--fmriprep-root",
        windows_to_wsl(fmriprep_root),
        "--bids-root",
        windows_to_wsl(bids_root),
        "--branch",
        branch,
        "--output",
        windows_to_wsl(output),
    ]


def launch_one(
    subject: int,
    cohort_config: Path,
    fmriprep_derivatives_root: Path,
    per_subject_bids_root: Path,
    output_root: Path,
    branch: str,
) -> dict:
    label = f"Sub{subject:02d}"
    fmriprep_root = fmriprep_derivatives_root / label
    bids_root = per_subject_bids_root / label
    output = output_root / f"sub-{subject:02d}"

    if not fmriprep_root.is_dir():
        detail = f"fMRIPrep final_root missing, cannot launch GLMsingle: {fmriprep_root}"
        ledger_append({"subject": subject, "stage": "glmsingle", "status": "refused_missing_fmriprep", "detail": detail})
        return {"subject": subject, "status": "refused_missing_fmriprep", "detail": detail}
    if not bids_root.is_dir():
        detail = f"Per-subject BIDS view missing: {bids_root}"
        ledger_append({"subject": subject, "stage": "glmsingle", "status": "refused_missing_bids_view", "detail": detail})
        return {"subject": subject, "status": "refused_missing_bids_view", "detail": detail}
    if output.exists():
        detail = f"Refusing to overwrite existing GLMsingle output: {output}"
        ledger_append({"subject": subject, "stage": "glmsingle", "status": "refused_existing", "detail": detail})
        return {"subject": subject, "status": "refused_existing", "detail": detail}

    command = build_command(subject, cohort_config, fmriprep_root, bids_root, output_root, branch)
    ledger_append({"subject": subject, "stage": "glmsingle", "status": "started", "command": command})

    stdout_log = LOG_DIR / f"sub{subject:02d}_glmsingle_pool_{utc_now().replace(':', '').replace('-', '').split('.')[0]}.log"
    with stdout_log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)

    if completed.returncode != 0:
        detail = f"GLMsingle failed for {label}, exit code {completed.returncode}; see {stdout_log}"
        ledger_append({"subject": subject, "stage": "glmsingle", "status": "fail", "detail": detail, "log": str(stdout_log)})
        return {"subject": subject, "status": "fail", "detail": detail, "log": str(stdout_log)}

    ledger_append({"subject": subject, "stage": "glmsingle", "status": "pass", "output": str(output), "log": str(stdout_log)})
    return {"subject": subject, "status": "pass", "output": str(output), "log": str(stdout_log)}


def run_pool(
    subjects: list[int],
    cohort_config: Path,
    waves_config: Path,
    fmriprep_derivatives_root: Path,
    per_subject_bids_root: Path,
    output_root: Path,
    branch: str,
    max_workers_override: int | None,
) -> list[dict]:
    max_workers = max_workers_override or load_glmsingle_worker_cap(waves_config)
    ledger_append({"subject": None, "stage": "glmsingle_pool", "status": "started", "subjects": subjects, "max_workers": max_workers})
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(
                launch_one, subject, cohort_config, fmriprep_derivatives_root, per_subject_bids_root, output_root, branch
            ): subject
            for subject in subjects
        }
        for future in as_completed(futures):
            subject = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 - fail closed per subject
                detail = f"{type(error).__name__}: {error}"
                ledger_append({"subject": subject, "stage": "glmsingle", "status": "unhandled_exception", "detail": detail})
                results.append({"subject": subject, "status": "unhandled_exception", "detail": detail})
    ledger_append(
        {
            "subject": None,
            "stage": "glmsingle_pool",
            "status": "finished",
            "subject_statuses": {item["subject"]: item.get("status") for item in results},
        }
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", required=True, type=lambda v: [int(x) for x in v.split(",") if x.strip()])
    parser.add_argument("--cohort-config", type=Path, default=DEFAULT_COHORT_CONFIG)
    parser.add_argument("--waves-config", type=Path, default=DEFAULT_WAVES_CONFIG)
    parser.add_argument("--fmriprep-derivatives-root", type=Path, default=DEFAULT_FMRIPREP_DERIVATIVES_ROOT)
    parser.add_argument("--per-subject-bids-root", type=Path, default=DEFAULT_PER_SUBJECT_BIDS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_GLMSINGLE_OUTPUT_ROOT)
    parser.add_argument("--branch", default="mni_res_native")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override cohort_waves.json's concurrency.glmsingle.workers (rarely needed).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_pool(
        args.subjects,
        args.cohort_config,
        args.waves_config,
        args.fmriprep_derivatives_root,
        args.per_subject_bids_root,
        args.output_root,
        args.branch,
        args.max_workers,
    )
    print(json.dumps(results, indent=2, default=str))
    return 0 if all(item.get("status") == "pass" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
