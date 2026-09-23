#!/usr/bin/env python3
"""Resume production_v2_40_attempt02 with higher parallelism for the
remaining work, without disturbing whatever the original orchestrator
(PID 42948, run_production_after_fmriprep_attempt02.py) is actively
computing when this script starts.

Why this exists: the original orchestrator hard-codes workers=2 for every
stage. Two SPM LSA jobs (mni_res_native smoothing_3mm and smoothing_5mm) were
already several hours into whole-brain AR(1) estimation when this was
written. Restarting the orchestrator with a higher worker count would kill
those two jobs and lose that compute. Instead, this script:

1. Waits for those two specific jobs to finish *naturally* (detected by the
   appearance of their `provenance.json`, which run_spm_lsa.m writes only at
   the very end -- the output directory itself is created immediately, so it
   is NOT a valid completion marker).
2. Only then stops the original orchestrator process. At that point it has
   at most just started its next two queued jobs (a few seconds old, no real
   compute invested), so nothing of substance is lost.
3. Deletes any stale/incomplete SPM output directories that orchestrator may
   have just created for those next jobs (run_spm_lsa.m refuses to reuse a
   non-empty output directory, so a partial dir left behind by the kill would
   otherwise block reruns).
4. Reruns whatever is genuinely still incomplete -- determined dynamically by
   checking for each stage's real completion marker, not by assuming a fixed
   set of remaining jobs -- with much higher concurrency (28 logical cores,
   ~69 GiB free RAM measured available on this machine at the time of
   writing).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT = Path(
    r"N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding\sub4_spatial_sensitivity_32"
)
CODE = PROJECT / "code"
AI_IAPS = PROJECT.parents[1]
BIDS = AI_IAPS / "Decoding" / "pilot_raw_pipeline_sub4_6" / "bids"
FMRIPREP = PROJECT / "preprocessing" / "fmriprep_multispace_attempt_05"
PRODUCTION = PROJECT / "production_v2_40_attempt02"
LOG_ROOT = PROJECT / "logs" / "production_v2_40_attempt02_finish"
LEDGER = LOG_ROOT / "ledger.jsonl"
ATLAS = AI_IAPS / "Decoding" / "full_cohort_raw_pipeline_28" / "resources" / "kastner" / "kastner.nii.gz"
ATLAS_LABELS = ATLAS.with_name("kastner.nii.txt")
WINDOWS_PYTHON = Path(sys.executable)
SPM_WRAPPER = CODE / "run_spm_lsa_job.ps1"
BRANCHES = ("bold_acquired_grid", "subject_t1w", "mni_res_native", "mni_res_2")
SMOOTHING = (0, 3, 5, 8)
C_HARD_FLOOR_GIB = 50.0

ORIGINAL_ORCHESTRATOR_PID = 42948
PROTECTED_CELLS = [("mni_res_native", 3), ("mni_res_native", 5)]
SPM_WORKERS = 5
# 5-way transform concurrency OOM-killed every container (WSL2 here is capped
# at 32 GiB via .wslconfig, not the host's 128 GiB; antsApplyTransforms on a
# 600-volume 4D image is memory-heavy). Confirmed exit code 137 (SIGKILL/OOM)
# on all 5 jobs at workers=5. Dropping to 2, which is what the original
# orchestrator design assumed for this stage.
TRANSFORM_WORKERS = 2
DECODE_WORKERS = 10


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_ledger(event: dict) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": now(), **event}
    with LEDGER.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def check_c_drive() -> float:
    free = shutil.disk_usage("C:\\").free / 1024**3
    if free < C_HARD_FLOOR_GIB:
        raise RuntimeError(f"C: free space {free:.2f} GiB is below {C_HARD_FLOOR_GIB:g} GiB")
    return free


def wsl_path(path: Path) -> str:
    full = os.path.abspath(str(path))
    drive, tail = os.path.splitdrive(full)
    normalized = tail.lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive.rstrip(':').lower()}/{normalized}"


def run_job(name: str, command: list[str]) -> dict:
    check_c_drive()
    job_log = LOG_ROOT / "jobs" / name
    if job_log.exists():
        raise RuntimeError(f"Job log already exists; refusing ambiguous resume: {job_log}")
    job_log.mkdir(parents=True)
    append_ledger({"event": "JOB_START", "job": name, "command": command})
    started = time.monotonic()
    with (job_log / "stdout.log").open("wb") as stdout, (job_log / "stderr.log").open("wb") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    record = {
        "event": "JOB_END",
        "job": name,
        "exit_code": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
    }
    append_ledger(record)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {job_log}")
    return record


def run_parallel(stage: str, jobs: list[tuple[str, list[str]]], workers: int) -> None:
    if not jobs:
        append_ledger({"event": "STAGE_SKIPPED_ALREADY_DONE", "stage": stage})
        return
    append_ledger({"event": "STAGE_START", "stage": stage, "jobs": len(jobs), "workers": workers})
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, name, command): name for name, command in jobs}
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except Exception:
            for future in futures:
                future.cancel()
            raise
    append_ledger({"event": "STAGE_COMPLETE", "stage": stage})


def python_command(script: str, *arguments: object) -> list[str]:
    return [str(WINDOWS_PYTHON), str(CODE / script), *(str(value) for value in arguments)]


def glmsingle_command(branch: str, output: Path) -> list[str]:
    command_text = " ".join(
        [
            "source /home/yujun/.cache/ai_iaps_pilot_venv/bin/activate && python",
            wsl_path(CODE / "run_glmsingle_space.py"),
            "--fmriprep-root", wsl_path(FMRIPREP),
            "--bids-root", wsl_path(BIDS),
            "--branch", branch,
            "--output", wsl_path(output),
        ]
    )
    return ["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-lc", command_text]


def spm_provenance(branch: str, smoothing: int) -> Path:
    return PRODUCTION / "spm_models" / branch / f"smoothing_{smoothing}mm" / "provenance.json"


def wait_for_protected_cells() -> None:
    append_ledger({"event": "WAIT_PROTECTED_CELLS_START", "cells": PROTECTED_CELLS})
    last_notice = 0.0
    while True:
        paths = [spm_provenance(branch, smoothing) for branch, smoothing in PROTECTED_CELLS]
        if all(path.is_file() for path in paths):
            break
        if time.monotonic() - last_notice > 300:
            append_ledger(
                {
                    "event": "WAIT_PROTECTED_CELLS_HEARTBEAT",
                    "done": [path.is_file() for path in paths],
                }
            )
            last_notice = time.monotonic()
        time.sleep(15)
    append_ledger({"event": "WAIT_PROTECTED_CELLS_COMPLETE"})


def stop_original_orchestrator() -> None:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {ORIGINAL_ORCHESTRATOR_PID} -ErrorAction SilentlyContinue"],
        capture_output=True,
        text=True,
    )
    if str(ORIGINAL_ORCHESTRATOR_PID) not in result.stdout:
        append_ledger({"event": "ORIGINAL_ORCHESTRATOR_ALREADY_EXITED", "pid": ORIGINAL_ORCHESTRATOR_PID})
    else:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {ORIGINAL_ORCHESTRATOR_PID} -Force"],
            check=False,
        )
        append_ledger({"event": "ORIGINAL_ORCHESTRATOR_STOPPED", "pid": ORIGINAL_ORCHESTRATOR_PID})
    kill_orphaned_children()


def kill_orphaned_children() -> None:
    # Stop-Process on the parent python.exe does not kill its spawned
    # pwsh.exe/MATLAB.exe children (no job-object binding). Any child the
    # orchestrator spawned for a *new* SPM cell in the seconds before it was
    # stopped would otherwise keep running orphaned, silently duplicating
    # whatever this script reruns for that same cell. Only ever targets
    # processes for cells that are NOT one of the two protected,
    # already-finished cells.
    #
    # IMPORTANT: matching only on "run_spm_lsa_job.ps1" is NOT sufficient --
    # that string only appears on the pwsh.exe wrapper's own command line.
    # The matlab.exe launcher and the actual MATLAB.exe engine it spawns
    # (parent -> child, two separate processes) carry a *different* command
    # line that instead contains the "run_spm_lsa(...)" MATLAB call with the
    # config/output paths as literal arguments. Missing this once already
    # left two orphaned MATLAB processes running unsupervised with a lock on
    # a partial output file. Match both patterns so every process in the
    # pwsh -> matlab.exe -> MATLAB.exe chain is caught.
    protected_tokens = [f"smoothing_{smoothing}mm" for _, smoothing in PROTECTED_CELLS]
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -in @('pwsh.exe','matlab.exe','MATLAB.exe') -and "
        "($_.CommandLine -like '*run_spm_lsa_job.ps1*' -or $_.CommandLine -like '*run_spm_lsa(*') } | "
        "Select-Object ProcessId, CommandLine | ConvertTo-Json"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True)
    text = result.stdout.strip()
    if not text:
        append_ledger({"event": "NO_ORPHANED_SPM_CHILDREN"})
        return
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        append_ledger({"event": "ORPHAN_SCAN_PARSE_FAILED", "raw": text})
        return
    if isinstance(rows, dict):
        rows = [rows]
    killed = []
    for row in rows:
        command_line = row.get("CommandLine") or ""
        if any(token in command_line for token in protected_tokens):
            continue
        pid = row.get("ProcessId")
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
            check=False,
        )
        killed.append({"pid": pid, "command_line": command_line})
    append_ledger({"event": "KILLED_ORPHANED_SPM_CHILDREN", "killed": killed})


def rmtree_with_retry(directory: Path, attempts: int = 5, delay_seconds: float = 3.0) -> None:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            shutil.rmtree(directory)
            return
        except PermissionError as error:
            last_error = error
            append_ledger(
                {
                    "event": "RMTREE_RETRY",
                    "directory": str(directory),
                    "attempt": attempt + 1,
                    "error": str(error),
                }
            )
            time.sleep(delay_seconds)
    raise RuntimeError(f"Could not remove {directory} after {attempts} attempts") from last_error


def clear_stale_spm_outputs() -> None:
    # Defense in depth: re-scan for orphans immediately before deleting, in
    # case anything survived the earlier kill pass.
    kill_orphaned_children()
    cleared = []
    for branch in BRANCHES:
        for smoothing in SMOOTHING:
            directory = PRODUCTION / "spm_models" / branch / f"smoothing_{smoothing}mm"
            provenance = directory / "provenance.json"
            if directory.exists() and not provenance.is_file():
                rmtree_with_retry(directory)
                cleared.append(str(directory))
    append_ledger({"event": "CLEARED_STALE_SPM_OUTPUTS", "directories": cleared})


def clear_stale_transform_outputs() -> None:
    # transform_native_betas_to_mni.py refuses to run if its --output
    # directory already exists at all (even partially/corrupted from a
    # killed antsApplyTransforms container). provenance.json is written last
    # and is the true completion marker, so any of these 5 output
    # directories that exist without it are a stale/failed attempt: full
    # cleanup and rerun is far simpler and safer than trying to resume a
    # multi-step external-process pipeline mid-way.
    transformed_root = PRODUCTION / "native_beta_to_mni_res_2"
    candidates = [transformed_root / "glmsingle_typed"] + [
        transformed_root / "spm_lsa_historical_model" / f"smoothing_{smoothing}mm" for smoothing in SMOOTHING
    ]
    cleared = []
    for directory in candidates:
        if directory.exists() and not (directory / "provenance.json").is_file():
            rmtree_with_retry(directory)
            cleared.append(str(directory))
    append_ledger({"event": "CLEARED_STALE_TRANSFORM_OUTPUTS", "directories": cleared})


def clear_job_logs_without_output(job_names: list[str]) -> None:
    # A job_log directory from a previous failed attempt at one of these
    # specific jobs would otherwise trip run_job's ambiguous-resume guard.
    # Safe to clear here because we've just independently confirmed (via
    # clear_stale_transform_outputs) that none of these jobs' real output is
    # both present and complete.
    cleared = []
    for name in job_names:
        job_log = LOG_ROOT / "jobs" / name
        if job_log.exists():
            shutil.rmtree(job_log)
            cleared.append(str(job_log))
    append_ledger({"event": "CLEARED_STALE_JOB_LOGS", "directories": cleared})


def remaining_spm_jobs() -> list[tuple[str, list[str]]]:
    jobs = []
    for branch in BRANCHES:
        config = PRODUCTION / "spm_inputs" / branch / "spm_branch_config.json"
        for smoothing in SMOOTHING:
            if spm_provenance(branch, smoothing).is_file():
                continue
            output = PRODUCTION / "spm_models" / branch / f"smoothing_{smoothing}mm"
            jobs.append(
                (
                    f"spm_{branch}_{smoothing}mm",
                    [
                        "C:\\Program Files\\PowerShell\\7\\pwsh.exe",
                        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
                        "-File", str(SPM_WRAPPER),
                        "-ConfigPath", str(config),
                        "-SmoothingFwhmMm", str(smoothing),
                        "-OutputDirectory", str(output),
                    ],
                )
            )
    return jobs


def main() -> None:
    append_ledger(
        {
            "event": "FINISH_RUN_START",
            "python": str(WINDOWS_PYTHON),
            "c_free_gib": check_c_drive(),
            "spm_workers": SPM_WORKERS,
            "transform_workers": TRANSFORM_WORKERS,
            "decode_workers": DECODE_WORKERS,
        }
    )
    try:
        wait_for_protected_cells()
        stop_original_orchestrator()
        clear_stale_spm_outputs()

        run_parallel("spm_lsa_remaining", remaining_spm_jobs(), workers=SPM_WORKERS)

        clear_stale_transform_outputs()
        clear_job_logs_without_output(
            [
                "transform_glmsingle_native_betas_to_mni_res_2",
                *(f"transform_spm_native_betas_{smoothing}mm_to_mni_res_2" for smoothing in SMOOTHING),
            ]
        )

        transformed_root = PRODUCTION / "native_beta_to_mni_res_2"
        transform_jobs = []
        glmsingle_transform_marker = transformed_root / "glmsingle_typed" / "provenance.json"
        if not glmsingle_transform_marker.is_file():
            transform_jobs.append(
                (
                    "transform_glmsingle_native_betas_to_mni_res_2",
                    python_command(
                        "transform_native_betas_to_mni.py",
                        "--estimator", "glmsingle_typed",
                        "--input-root", PRODUCTION / "glmsingle" / "bold_acquired_grid",
                        "--fmriprep-root", FMRIPREP,
                        "--output", transformed_root / "glmsingle_typed",
                    ),
                )
            )
        native_manifest = PRODUCTION / "spm_inputs" / "bold_acquired_grid" / "trial_manifest.tsv"
        for smoothing in SMOOTHING:
            marker = transformed_root / "spm_lsa_historical_model" / f"smoothing_{smoothing}mm" / "provenance.json"
            if marker.is_file():
                continue
            transform_jobs.append(
                (
                    f"transform_spm_native_betas_{smoothing}mm_to_mni_res_2",
                    python_command(
                        "transform_native_betas_to_mni.py",
                        "--estimator", "spm_lsa_historical_model",
                        "--input-root",
                        PRODUCTION / "spm_models" / "bold_acquired_grid" / f"smoothing_{smoothing}mm",
                        "--manifest", native_manifest,
                        "--fmriprep-root", FMRIPREP,
                        "--output",
                        transformed_root / "spm_lsa_historical_model" / f"smoothing_{smoothing}mm",
                    ),
                )
            )
        run_parallel("normalize_native_betas_to_mni_res_2", transform_jobs, workers=TRANSFORM_WORKERS)

        atlas_root = PRODUCTION / "atlases"
        decode_jobs = []
        for branch in BRANCHES:
            branch_atlas = atlas_root / branch / "kastner_labels.nii.gz"
            glm_root = PRODUCTION / "glmsingle" / branch
            for smoothing in SMOOTHING:
                output = PRODUCTION / "results" / "glmsingle_typed" / branch / f"smoothing_{smoothing}mm"
                if (output / "subject_results.csv").is_file():
                    continue
                decode_jobs.append(
                    (
                        f"decode_glmsingle_{branch}_{smoothing}mm",
                        python_command(
                            "decode_roi_singletrial.py",
                            "--estimator", "glmsingle_typed",
                            "--branch", branch,
                            "--smoothing-mm", smoothing,
                            "--input-root", glm_root,
                            "--atlas", branch_atlas,
                            "--atlas-labels", ATLAS_LABELS,
                            "--output", output,
                        ),
                    )
                )
            manifest = PRODUCTION / "spm_inputs" / branch / "trial_manifest.tsv"
            for smoothing in SMOOTHING:
                model = PRODUCTION / "spm_models" / branch / f"smoothing_{smoothing}mm"
                output = PRODUCTION / "results" / "spm_lsa_historical_model" / branch / f"smoothing_{smoothing}mm"
                if (output / "subject_results.csv").is_file():
                    continue
                decode_jobs.append(
                    (
                        f"decode_spm_{branch}_{smoothing}mm",
                        python_command(
                            "decode_roi_singletrial.py",
                            "--estimator", "spm_lsa_historical_model",
                            "--branch", branch,
                            "--smoothing-mm", smoothing,
                            "--input-root", model,
                            "--manifest", manifest,
                            "--atlas", branch_atlas,
                            "--atlas-labels", ATLAS_LABELS,
                            "--output", output,
                        ),
                    )
                )
        added_branch = "native_beta_to_mni_res_2"
        mni_atlas = atlas_root / "mni_res_2" / "kastner_labels.nii.gz"
        transformed_glm = transformed_root / "glmsingle_typed"
        for smoothing in SMOOTHING:
            output = PRODUCTION / "results" / "glmsingle_typed" / added_branch / f"smoothing_{smoothing}mm"
            if (output / "subject_results.csv").is_file():
                continue
            decode_jobs.append(
                (
                    f"decode_glmsingle_{added_branch}_{smoothing}mm",
                    python_command(
                        "decode_roi_singletrial.py",
                        "--estimator", "glmsingle_typed",
                        "--input-format", "transformed_4d",
                        "--branch", added_branch,
                        "--smoothing-mm", smoothing,
                        "--input-root", transformed_glm,
                        "--atlas", mni_atlas,
                        "--atlas-labels", ATLAS_LABELS,
                        "--output", output,
                    ),
                )
            )
        for smoothing in SMOOTHING:
            transformed_spm = transformed_root / "spm_lsa_historical_model" / f"smoothing_{smoothing}mm"
            output = PRODUCTION / "results" / "spm_lsa_historical_model" / added_branch / f"smoothing_{smoothing}mm"
            if (output / "subject_results.csv").is_file():
                continue
            decode_jobs.append(
                (
                    f"decode_spm_{added_branch}_{smoothing}mm",
                    python_command(
                        "decode_roi_singletrial.py",
                        "--estimator", "spm_lsa_historical_model",
                        "--input-format", "transformed_4d",
                        "--branch", added_branch,
                        "--smoothing-mm", smoothing,
                        "--input-root", transformed_spm,
                        "--atlas", mni_atlas,
                        "--atlas-labels", ATLAS_LABELS,
                        "--output", output,
                    ),
                )
            )
        run_parallel("single_trial_roi_decoding", decode_jobs, workers=DECODE_WORKERS)

        subject_files = sorted((PRODUCTION / "results").glob("**/subject_results.csv"))
        fold_files = sorted((PRODUCTION / "results").glob("**/fold_results.csv"))
        if len(subject_files) != 40 or len(fold_files) != 40:
            raise RuntimeError(
                f"Expected 40 result files; subject={len(subject_files)}, fold={len(fold_files)}"
            )
        subjects = pd.concat([pd.read_csv(path) for path in subject_files], ignore_index=True)
        folds = pd.concat([pd.read_csv(path) for path in fold_files], ignore_index=True)
        if len(subjects) != 5440 or len(folds) != 54400:
            raise RuntimeError(f"Unexpected aggregate sizes: {len(subjects)}, {len(folds)}")
        summary = PRODUCTION / "summary"
        summary.mkdir(exist_ok=True)
        subjects.to_csv(summary / "all_subject_results.csv", index=False)
        folds.to_csv(summary / "all_fold_results.csv", index=False)
        completion = {
            "analysis_id": "ai_iaps_sub04_spatial_estimator_sensitivity_v2_40",
            "completed_at_utc": now(),
            "n_setups": 40,
            "n_subject_result_rows": len(subjects),
            "n_fold_result_rows": len(folds),
            "c_free_gib": check_c_drive(),
        }
        (PRODUCTION / "FULL_40_COMPLETE.json").write_text(
            json.dumps(completion, indent=2) + "\n", encoding="utf-8"
        )
        append_ledger({"event": "FULL_40_COMPLETE", **completion})
    except Exception as error:
        append_ledger(
            {
                "event": "RUN_FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise


if __name__ == "__main__":
    main()
