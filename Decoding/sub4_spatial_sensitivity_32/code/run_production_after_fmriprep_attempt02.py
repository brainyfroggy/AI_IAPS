#!/usr/bin/env python3
"""Fail-closed, resumable-by-new-namespace production runner for all 40 cells.

This runner waits for the already launched shared fMRIPrep attempt 05. It then
validates all four derivative branches and runs bounded parallel jobs. It never
deletes, overwrites, or automatically resumes a failed job.
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


# Preserve the mapped N: spelling. Path.resolve() converts it to a UNC path,
# which Ubuntu-22.04 cannot map deterministically back to /mnt/n.
PROJECT = Path(os.path.abspath(__file__)).parents[1]
CODE = PROJECT / "code"
AI_IAPS = PROJECT.parents[1]
BIDS = AI_IAPS / "Decoding" / "pilot_raw_pipeline_sub4_6" / "bids"
FMRIPREP = PROJECT / "preprocessing" / "fmriprep_multispace_attempt_05"
FMRIPREP_COMPLETION = PROJECT / "logs" / "fmriprep_multispace" / "attempt_05" / "completion.json"
PRODUCTION = PROJECT / "production_v2_40_attempt02"
LOG_ROOT = PROJECT / "logs" / "production_v2_40_attempt02"
LEDGER = LOG_ROOT / "ledger.jsonl"
ATLAS = AI_IAPS / "Decoding" / "full_cohort_raw_pipeline_28" / "resources" / "kastner" / "kastner.nii.gz"
ATLAS_LABELS = ATLAS.with_name("kastner.nii.txt")
WINDOWS_PYTHON = Path(sys.executable)
SPM_WRAPPER = CODE / "run_spm_lsa_job.ps1"
BRANCHES = ("bold_acquired_grid", "subject_t1w", "mni_res_native", "mni_res_2")
SMOOTHING = (0, 3, 5, 8)
C_HARD_FLOOR_GIB = 50.0


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


def wait_for_fmriprep() -> None:
    append_ledger({"event": "WAIT_FMRIPREP_START", "completion": str(FMRIPREP_COMPLETION)})
    last_notice = 0.0
    while not FMRIPREP_COMPLETION.is_file():
        inspect = subprocess.run(
            ["docker", "container", "inspect", "ai-iaps-sub04-spatial-v1-a05"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if inspect.returncode != 0:
            raise RuntimeError("fMRIPrep container stopped without a completion record")
        if time.monotonic() - last_notice > 600:
            append_ledger({"event": "WAIT_FMRIPREP_HEARTBEAT", "c_free_gib": check_c_drive()})
            last_notice = time.monotonic()
        time.sleep(30)
    completion = json.loads(FMRIPREP_COMPLETION.read_text(encoding="utf-8-sig"))
    if int(completion.get("exit_code", -1)) != 0:
        raise RuntimeError(f"fMRIPrep completion is not successful: {completion}")
    append_ledger({"event": "WAIT_FMRIPREP_COMPLETE", "record": completion})


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


def main() -> None:
    if PRODUCTION.exists() or LEDGER.exists():
        raise RuntimeError(
            f"Production namespace already exists; inspect rather than resume automatically: {PRODUCTION}"
        )
    LOG_ROOT.mkdir(parents=True)
    append_ledger(
        {
            "event": "RUN_START",
            "analysis_id": "ai_iaps_sub04_spatial_estimator_sensitivity_v2_40",
            "python": str(WINDOWS_PYTHON),
            "c_free_gib": check_c_drive(),
        }
    )
    try:
        wait_for_fmriprep()
        PRODUCTION.mkdir()

        validation = PRODUCTION / "validation" / "fmriprep_multispace.json"
        run_job(
            "validate_fmriprep_multispace",
            python_command(
                "validate_fmriprep_multispace.py",
                "--fmriprep-root", FMRIPREP,
                "--bids-root", BIDS,
                "--output", validation,
            ),
        )
        atlas_root = PRODUCTION / "atlases"
        run_job(
            "prepare_kastner_atlases",
            python_command(
                "prepare_kastner_atlases.py",
                "--fmriprep-root", FMRIPREP,
                "--atlas", ATLAS,
                "--atlas-labels", ATLAS_LABELS,
                "--output", atlas_root,
            ),
        )

        spm_input_jobs = []
        for branch in BRANCHES:
            output = PRODUCTION / "spm_inputs" / branch
            spm_input_jobs.append(
                (
                    f"prepare_spm_{branch}",
                    python_command(
                        "prepare_spm_branch.py",
                        "--fmriprep-root", FMRIPREP,
                        "--bids-root", BIDS,
                        "--branch", branch,
                        "--output", output,
                    ),
                )
            )
        run_parallel("prepare_spm_inputs", spm_input_jobs, workers=2)

        glmsingle_jobs = []
        for branch in BRANCHES:
            output = PRODUCTION / "glmsingle" / branch
            glmsingle_jobs.append(
                (f"glmsingle_{branch}", glmsingle_command(branch, output))
            )
        run_parallel("glmsingle", glmsingle_jobs, workers=2)

        spm_jobs = []
        for branch in BRANCHES:
            config = PRODUCTION / "spm_inputs" / branch / "spm_branch_config.json"
            for smoothing in SMOOTHING:
                output = PRODUCTION / "spm_models" / branch / f"smoothing_{smoothing}mm"
                spm_jobs.append(
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
        run_parallel("spm_lsa", spm_jobs, workers=2)

        # Fifth workflow: estimate on the acquired/native grid, explicitly
        # normalize the 600 beta maps to the 2-mm MNI target, then decode there.
        transformed_root = PRODUCTION / "native_beta_to_mni_res_2"
        transform_jobs = [
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
        ]
        native_manifest = PRODUCTION / "spm_inputs" / "bold_acquired_grid" / "trial_manifest.tsv"
        for smoothing in SMOOTHING:
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
        run_parallel("normalize_native_betas_to_mni_res_2", transform_jobs, workers=2)

        decode_jobs = []
        for branch in BRANCHES:
            branch_atlas = atlas_root / branch / "kastner_labels.nii.gz"
            glm_root = PRODUCTION / "glmsingle" / branch
            for smoothing in SMOOTHING:
                output = (
                    PRODUCTION / "results" / "glmsingle_typed" / branch
                    / f"smoothing_{smoothing}mm"
                )
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
                output = (
                    PRODUCTION / "results" / "spm_lsa_historical_model" / branch
                    / f"smoothing_{smoothing}mm"
                )
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
            output = (
                PRODUCTION / "results" / "glmsingle_typed" / added_branch
                / f"smoothing_{smoothing}mm"
            )
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
            transformed_spm = (
                transformed_root / "spm_lsa_historical_model" / f"smoothing_{smoothing}mm"
            )
            output = (
                PRODUCTION / "results" / "spm_lsa_historical_model" / added_branch
                / f"smoothing_{smoothing}mm"
            )
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
        run_parallel("single_trial_roi_decoding", decode_jobs, workers=2)

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
        summary.mkdir()
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
