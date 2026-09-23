#!/usr/bin/env python3
"""Prepare immutable uncompressed inputs and manifests for one SPM LS-A branch."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from run_glmsingle_space import discover_branch


MOTION = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(block):
            digest.update(payload)
    return digest.hexdigest()


def events_path(bids_root: Path, run: int) -> Path:
    hits = list(bids_root.glob(f"sub-04/**/func/*_run-{run:02d}_events.tsv"))
    if len(hits) != 1:
        raise RuntimeError(f"Run {run:02d}: expected one events file, found {hits}")
    return hits[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmriprep-root", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument(
        "--branch",
        choices=["bold_acquired_grid", "subject_t1w", "mni_res_native", "mni_res_2"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite existing preparation: {args.output}")
    args.output.mkdir(parents=True)
    inputs = args.output / "uncompressed_bold"
    motion_dir = args.output / "motion6"
    inputs.mkdir()
    motion_dir.mkdir()

    records = discover_branch(args.fmriprep_root, 4, args.branch)
    config_runs = []
    manifests = []
    for run, record in enumerate(records, start=1):
        source = record["bold"]
        destination = inputs / f"run-{run:02d}_desc-preproc_bold.nii"
        if source.suffix == ".gz":
            with gzip.open(source, "rb") as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        else:
            shutil.copy2(source, destination)

        image = nib.load(destination)
        if image.shape[-1] != 224:
            raise RuntimeError(f"Run {run:02d}: expected 224 volumes, got {image.shape}")

        confounds = pd.read_csv(record["confounds"], sep="\t")
        if len(confounds) != 224 or any(name not in confounds for name in MOTION):
            raise RuntimeError(f"Run {run:02d}: invalid six-parameter motion input")
        motion = confounds[MOTION].fillna(0.0).to_numpy(float)
        motion_path = motion_dir / f"run-{run:02d}_motion6.txt"
        np.savetxt(motion_path, motion, fmt="%.12g", delimiter="\t")

        source_events = events_path(args.bids_root, run)
        events = pd.read_csv(source_events, sep="\t")
        if len(events) != 60 or not np.allclose(events["duration"], 3.0, atol=1e-6):
            raise RuntimeError(f"Run {run:02d}: expected 60 events of duration 3 s")
        events.insert(0, "events_file_row", np.arange(60, dtype=int))
        events = events.sort_values(["onset", "events_file_row"], kind="stable").reset_index(drop=True)
        events.insert(0, "trial_in_run", np.arange(1, 61, dtype=int))
        events.insert(0, "run", run)
        events.insert(0, "beta_index", np.arange((run - 1) * 60, run * 60, dtype=int))
        manifests.append(events)

        config_runs.append(
            {
                "run": run,
                "bold_nii": str(destination),
                "bold_nii_sha256": sha256(destination),
                "motion_txt": str(motion_path),
                "motion_txt_sha256": sha256(motion_path),
                "onsets_seconds": events["onset"].astype(float).tolist(),
                "condition_names": [f"trial_{index}" for index in range(1, 61)],
            }
        )

    manifest = pd.concat(manifests, ignore_index=True)
    if manifest["beta_index"].tolist() != list(range(600)):
        raise RuntimeError("Trial chronology is not exactly 0..599")
    manifest_path = args.output / "trial_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    config = {
        "analysis_id": "ai_iaps_sub04_spm_lsa_historical_model_v1",
        "subject": 4,
        "branch": args.branch,
        "tr_seconds": 1.8,
        "stimulus_duration_seconds": 3.0,
        "runs": config_runs,
        "trial_manifest": str(manifest_path),
        "trial_manifest_sha256": sha256(manifest_path),
        "spm_runtime": "C:\\spm12_complete_r7771, SPM12 r7771, Git 3085dac00ac804adb190a7e82c6ef11866c8af02",
        "historical_runtime_note": "Official package r7771 contains spm_spm.m r7738, matching the historical Subject 4 SPM.mat; MATLAB runtime may differ",
    }
    (args.output / "spm_branch_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
