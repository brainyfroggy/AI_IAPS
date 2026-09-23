#!/usr/bin/env python3
"""Fail-closed validation of the shared Subject 4 multi-space derivatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from run_glmsingle_space import BRANCH_GLOBS, discover_branch


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(block):
            digest.update(payload)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmriprep-root", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite validation: {args.output}")

    description = args.fmriprep_root / "dataset_description.json"
    metadata = json.loads(description.read_text(encoding="utf-8"))
    generated = metadata.get("GeneratedBy", [])
    versions = [str(item.get("Version")) for item in generated if item.get("Name") == "fMRIPrep"]
    if versions != ["25.1.3"]:
        raise RuntimeError(f"Expected fMRIPrep 25.1.3, found {versions}")

    raw_bolds = sorted(args.bids_root.glob("sub-04/**/func/*_bold.nii.gz"))
    if len(raw_bolds) != 10:
        raise RuntimeError(f"Expected ten raw BOLD runs, found {len(raw_bolds)}")
    raw_reference = nib.load(raw_bolds[0])

    report = {
        "validated": True,
        "fmriprep_version": versions[0],
        "fmriprep_dataset_description_sha256": sha256(description),
        "branches": {},
    }
    for branch in BRANCH_GLOBS:
        records = discover_branch(args.fmriprep_root, 4, branch)
        first = nib.load(records[0]["bold"])
        branch_rows = []
        for run, record in enumerate(records, start=1):
            image = nib.load(record["bold"])
            mask = nib.load(record["mask"])
            confounds = pd.read_csv(record["confounds"], sep="\t")
            sidecar = json.loads(record["json"].read_text(encoding="utf-8"))
            if image.shape[-1] != 224 or len(confounds) != 224:
                raise RuntimeError(f"{branch} run {run:02d}: not 224 volumes/confound rows")
            if not np.isclose(float(sidecar.get("RepetitionTime", np.nan)), 1.8, atol=1e-6):
                raise RuntimeError(f"{branch} run {run:02d}: TR is not 1.8 s")
            if image.shape[:3] != first.shape[:3] or not np.allclose(
                image.affine, first.affine, atol=1e-5, rtol=0
            ):
                raise RuntimeError(f"{branch}: run grids are not identical")
            if mask.shape != first.shape[:3] or not np.allclose(mask.affine, first.affine, atol=1e-5):
                raise RuntimeError(f"{branch} run {run:02d}: mask grid mismatch")
            branch_rows.append(
                {
                    "run": run,
                    "bold": str(record["bold"]),
                    "bold_size_bytes": record["bold"].stat().st_size,
                    "bold_sha256": sha256(record["bold"]),
                    "mask": str(record["mask"]),
                    "mask_voxels": int(np.count_nonzero(np.asarray(mask.dataobj) > 0)),
                    "confounds": str(record["confounds"]),
                    "metadata": str(record["json"]),
                }
            )
        if branch == "bold_acquired_grid":
            if first.shape[:3] != raw_reference.shape[:3] or not np.allclose(
                first.affine, raw_reference.affine, atol=1e-5, rtol=0
            ):
                raise RuntimeError("func branch does not preserve the acquired grid")
        if branch == "mni_res_2" and not np.allclose(first.header.get_zooms()[:3], [2, 2, 2]):
            raise RuntimeError("MNI res-2 is not 2 mm isotropic")
        report["branches"][branch] = {
            "shape": list(first.shape),
            "voxel_sizes_mm": [float(value) for value in first.header.get_zooms()[:3]],
            "affine": np.asarray(first.affine).tolist(),
            "runs": branch_rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("FMRIPREP_MULTISPACE_VALIDATION_COMPLETE")


if __name__ == "__main__":
    main()
