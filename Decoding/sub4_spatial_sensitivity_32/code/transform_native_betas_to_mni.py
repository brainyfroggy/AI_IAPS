#!/usr/bin/env python3
"""Normalize native-grid single-trial beta maps to the 2-mm MNI target.

The same fixed run-01 functional-to-T1w affine used for the acquired-grid
Kastner atlas is composed with Subject 4's T1w-to-MNI nonlinear transform.
Outputs are auditable 4D beta maps plus a nearest-neighbor support mask.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

from run_glmsingle_space import discover_branch


def one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {description}; found {paths}")
    return paths[0]


def wsl_path(path: Path) -> str:
    full = os.path.abspath(str(path))
    drive, tail = os.path.splitdrive(full)
    if not drive:
        raise ValueError(f"Path lacks drive letter: {path}")
    normalized = tail.lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive.rstrip(':').lower()}/{normalized}"


def materialize_glmsingle(root: Path, output: Path) -> tuple[Path, Path, Path]:
    mask_path = root / "analysis_mask.nii.gz"
    mask_image = nib.load(mask_path)
    mask = np.asarray(mask_image.dataobj) > 0
    flat = np.load(root / "flat_mask_indices.npy").astype(np.int64)
    if not np.array_equal(flat, np.flatnonzero(mask.ravel())):
        raise RuntimeError("GLMsingle mask index mismatch")
    destination = output / "native_betas.nii"
    data = np.zeros((*mask.shape, 600), dtype=np.float32)
    with h5py.File(root / "glmsingle" / "TYPED_FITHRF_GLMDENOISE_RR.hdf5", "r") as handle:
        betas = handle["betasmd"]
        if betas.shape != (len(flat), 1, 1, 600):
            raise RuntimeError(f"Unexpected GLMsingle beta shape: {betas.shape}")
        flat_output = data.reshape((-1, 600))
        for start in range(0, len(flat), 8192):
            stop = min(start + 8192, len(flat))
            flat_output[flat[start:stop], :] = np.asarray(betas[start:stop, 0, 0, :])
    header = mask_image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, mask_image.affine, header), destination)
    manifest = root / "trial_manifest.tsv"
    return destination, mask_path, manifest


def materialize_spm(root: Path, manifest: Path, output: Path) -> tuple[Path, Path, Path]:
    index = pd.read_csv(root / "trial_beta_index.tsv", sep="\t").sort_values("beta_index")
    if index["beta_index"].astype(int).tolist() != list(range(600)):
        raise RuntimeError("SPM beta index is not exactly 0..599")
    paths = [Path(value) for value in index["beta_file"]]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("At least one SPM beta is missing")
    reference = nib.load(paths[0])
    data = np.empty((*reference.shape[:3], 600), dtype=np.float32)
    for number, path in enumerate(paths):
        image = nib.load(path)
        if image.shape[:3] != reference.shape[:3] or not np.allclose(
            image.affine, reference.affine, atol=1e-5, rtol=0
        ):
            raise RuntimeError(f"SPM beta grid mismatch: {path}")
        data[..., number] = np.asarray(image.dataobj, dtype=np.float32)
    destination = output / "native_betas.nii"
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, reference.affine, header), destination)
    mask_path = root / "mask.nii"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing SPM analysis mask: {mask_path}")
    return destination, mask_path, manifest


def ants_apply(
    image: Path,
    reference: Path,
    output: Path,
    coreg: Path,
    t1w_to_mni: Path,
    interpolation: str,
    time_series: bool,
) -> None:
    def image_suffix(path: Path) -> str:
        return ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix

    input_container = "/input_image" + image_suffix(image)
    reference_container = "/reference_image" + image_suffix(reference)
    command = ["wsl", "-d", "Ubuntu-22.04", "--", "docker", "run", "--rm"]
    command.extend(["-v", f"{wsl_path(image)}:{input_container}:ro"])
    command.extend(["-v", f"{wsl_path(reference)}:{reference_container}:ro"])
    command.extend(["-v", f"{wsl_path(output.parent)}:/output"])
    command.extend(["-v", f"{wsl_path(coreg)}:/coreg_transform.txt:ro"])
    command.extend(["-v", f"{wsl_path(t1w_to_mni)}:/anat_transform.h5:ro"])
    command.extend(
        [
            "--entrypoint", "antsApplyTransforms", "nipreps/fmriprep:25.1.3",
            "-d", "3",
            "-i", input_container,
            "-r", reference_container,
            "-o", f"/output/{output.name}",
            "-n", interpolation,
            "-t", "/anat_transform.h5",
            "-t", "/coreg_transform.txt",
        ]
    )
    if time_series:
        command.extend(["-e", "3"])
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", choices=["glmsingle_typed", "spm_lsa_historical_model"], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fmriprep-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    if args.estimator == "glmsingle_typed":
        native_betas, native_mask, manifest = materialize_glmsingle(args.input_root, args.output)
    else:
        if args.manifest is None:
            raise ValueError("--manifest is required for SPM")
        native_betas, native_mask, manifest = materialize_spm(
            args.input_root, args.manifest, args.output
        )

    acquired = discover_branch(args.fmriprep_root, 4, "bold_acquired_grid")
    target = discover_branch(args.fmriprep_root, 4, "mni_res_2")[0]["mask"]
    run1 = acquired[0]["bold"]
    coreg = one(
        list(run1.parent.glob(run1.name.split("_desc-preproc_bold", 1)[0] + "_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt")),
        "run-01 boldref-to-T1w affine",
    )
    anat = args.fmriprep_root / "sub-04" / "anat"
    t1w_to_mni = one(
        list(anat.glob("*from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5")),
        "T1w-to-MNI transform",
    )
    transformed_betas = args.output / "transformed_betas.nii.gz"
    transformed_mask = args.output / "transformed_mask.nii.gz"
    ants_apply(native_betas, target, transformed_betas, coreg, t1w_to_mni, "Linear", True)
    ants_apply(native_mask, target, transformed_mask, coreg, t1w_to_mni, "GenericLabel", False)
    shutil.copy2(manifest, args.output / "trial_manifest.tsv")

    beta_image = nib.load(transformed_betas)
    mask_image = nib.load(transformed_mask)
    reference = nib.load(target)
    if beta_image.shape != (*reference.shape[:3], 600):
        raise RuntimeError(f"Unexpected transformed beta shape: {beta_image.shape}")
    if mask_image.shape[:3] != reference.shape[:3]:
        raise RuntimeError("Transformed mask shape differs from reference")
    for image in (beta_image, mask_image):
        if not np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0):
            raise RuntimeError("Transformed output affine differs from MNI reference")
    mask_values = np.asarray(mask_image.dataobj)
    if np.count_nonzero(mask_values > 0) < 1000:
        raise RuntimeError("Transformed support mask is unexpectedly small")
    provenance = {
        "analysis_id": "ai_iaps_sub04_native_beta_to_mni_v1",
        "estimator": args.estimator,
        "input_root": str(args.input_root),
        "source_beta_grid": "fMRIPrep func/acquired grid",
        "target": str(target),
        "target_shape": list(reference.shape[:3]),
        "target_voxel_sizes_mm": [float(value) for value in reference.header.get_zooms()[:3]],
        "beta_interpolation": "Linear",
        "mask_interpolation": "GenericLabel",
        "transforms_in_ants_cli_order": [str(t1w_to_mni), str(coreg)],
        "functional_to_t1w_reference_run": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"NATIVE_BETA_TO_MNI_COMPLETE {args.estimator} {args.output}")


if __name__ == "__main__":
    main()
