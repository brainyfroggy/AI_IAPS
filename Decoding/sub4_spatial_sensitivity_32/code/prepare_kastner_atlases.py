#!/usr/bin/env python3
"""Create branch-grid Kastner/Wang label maps with nearest-neighbor transforms."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from run_glmsingle_space import discover_branch


BRANCHES = ("bold_acquired_grid", "subject_t1w", "mni_res_native", "mni_res_2")


def wsl_path(path: Path) -> str:
    # Do not call Path.resolve(): Windows resolves mapped N: to a UNC path,
    # while Ubuntu-22.04 intentionally accesses the same data through /mnt/n.
    full = os.path.abspath(str(path))
    drive, tail = os.path.splitdrive(full)
    drive = drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Path lacks a drive letter: {full}")
    normalized_tail = tail.lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive}/{normalized_tail}"


def single(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"Expected one {pattern}; found {hits}")
    return hits[0]


def ants_apply(
    image: Path,
    reference: Path,
    output: Path,
    transforms: list[str],
) -> None:
    command = [
        "wsl", "-d", "Ubuntu-22.04", "--", "docker", "run", "--rm",
        "-v", f"{wsl_path(image.parent)}:/input:ro",
        "-v", f"{wsl_path(reference.parent)}:/reference:ro",
        "-v", f"{wsl_path(output.parent)}:/output",
    ]
    transform_mounts = []
    transform_arguments = []
    for number, specification in enumerate(transforms, start=1):
        inverse = specification.startswith("[") and specification.endswith(",1]")
        raw = specification[1:-3] if inverse else specification
        path = Path(raw)
        mount = f"/transform-{number}{path.suffix}"
        transform_mounts.extend(["-v", f"{wsl_path(path)}:{mount}:ro"])
        transform_arguments.extend(["-t", f"[{mount},1]" if inverse else mount])
    command.extend(transform_mounts)
    command.extend(
        [
            "--entrypoint", "antsApplyTransforms", "nipreps/fmriprep:25.1.3",
            "-d", "3", "-i", f"/input/{image.name}",
            "-r", f"/reference/{reference.name}",
            "-o", f"/output/{output.name}", "-n", "GenericLabel",
            *transform_arguments,
        ]
    )
    subprocess.run(command, check=True)


def validate_label_map(path: Path, reference: Path, allowed: set[int]) -> dict:
    image = nib.load(path)
    target = nib.load(reference)
    if image.shape != target.shape[:3] or not np.allclose(image.affine, target.affine, atol=1e-5):
        raise RuntimeError(f"Atlas/reference grid mismatch: {path} versus {reference}")
    values = np.rint(np.asarray(image.dataobj)).astype(np.int16)
    observed = set(int(value) for value in np.unique(values))
    if not observed.issubset(allowed | {0}):
        raise RuntimeError(f"Unexpected transformed atlas values: {sorted(observed - allowed - {0})}")
    return {
        "path": str(path),
        "reference": str(reference),
        "shape": list(image.shape),
        "voxel_sizes_mm": [float(value) for value in image.header.get_zooms()[:3]],
        "observed_labels": sorted(observed - {0}),
        "n_labeled_voxels": int(np.count_nonzero(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmriprep-root", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--atlas-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite atlas output: {args.output}")
    args.output.mkdir(parents=True)

    records = {
        branch: discover_branch(args.fmriprep_root, 4, branch) for branch in BRANCHES
    }
    references = {branch: item[0]["mask"] for branch, item in records.items()}
    source = nib.load(args.atlas)
    allowed = set()
    for line in args.atlas_labels.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if parts:
            try:
                value = int(float(parts[0]))
            except ValueError:
                continue
            if value > 0:
                allowed.add(value)

    provenance = {"interpolation": "nearest-neighbor / GenericLabel", "branches": {}}
    for branch in ("mni_res_native", "mni_res_2"):
        reference = nib.load(references[branch])
        transformed = resample_from_to(
            source, (reference.shape, reference.affine), order=0, mode="constant", cval=0
        )
        output = args.output / branch / "kastner_labels.nii.gz"
        output.parent.mkdir()
        data = np.rint(np.asarray(transformed.dataobj)).astype(np.int16)
        header = reference.header.copy()
        header.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(data, reference.affine, header), output)
        provenance["branches"][branch] = validate_label_map(
            output, references[branch], allowed
        )

    anat = args.fmriprep_root / "sub-04" / "anat"
    mni_to_t1w = single(anat, "*from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5")
    t1w_output = args.output / "subject_t1w" / "kastner_labels.nii.gz"
    t1w_output.parent.mkdir()
    ants_apply(args.atlas, references["subject_t1w"], t1w_output, [str(mni_to_t1w)])
    provenance["branches"]["subject_t1w"] = validate_label_map(
        t1w_output, references["subject_t1w"], allowed
    )
    provenance["branches"]["subject_t1w"]["transforms"] = [str(mni_to_t1w)]

    run1 = records["bold_acquired_grid"][0]["bold"]
    coreg = single(
        run1.parent,
        run1.name.split("_desc-preproc_bold", maxsplit=1)[0]
        + "_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt",
    )
    func_output = args.output / "bold_acquired_grid" / "kastner_labels.nii.gz"
    func_output.parent.mkdir()
    ants_apply(
        args.atlas,
        references["bold_acquired_grid"],
        func_output,
        [f"[{coreg},1]", str(mni_to_t1w)],
    )
    provenance["branches"]["bold_acquired_grid"] = validate_label_map(
        func_output, references["bold_acquired_grid"], allowed
    )
    provenance["branches"]["bold_acquired_grid"]["transforms"] = [
        f"inverse({coreg})", str(mni_to_t1w)
    ]
    provenance["branches"]["bold_acquired_grid"]["reference_run"] = 1
    provenance["branches"]["bold_acquired_grid"]["warning"] = (
        "A fixed run-01 atlas transform is used on the common acquired grid; "
        "no explicit between-run registration is introduced."
    )
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
