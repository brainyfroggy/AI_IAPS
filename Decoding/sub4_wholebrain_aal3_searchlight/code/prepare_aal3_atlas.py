#!/usr/bin/env python3
"""Resample the AAL3 2mm atlas onto Subject 4's native-res beta grid.

AAL3v1.nii.gz and the GLMsingle mni_res_native betas are both in the
MNI152-family template space, just at different voxel grids (2mm vs native
~1.7966x1.7966x2.25mm) -- unlike the Kastner/Wang subject_t1w and
bold_acquired_grid branches in sub4_spatial_sensitivity_32, no ANTs/Docker
cross-space warp is needed here. This mirrors that project's mni_res_native
branch handling in prepare_kastner_atlases.py: a single nibabel
resample_from_to with nearest-neighbor (order=0) interpolation.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to


def load_roi_groups(labels_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(labels_csv)
    required = {"id", "roi_name", "group_id", "group_roi_name"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"roi_labels.csv missing columns: {frame.columns.tolist()}")
    return frame


def validate_label_map(path: Path, reference: Path, allowed: set[int]) -> dict:
    image = nib.load(path)
    target = nib.load(reference)
    if image.shape != target.shape[:3] or not np.allclose(image.affine, target.affine, atol=1e-5):
        raise RuntimeError(f"Atlas/reference grid mismatch: {path} versus {reference}")
    values = np.rint(np.asarray(image.dataobj)).astype(np.int32)
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
    parser.add_argument("--atlas", type=Path, required=True, help="AAL3v1.nii.gz (2mm)")
    parser.add_argument("--atlas-labels", type=Path, required=True, help="roi_labels.csv")
    parser.add_argument("--reference", type=Path, required=True, help="analysis_mask.nii.gz (native beta grid)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite atlas output: {args.output}")
    args.output.mkdir(parents=True)

    groups = load_roi_groups(args.atlas_labels)
    allowed = set(int(value) for value in groups["id"].tolist())
    n_groups = groups["group_id"].nunique()

    source = nib.load(args.atlas)
    reference = nib.load(args.reference)
    transformed = resample_from_to(
        source, (reference.shape[:3], reference.affine), order=0, mode="constant", cval=0
    )
    data = np.rint(np.asarray(transformed.dataobj)).astype(np.int32)
    header = reference.header.copy()
    header.set_data_dtype(np.int32)

    output = args.output / "aal3_native_labels.nii.gz"
    nib.save(nib.Nifti1Image(data, reference.affine, header), output)

    validation = validate_label_map(output, args.reference, allowed)
    validation["n_expected_groups"] = int(n_groups)
    validation["n_expected_raw_labels"] = int(len(allowed))
    observed_groups = groups.loc[groups["id"].isin(validation["observed_labels"]), "group_id"].nunique()
    validation["n_observed_groups_with_any_voxel"] = int(observed_groups)

    provenance = {
        "interpolation": "nearest-neighbor (order=0)",
        "source_atlas": str(args.atlas),
        "atlas_labels": str(args.atlas_labels),
        "reference": str(args.reference),
        "n_groups": int(n_groups),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "nibabel": nib.__version__,
        "pandas": pd.__version__,
        "validation": validation,
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"AAL3_ATLAS_PREPARED groups={n_groups} labeled_voxels={validation['n_labeled_voxels']}")


if __name__ == "__main__":
    main()
