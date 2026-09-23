#!/usr/bin/env python3
"""Phase 0 of the whole-brain AAL3 cross-source decoding plan (see
docs/AAL3_WHOLEBRAIN_CROSSSOURCE_PLAN.md).

Resamples AAL3v1.nii.gz (2mm) onto the new_pipeline native beta grid
(nearest-neighbor, matching the validated Sub4 prepare_aal3_atlas.py approach),
then freezes the ROI set: intersects with ALL 30 subjects' analysis masks and
keeps only regions with >=10 voxels in every subject. Regions are kept
L/R-separate (170 raw AAL3 ids), not bilateral-merged, per this task's scope.

All 30 subjects share an identical beta grid/affine (verified), so one
resample serves the whole cohort.
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

MIN_VOXELS = 10

DEFAULT_SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31, 33, 34]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", type=Path, required=True, help="AAL3v1.nii.gz (2mm)")
    ap.add_argument("--labels-csv", type=Path, required=True, help="roi_labels.csv (id, roi_name, ...)")
    ap.add_argument("--glmsingle-root", type=Path, required=True,
                    help="dir containing sub-XX/analysis_mask.nii.gz for all subjects")
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-voxels", type=int, default=MIN_VOXELS)
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    labels = pd.read_csv(args.labels_csv)
    required = {"id", "roi_name"}
    if not required.issubset(labels.columns):
        raise RuntimeError(f"labels csv missing columns: {labels.columns.tolist()}")
    id_to_name = dict(zip(labels["id"].astype(int), labels["roi_name"]))
    ids = sorted(id_to_name)
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate ids in labels csv")

    sub_dirs = [(s, args.glmsingle_root / f"sub-{s:02d}") for s in args.subjects]
    missing = [d for s, d in sub_dirs if not (d / "analysis_mask.nii.gz").exists()]
    if missing:
        raise RuntimeError(f"missing analysis_mask.nii.gz for: {missing}")

    reference = nib.load(sub_dirs[0][1] / "analysis_mask.nii.gz")
    for s, d in sub_dirs[1:]:
        img = nib.load(d / "analysis_mask.nii.gz")
        if img.shape[:3] != reference.shape[:3] or not np.allclose(img.affine, reference.affine, atol=1e-5):
            raise RuntimeError(f"sub-{s:02d}: grid/affine mismatch vs sub-{sub_dirs[0][0]:02d}")

    source = nib.load(args.atlas)
    resampled = resample_from_to(
        source, (reference.shape[:3], reference.affine), order=0, mode="constant", cval=0
    )
    atlas_vals = np.rint(np.asarray(resampled.dataobj)).astype(np.int32)
    header = reference.header.copy()
    header.set_data_dtype(np.int32)
    atlas_out = args.output / "aal3_native_labels.nii.gz"
    nib.save(nib.Nifti1Image(atlas_vals, reference.affine, header), atlas_out)

    n_subjects = len(sub_dirs)
    counts = np.zeros((n_subjects, len(ids)), dtype=np.int64)
    for si, (s, d) in enumerate(sub_dirs):
        mask = np.asarray(nib.load(d / "analysis_mask.nii.gz").dataobj) > 0
        masked_vals = np.where(mask, atlas_vals, 0)
        bincount = np.bincount(masked_vals.ravel(), minlength=int(atlas_vals.max()) + 1)
        for j, rid in enumerate(ids):
            counts[si, j] = bincount[rid] if rid < len(bincount) else 0

    min_across = counts.min(axis=0)
    rows = []
    for j, rid in enumerate(ids):
        kept = bool(min_across[j] >= args.min_voxels)
        rows.append({
            "id": int(rid),
            "roi_name": id_to_name[rid],
            "min_voxels_across_subjects": int(min_across[j]),
            "median_voxels_across_subjects": int(np.median(counts[:, j])),
            "max_voxels_across_subjects": int(counts[:, j].max()),
            "kept": kept,
            "reason": "" if kept else f"< {args.min_voxels} voxels in at least one subject",
        })
    manifest = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    manifest.to_csv(args.output / "roi_manifest.csv", index=False)

    counts_df = pd.DataFrame(counts, columns=[id_to_name[i] for i in ids])
    counts_df.insert(0, "subject", [f"Sub{s:02d}" for s, _ in sub_dirs])
    counts_df.to_csv(args.output / "per_subject_voxel_counts.csv", index=False)

    n_kept = int(manifest["kept"].sum())
    n_dropped = len(manifest) - n_kept
    provenance = {
        "atlas_source": str(args.atlas),
        "labels_csv": str(args.labels_csv),
        "n_raw_ids": len(ids),
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "dropped_regions": manifest.loc[~manifest["kept"], "roi_name"].tolist(),
        "min_voxels_threshold": args.min_voxels,
        "n_subjects": n_subjects,
        "subjects": [s for s, _ in sub_dirs],
        "reference_shape": list(reference.shape[:3]),
        "reference_affine": np.asarray(reference.affine).tolist(),
        "interpolation": "nearest-neighbor (order=0)",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "nibabel": nib.__version__,
        "pandas": pd.__version__,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"AAL3_MANIFEST_COMPLETE n_raw={len(ids)} kept={n_kept} dropped={n_dropped}")
    print(f"dropped: {provenance['dropped_regions']}")


if __name__ == "__main__":
    main()
