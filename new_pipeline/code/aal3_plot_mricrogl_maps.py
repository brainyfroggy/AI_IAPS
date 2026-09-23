#!/usr/bin/env python3
"""Build whole-brain AAL3 decoding-accuracy heatmaps on MRIcroGL's own bundled
AAL3.nii.gz grid, for viewing directly in MRIcroGL.

MRIcroGL's AAL3.nii.gz (C:\\MRIcroGL\\Resources\\atlas\\AAL3.nii.gz) uses a
different grid/affine than the atlas we resampled onto the GLMsingle native
beta grid for decoding, but its label ids/names match our roi_labels.csv
exactly (166 nonzero ids, same 4 missing placeholder ids: 35, 36, 81, 82).
Because the mapping is by label VALUE, not by spatial resampling, no
reprojection is needed or wanted here -- each output voxel is set to the
decoded region's group-mean accuracy directly on MRIcroGL's own grid, so the
files display correctly against MRIcroGL's bundled templates.

Background (unlabeled + dropped-region) voxels are left as NaN so MRIcroGL's
"hide near-zero" / colormap range works cleanly.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/aal3_crosssource")
MRICROGL_ATLAS = Path("/mnt/c/MRIcroGL/Resources/atlas/AAL3.nii.gz")

CROSS_CONTRASTS = (
    "train_natural_pleasant_vs_neutral_test_ai",
    "train_ai_pleasant_vs_neutral_test_natural",
    "train_natural_unpleasant_vs_neutral_test_ai",
    "train_ai_unpleasant_vs_neutral_test_natural",
)


def main() -> None:
    manifest = pd.read_csv(ROOT / "phase0_manifest" / "roi_manifest.csv")
    kept = manifest[manifest["kept"]]
    name_to_id = dict(zip(kept["roi_name"], kept["id"].astype(int)))

    group_df = pd.read_csv(ROOT / "smooth08" / "group" / "group_aal3_stats.csv")

    atlas_img = nib.load(MRICROGL_ATLAS)
    atlas_vals = np.rint(np.asarray(atlas_img.dataobj)).astype(np.int32)
    present_ids = set(int(v) for v in np.unique(atlas_vals) if v != 0)
    missing = sorted(set(name_to_id.values()) - present_ids)
    if missing:
        raise RuntimeError(f"MRIcroGL atlas is missing kept region ids: {missing}")

    out_dir = ROOT / "smooth08" / "group" / "accuracy_maps_mricrogl"
    out_dir.mkdir(parents=True, exist_ok=True)

    for contrast in CROSS_CONTRASTS:
        sub = group_df[group_df["contrast"] == contrast].set_index("roi")
        if len(sub) != len(kept):
            raise RuntimeError(f"{contrast}: expected {len(kept)} regions, got {len(sub)}")

        acc_map = np.full(atlas_vals.shape, np.nan, dtype=np.float32)
        for roi_name, row in sub.iterrows():
            rid = name_to_id[roi_name]
            acc_map[atlas_vals == rid] = row["mean_accuracy"]

        # Build a fresh header rather than copying the source atlas's: the
        # source sets intent_code='label' and cal_min/max=0/170 (the label-id
        # range) so viewers auto-apply a discrete atlas LUT instead of a
        # continuous colormap -- exactly the wrong behavior for a
        # per-region-averaged accuracy map, which needs an adjustable
        # continuous heatmap colormap.
        img_out = nib.Nifti1Image(acc_map, atlas_img.affine)
        img_out.header.set_data_dtype(np.float32)
        img_out.header.set_intent("none")
        img_out.header["cal_min"] = float(np.nanmin(acc_map))
        img_out.header["cal_max"] = float(np.nanmax(acc_map))
        img_out.header["scl_slope"] = 1.0
        img_out.header["scl_inter"] = 0.0
        img_out.header["descrip"] = b"AAL3 group-mean cross-source decoding accuracy"

        out_file = out_dir / f"aal3_mricrogl_accuracy_{contrast}.nii.gz"
        nib.save(img_out, out_file)
        n_painted = int(np.isfinite(acc_map).sum())
        print(f"saved {out_file}  (painted {n_painted} voxels, "
              f"range {np.nanmin(acc_map)*100:.1f}%-{np.nanmax(acc_map)*100:.1f}%)")

    print("AAL3_MRICROGL_MAPS_COMPLETE")


if __name__ == "__main__":
    main()
