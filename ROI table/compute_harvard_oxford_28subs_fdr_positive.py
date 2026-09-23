from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"N:\Experimental_Data\yujunchen\projects\AI_IAPS")
ROI_TABLE_DIR = PROJECT_ROOT / "ROI table"
LOCAL_PYTHON_DEPS = ROI_TABLE_DIR / ".python_deps"
if LOCAL_PYTHON_DEPS.exists():
    sys.path.insert(0, str(LOCAL_PYTHON_DEPS))

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets


GENERATOR_PATH = (
    ROI_TABLE_DIR / "compute_roi_tables_28subs_fdr_positive_merged_rois_with_masks.py"
)
OUT_DIR = ROI_TABLE_DIR / "28subs_fdr_positive_single_contrast_rois_merged_mricrogl"
ATLAS_DIR = OUT_DIR / "atlas"
HARVARD_DIR = ATLAS_DIR / "HarvardOxford"
SOURCE_CSV = (
    ROI_TABLE_DIR
    / "all_single_contrast_merged_positive_roi_tables_HCP_Schaefer_AAL3_KastnerWang.csv"
)
COMBINED_CSV = (
    ROI_TABLE_DIR
    / "all_single_contrast_merged_positive_roi_tables_HCP_Schaefer_AAL3_"
    "KastnerWang_HarvardOxford.csv"
)

# These broad structural masks duplicate the cortical atlas or are not gray-matter ROIs.
EXCLUDED_SUBCORTICAL_LABELS = {
    "Left Cerebral White Matter",
    "Right Cerebral White Matter",
    "Left Cerebral Cortex",
    "Right Cerebral Cortex",
    "Left Lateral Ventricle",
    "Right Lateral Ventricle",
}


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atlas_label_name(name: str) -> str:
    return "_".join(str(name).strip().split())


def build_combined_harvard_oxford() -> tuple[nib.Nifti1Image, dict[int, str]]:
    HARVARD_DIR.mkdir(parents=True, exist_ok=True)
    cortical = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm",
        data_dir=str(HARVARD_DIR),
        symmetric_split=True,
        verbose=0,
    )
    subcortical = datasets.fetch_atlas_harvard_oxford(
        "sub-maxprob-thr25-2mm",
        data_dir=str(HARVARD_DIR),
        symmetric_split=True,
        verbose=0,
    )

    cortical_img = cortical.maps
    subcortical_img = subcortical.maps
    if cortical_img.shape != subcortical_img.shape or not np.allclose(
        cortical_img.affine, subcortical_img.affine
    ):
        raise RuntimeError("Harvard-Oxford cortical and subcortical grids do not match")

    combined = np.rint(cortical_img.get_fdata()).astype(np.int16)
    sub_data = np.rint(subcortical_img.get_fdata()).astype(np.int16)
    labels = {
        roi_id: atlas_label_name(label)
        for roi_id, label in enumerate(cortical.labels)
        if roi_id > 0
    }

    next_id = max(labels) + 1
    for sub_id, label in enumerate(subcortical.labels):
        if sub_id == 0 or label in EXCLUDED_SUBCORTICAL_LABELS:
            continue
        mask = sub_data == sub_id
        if not np.any(mask):
            continue
        combined[mask] = next_id
        labels[next_id] = atlas_label_name(label)
        next_id += 1

    header = cortical_img.header.copy()
    header.set_data_dtype(np.int16)
    header.set_intent("label")
    atlas_img = nib.Nifti1Image(combined, cortical_img.affine, header)

    atlas_path = HARVARD_DIR / "HarvardOxford_cort_sub_maxprob_thr25_2mm_split.nii.gz"
    nib.save(atlas_img, str(atlas_path))
    label_lines = [f"{roi_id} {labels[roi_id]} {roi_id}\n" for roi_id in sorted(labels)]
    for suffix in (".nii.txt", ".txt"):
        label_path = HARVARD_DIR / f"HarvardOxford_cort_sub_maxprob_thr25_2mm_split{suffix}"
        label_path.write_text("".join(label_lines), encoding="utf-8")

    return atlas_img, labels


def main() -> None:
    generator = load_module(GENERATOR_PATH, "roi_generator")
    base = generator.load_base_module()
    generator.OUT_DIR.mkdir(parents=True, exist_ok=True)
    generator.ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    base.ATLAS_DIR = generator.ATLAS_DIR

    atlas_img, labels = build_combined_harvard_oxford()
    ref_img = nib.load(str(generator.CONTRASTS["Pleasant_vs_Neutral"]))
    resampled = base.image.resample_to_img(
        atlas_img,
        ref_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    atlas_name = "HarvardOxford_cort_sub_maxprob_thr25_2mm_split"
    generator.save_mricrogl_atlas(atlas_name, resampled, labels)

    all_rows: list[dict] = []
    for contrast_name, contrast_path in generator.CONTRASTS.items():
        rows = generator.summarize_contrast_and_save_masks(
            base,
            contrast_name,
            contrast_path,
            atlas_name,
            resampled,
            labels,
        )
        all_rows.extend(rows)
        contrast_csv = OUT_DIR / f"{contrast_name}_HarvardOxford_merged_positive_rois.csv"
        pd.DataFrame(rows).to_csv(contrast_csv, index=False)
        print(f"Wrote {contrast_csv} ({len(rows)} ROIs)")

    harvard_df = pd.DataFrame(all_rows)
    harvard_df = harvard_df.sort_values(
        ["contrast", "positive_voxels_in_merged_roi", "mean_positive_t"],
        ascending=[True, False, False],
    )
    harvard_csv = OUT_DIR / "all_HarvardOxford_single_contrast_merged_positive_rois.csv"
    harvard_df.to_csv(harvard_csv, index=False)

    source_df = pd.read_csv(SOURCE_CSV)
    combined_df = pd.concat([source_df, harvard_df], ignore_index=True)
    combined_df = combined_df.sort_values(
        ["contrast", "atlas", "positive_voxels_in_merged_roi", "mean_positive_t"],
        ascending=[True, True, False, False],
    )
    combined_df.to_csv(COMBINED_CSV, index=False)
    (OUT_DIR / COMBINED_CSV.name).write_bytes(COMBINED_CSV.read_bytes())

    print(f"Wrote {harvard_csv} ({len(harvard_df)} rows)")
    print(f"Wrote {COMBINED_CSV} ({len(combined_df)} rows)")


if __name__ == "__main__":
    main()
