from __future__ import annotations

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(r"N:\Experimental_Data\yujunchen\projects\AI_IAPS")
ROI_TABLE_DIR = PROJECT_ROOT / "ROI table"
BASE_SCRIPT = ROI_TABLE_DIR / "compute_roi_overlap_tables.py"
CONTRAST_DIR = PROJECT_ROOT / "GLM" / "outputs_seclvl" / "28subs"
OUT_DIR = ROI_TABLE_DIR / "28subs_fdr_positive_roi_overlap"

CONTRASTS = {
    "Pleasant_vs_Neutral": CONTRAST_DIR / "fdr_tstat_cond_Pl-Nt.nii",
    "AI_Pleasant_vs_AI_Neutral": CONTRAST_DIR / "fdr_tstat_cond_AI_Pl-AI_Nt.nii",
    "Unpleasant_vs_Neutral": CONTRAST_DIR / "fdr_tstat_cond_Up-Nt.nii",
    "AI_Unpleasant_vs_AI_Neutral": CONTRAST_DIR / "fdr_tstat_cond_AI_Up-AI_Nt.nii",
}

PAIRS = [
    (
        "Pleasant_vs_Neutral_AND_AI_Pleasant_vs_AI_Neutral",
        "Pleasant_vs_Neutral",
        "AI_Pleasant_vs_AI_Neutral",
    ),
    (
        "Unpleasant_vs_Neutral_AND_AI_Unpleasant_vs_AI_Neutral",
        "Unpleasant_vs_Neutral",
        "AI_Unpleasant_vs_AI_Neutral",
    ),
]


def parse_aal_labels(xml_path: Path) -> dict[int, str]:
    root = ET.parse(xml_path).getroot()
    labels: dict[int, str] = {}
    for label in root.findall(".//label"):
        index = label.findtext("index")
        name = label.findtext("name")
        if index and name:
            labels[int(index)] = name.strip()
    return labels


def load_aal_anatomy(ref_img):
    atlas_path = (
        ROI_TABLE_DIR
        / "atlases"
        / "aal_SPM12_manual"
        / "aal"
        / "atlas"
        / "AAL.nii"
    )
    labels_path = atlas_path.with_suffix(".xml")
    if not atlas_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "AAL atlas files are missing. Expected downloaded/extracted files at "
            f"{atlas_path.parent}"
        )
    atlas_img = nib.load(str(atlas_path))
    labels = parse_aal_labels(labels_path)
    resampled = image.resample_to_img(
        atlas_img,
        ref_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    return np.rint(resampled.get_fdata()).astype(np.int32), labels


def anatomical_lobe(label: str) -> str:
    lower = label.lower()
    if any(k in lower for k in ["calcarine", "cuneus", "lingual", "occipital"]):
        return "Occipital"
    if any(k in lower for k in ["fusiform", "temporal", "heschl"]):
        return "Temporal"
    if any(k in lower for k in ["frontal", "precentral", "supp_motor", "rolandic"]):
        return "Frontal/Motor"
    if any(k in lower for k in ["parietal", "postcentral", "precuneus", "angular", "supramarginal"]):
        return "Parietal"
    if any(k in lower for k in ["cingulum"]):
        return "Cingulate"
    if any(k in lower for k in ["hippocampus", "amygdala", "parahippocampal"]):
        return "Medial temporal"
    if any(k in lower for k in ["caudate", "putamen", "pallidum", "thalamus"]):
        return "Subcortical"
    if any(k in lower for k in ["cerebel", "vermis"]):
        return "Cerebellum"
    if "insula" in lower:
        return "Insula"
    return "Other"


def majority_anatomy(roi_mask: np.ndarray, aal_data: np.ndarray, aal_labels: dict[int, str]):
    values, counts = np.unique(aal_data[roi_mask], return_counts=True)
    valid = values > 0
    if not np.any(valid):
        return "", "", 0, 0.0
    values = values[valid]
    counts = counts[valid]
    best_idx = int(np.argmax(counts))
    aal_id = int(values[best_idx])
    count = int(counts[best_idx])
    label = aal_labels.get(aal_id, f"AAL_{aal_id}")
    return label, anatomical_lobe(label), count, 100.0 * count / int(roi_mask.sum())


def summarize_pair_with_anatomy(
    base,
    pair_name,
    map_a_name,
    map_b_name,
    atlas_name,
    atlas_img,
    labels,
    aal_data,
    aal_labels,
):
    img_a = nib.load(str(CONTRASTS[map_a_name]))
    img_b = nib.load(str(CONTRASTS[map_b_name]))
    data_a = img_a.get_fdata()
    data_b = img_b.get_fdata()
    atlas_data = np.rint(atlas_img.get_fdata()).astype(np.int32)

    overlap = (data_a > 0) & (data_b > 0) & (atlas_data > 0)
    pair_weights = (data_a + data_b) / 2.0
    voxel_volume = float(abs(np.linalg.det(img_a.affine[:3, :3])))

    rows = []
    for roi_id in sorted(int(v) for v in np.unique(atlas_data[overlap]) if v > 0):
        roi_mask = overlap & (atlas_data == roi_id)
        n_vox = int(roi_mask.sum())
        if n_vox == 0:
            continue

        indices = np.column_stack(np.where(roi_mask))
        weights = pair_weights[roi_mask]
        center = base.weighted_center_mni(indices, weights, img_a.affine)
        anatomy, lobe, anatomy_voxels, anatomy_percent = majority_anatomy(
            roi_mask, aal_data, aal_labels
        )

        atlas_roi_voxels = int((atlas_data == roi_id).sum())
        rows.append(
            {
                "comparison": pair_name,
                "atlas": atlas_name,
                "roi_id": roi_id,
                "roi_name": labels.get(roi_id, f"ROI_{roi_id}"),
                "anatomical_label_majority_AAL": anatomy,
                "anatomical_lobe_majority": lobe,
                "anatomical_label_voxels": anatomy_voxels,
                "anatomical_label_percent_of_overlap": anatomy_percent,
                "overlap_voxels": n_vox,
                "overlap_volume_mm3": n_vox * voxel_volume,
                "percent_of_atlas_roi": (
                    100.0 * n_vox / atlas_roi_voxels if atlas_roi_voxels else np.nan
                ),
                "center_x_mni": center[0],
                "center_y_mni": center[1],
                "center_z_mni": center[2],
                "mean_pair_value": float(np.mean(weights)),
                "max_pair_value": float(np.max(weights)),
                f"mean_{map_a_name}_value": float(np.mean(data_a[roi_mask])),
                f"mean_{map_b_name}_value": float(np.mean(data_b[roi_mask])),
            }
        )
    return rows


def load_base_module():
    spec = importlib.util.spec_from_file_location("roi_overlap_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    base = load_base_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base.PROJECT_ROOT = PROJECT_ROOT
    base.CONTRAST_DIR = CONTRAST_DIR
    base.OUT_DIR = OUT_DIR
    base.ATLAS_DIR = ROI_TABLE_DIR / "atlases"
    base.CONTRASTS = CONTRASTS
    base.PAIRS = PAIRS

    missing = [str(path) for path in CONTRASTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing contrast map(s):\n" + "\n".join(missing))

    ref_img = nib.load(str(CONTRASTS["Pleasant_vs_Neutral"]))
    atlases = [base.load_schaefer(ref_img), base.load_hcp(ref_img)]
    aal_data, aal_labels = load_aal_anatomy(ref_img)

    all_rows = []
    pair_summary_rows = []
    for pair_name, map_a_name, map_b_name in PAIRS:
        data_a = nib.load(str(CONTRASTS[map_a_name])).get_fdata()
        data_b = nib.load(str(CONTRASTS[map_b_name])).get_fdata()
        mask_a = data_a > 0
        mask_b = data_b > 0
        overlap = mask_a & mask_b
        pair_summary_rows.append(
            {
                "comparison": pair_name,
                "map_a": map_a_name,
                "map_b": map_b_name,
                "map_a_positive_voxels": int(mask_a.sum()),
                "map_b_positive_voxels": int(mask_b.sum()),
                "positive_overlap_voxels": int(overlap.sum()),
                "dice_coefficient_positive": float(
                    2 * overlap.sum() / (mask_a.sum() + mask_b.sum())
                ),
            }
        )

        for atlas_name, atlas_img, labels in atlases:
            rows = summarize_pair_with_anatomy(
                base,
                pair_name,
                map_a_name,
                map_b_name,
                atlas_name,
                atlas_img,
                labels,
                aal_data,
                aal_labels,
            )
            all_rows.extend(rows)
            short_name = "schaefer400" if atlas_name.startswith("Schaefer") else "hcp_mmp1"
            out_path = OUT_DIR / f"{pair_name}_{short_name}_roi_overlap.csv"
            base.write_csv(out_path, rows)
            print(f"Wrote {out_path} ({len(rows)} ROIs)")

    all_df = pd.DataFrame(all_rows)
    if not all_df.empty:
        all_df = all_df.sort_values(
            ["comparison", "atlas", "overlap_voxels", "mean_pair_value"],
            ascending=[True, True, False, False],
        )
        all_path = OUT_DIR / "all_roi_overlap_tables.csv"
        all_df.to_csv(all_path, index=False)
        print(f"Wrote {all_path} ({len(all_df)} rows)")

    summary_df = pd.DataFrame(pair_summary_rows)
    summary_path = OUT_DIR / "contrast_pair_positive_overlap_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    workbook_path = base.write_excel_workbook(summary_df, all_df)
    final_workbook_path = OUT_DIR / "roi_overlap_tables_28subs_fdr_positive.xlsx"
    if workbook_path != final_workbook_path:
        workbook_path.replace(final_workbook_path)
    print(f"Wrote {final_workbook_path}")


if __name__ == "__main__":
    main()
