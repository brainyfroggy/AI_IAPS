#!/usr/bin/env python3
"""Extract significant clusters from the n=29 categorical-GLM group maps and label each
cluster's peak with a real anatomical region, via MRIcroGL's bundled Harvard-Oxford
cortical + subcortical atlases (both registered to MNI152 / FSL-standard space, which is
the same template lineage as fMRIPrep's MNI152NLin6Asym output space used here).

This exists specifically so the publication figure's anatomical labels describe what is
actually significant in this cohort's data, rather than reusing labels from an unrelated
reference figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import coord_transform
from scipy import ndimage

import sys

# Group dir is overridable from the command line so this can label any group
# (n=29, n=30, ...) rather than only the one it was first written against.
GROUP_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/categorical_glm/group_n29")
ATLAS_ROOT = Path("/mnt/c/MRIcroGL/Resources/atlas")
OUT = GROUP_DIR / "cluster_labels"

ATLASES = {
    "cortical": (ATLAS_ROOT / "HarvardOxford.nii.gz", ATLAS_ROOT / "HarvardOxford.txt"),
    "subcortical": (ATLAS_ROOT / "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz",
                     ATLAS_ROOT / "HarvardOxford-sub-maxprob-thr25-1mm.txt"),
}

CONTRASTS = [
    "natural_pleasant_vs_neutral",
    "natural_unpleasant_vs_neutral",
    "ai_pleasant_vs_neutral",
    "ai_unpleasant_vs_neutral",
]

MIN_CLUSTER_VOXELS = 10


def load_atlas_labels(txt_path: Path) -> dict[int, str]:
    labels = {}
    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        idx_str, name = parts
        try:
            labels[int(idx_str)] = name.strip()
        except ValueError:
            continue
    return labels


def atlas_lookup(mni_xyz: tuple[float, float, float]) -> str:
    for kind, (nii_path, txt_path) in ATLASES.items():
        img = nib.load(nii_path)
        data = np.asarray(img.dataobj)
        labels = load_atlas_labels(txt_path)
        inv_affine = np.linalg.inv(img.affine)
        vox = inv_affine @ np.array([*mni_xyz, 1.0])
        i, j, k = (int(round(v)) for v in vox[:3])
        if not (0 <= i < data.shape[0] and 0 <= j < data.shape[1] and 0 <= k < data.shape[2]):
            continue
        idx = int(data[i, j, k])
        if idx != 0 and idx in labels:
            return f"{labels[idx]} ({kind})"
    return "no atlas label (likely white matter / CSF / outside atlas coverage)"


def clusters_for_contrast(name: str) -> list[dict]:
    img = nib.load(GROUP_DIR / f"{name}_tstat_fdr0.05.nii.gz")
    data = np.asarray(img.dataobj)
    sig = data != 0
    structure = np.ones((3, 3, 3))  # 26-connectivity, standard for cluster extraction
    labeled, n_clusters = ndimage.label(sig, structure=structure)

    clusters = []
    for cluster_id in range(1, n_clusters + 1):
        mask = labeled == cluster_id
        size = int(mask.sum())
        if size < MIN_CLUSTER_VOXELS:
            continue
        cluster_vals = np.where(mask, data, 0.0)
        peak_idx = np.unravel_index(np.argmax(np.abs(cluster_vals)), cluster_vals.shape)
        peak_t = float(data[peak_idx])
        x, y, z = coord_transform(peak_idx[0], peak_idx[1], peak_idx[2], img.affine)
        region = atlas_lookup((x, y, z))
        clusters.append({
            "contrast": name,
            "cluster_size_voxels": size,
            "peak_t": round(peak_t, 2),
            "peak_mni": [round(x, 1), round(y, 1), round(z, 1)],
            "region": region,
        })
    clusters.sort(key=lambda c: -abs(c["peak_t"]))
    return clusters


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_clusters = {}
    for contrast in CONTRASTS:
        cl = clusters_for_contrast(contrast)
        all_clusters[contrast] = cl
        print(f"\n=== {contrast}: {len(cl)} clusters (>= {MIN_CLUSTER_VOXELS} voxels) ===")
        for c in cl[:15]:
            print(f"  size={c['cluster_size_voxels']:5d} peak_t={c['peak_t']:+6.2f} "
                  f"mni={c['peak_mni']}  {c['region']}")

    (OUT / "cluster_labels.json").write_text(json.dumps(all_clusters, indent=2) + "\n")
    print(f"\nsaved {OUT / 'cluster_labels.json'}")


if __name__ == "__main__":
    main()
