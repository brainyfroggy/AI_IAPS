#!/usr/bin/env python3
"""Compose the MRIcroGL-rendered sagittal/coronal/axial panels into a publication figure,
matching the layout style of the user's reference figure (Natural/AI rows, Pleasant-Neutral /
Unpleasant-Neutral column groups, anatomical region labels), but using MRIcroGL's rendering
instead of nilearn, and labeling only regions actually confirmed significant in this cohort's
n=29 data via label_group_clusters.py's Harvard-Oxford atlas lookup (not copied from the
reference image, which was a different pipeline/dataset).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PANELS = Path("/mnt/c/Users/YUJUNC~1/AppData/Local/Temp/claude/C--/794a7e21-d3b3-4d39-be02-428ce42a9c00/scratchpad/panels4")
OUT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/categorical_glm/group_n29/figure_mricrogl")
CLUSTERS = json.loads(Path(
    "/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/categorical_glm/group_n29/cluster_labels/cluster_labels.json"
).read_text())

ROWS = ["natural", "ai"]
ROW_LABELS = ["Natural", "AI-generated"]
GROUPS = [
    ("pleasant", "Pleasant \u2212 Neutral"),
    ("unpleasant", "Unpleasant \u2212 Neutral"),
]
VIEWS = ["sagittal", "coronal", "axial"]


def crop_to_content(img: Image.Image, pad: int = 15) -> Image.Image:
    """Trim uniform black border down to the brain slice, with a small margin."""
    arr = np.asarray(img.convert("L"))
    rows = np.where(arr.max(axis=1) > 10)[0]
    cols = np.where(arr.max(axis=0) > 10)[0]
    if rows.size == 0 or cols.size == 0:
        return img
    top, bottom = max(rows[0] - pad, 0), min(rows[-1] + pad, arr.shape[0])
    left, right = max(cols[0] - pad, 0), min(cols[-1] + pad, arr.shape[1])
    return img.crop((left, top, right, bottom))


def region_summary(contrast_key: str, min_size: int = 15, max_items: int = 5) -> list[str]:
    clusters = CLUSTERS[contrast_key]
    seen = set()
    lines = []
    for c in clusters:
        if c["cluster_size_voxels"] < min_size:
            continue
        region = c["region"].split(" (")[0].replace("_", " ")
        if "Cerebral White Matter" in region or "Cerebral Cortex" in region or "no atlas label" in region or "Lateral Ventricle" in region:
            continue
        key = region.split(",")[0]
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{region} (t={c['peak_t']:+.1f}, {c['cluster_size_voxels']} vox)")
        if len(lines) >= max_items:
            break
    return lines


fig = plt.figure(figsize=(15, 8.5))
outer = fig.add_gridspec(1, 2, left=0.045, right=0.98, top=0.86, bottom=0.06, wspace=0.10)

for g, (group_key, group_label) in enumerate(GROUPS):
    inner = outer[g].subgridspec(2, 3, hspace=0.03, wspace=0.02)
    fig.text(0.27 + g * 0.50, 0.905, group_label, ha="center", fontsize=15, fontweight="bold")

    for r, row_key in enumerate(ROWS):
        contrast_key = f"{row_key}_{group_key}_vs_neutral"
        for v, view in enumerate(VIEWS):
            ax = fig.add_subplot(inner[r, v])
            img = Image.open(PANELS / f"{row_key}_{group_key}_{view}.png")
            img = crop_to_content(img)
            ax.imshow(np.asarray(img))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if v == 0:
                ax.text(-0.04, 0.5, ROW_LABELS[r], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=12, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="0.15", edgecolor="none"))

    regions_nat = region_summary(f"natural_{group_key}_vs_neutral")
    regions_ai = region_summary(f"ai_{group_key}_vs_neutral")
    caption = "Natural: " + "; ".join(regions_nat) + "\nAI-generated: " + "; ".join(regions_ai)
    fig.text(0.27 + g * 0.50, 0.075, caption, ha="center", va="top", fontsize=7.3, color="0.15",
              wrap=True)

cax = fig.add_axes([0.30, 0.955, 0.40, 0.018])
sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=3, vmax=11),
                            cmap=mpl.colors.LinearSegmentedColormap.from_list(
                                "hot4", ["#000000", "#7f0000", "#ff0000", "#ffff00", "#ffffff"]))
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("t (FDR q < 0.05)", fontsize=9)
cb.ax.tick_params(labelsize=8)

fig.suptitle("Emotional picture viewing: group univariate contrasts (n = 29)",
             fontsize=16, fontweight="bold", y=1.01)

OUT.mkdir(parents=True, exist_ok=True)
png = OUT / "univariate_group_contrasts_mricrogl.png"
pdf = OUT / "univariate_group_contrasts_mricrogl.pdf"
fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig(pdf, facecolor="white", bbox_inches="tight")
print(f"saved {png}")
print(f"saved {pdf}")
