#!/usr/bin/env python3
"""Compare the three Subject-30/Subject-33 handling variants of the categorical-GLM group result.

Three cases, all sharing the same 27 unambiguous subjects (1,2,4,5,6,7,9,11-29,31 minus Sub30):
  A. group_sub30runs1-7  -- n=28, Sub30 patched to its clean runs 1-7, Sub33 not yet onboarded.
  B. group_n28_no_sub30  -- n=28, Sub30 excluded entirely, Sub33 included instead.
  C. group_n29           -- n=29, Sub30 patched to runs 1-7, AND Sub33 included.

Produces one comparison figure (grouped bar chart of significant-voxel counts and peak |t| per
contrast, one bar-triplet per case) plus a TSV of the same numbers, so the trade-off documented in
COMPLETION_REPORT.md sections 7.4/8.4/8.5 (more subjects -> more significant voxels but a noisier
per-voxel estimate when Sub30 is included even patched) is visible at a glance rather than only in
prose.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/categorical_glm")
OUT = ROOT / "group_case_comparison"

CASES = [
    ("group_sub30runs1-7", "A: n=28\nSub30→runs1-7\n(no Sub33)"),
    ("group_n28_no_sub30", "B: n=28\nSub30 excluded\n(+Sub33)"),
    ("group_n29",          "C: n=29\nSub30→runs1-7\n(+Sub33)"),
]
CONTRASTS = [
    ("natural_pleasant_vs_neutral",   "Natural\nPleasant−Neutral"),
    ("natural_unpleasant_vs_neutral", "Natural\nUnpleasant−Neutral"),
    ("ai_pleasant_vs_neutral",        "AI\nPleasant−Neutral"),
    ("ai_unpleasant_vs_neutral",      "AI\nUnpleasant−Neutral"),
]
CASE_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {name: json.loads((ROOT / name / "group_summary.json").read_text()) for name, _ in CASES}

    rows = []
    for case_name, _ in CASES:
        s = summaries[case_name]
        for contrast_key, _ in CONTRASTS:
            c = s["contrasts"][contrast_key]
            rows.append({
                "case": case_name,
                "n_subjects": s["n_subjects"],
                "contrast": contrast_key,
                "n_significant_voxels": c["n_significant_voxels"],
                "pct_of_group_mask": c["pct_of_group_mask"],
                "peak_t": c["max_t"],
                "min_q": c["min_q"],
            })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    n_contrasts = len(CONTRASTS)
    x = np.arange(n_contrasts)
    width = 0.26

    ax = axes[0]
    for i, (case_name, case_label) in enumerate(CASES):
        vals = [summaries[case_name]["contrasts"][k]["n_significant_voxels"] for k, _ in CONTRASTS]
        ax.bar(x + (i - 1) * width, vals, width, label=case_label.replace("\n", " "), color=CASE_COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in CONTRASTS], fontsize=8)
    ax.set_ylabel("significant voxels (FDR q<0.05)")
    ax.set_title("Significant-voxel count", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for i, (case_name, case_label) in enumerate(CASES):
        vals = [summaries[case_name]["contrasts"][k]["max_t"] for k, _ in CONTRASTS]
        ax.bar(x + (i - 1) * width, vals, width, label=case_label.replace("\n", " "), color=CASE_COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in CONTRASTS], fontsize=8)
    ax.set_ylabel("peak t")
    ax.set_title("Peak t-value", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle(
        "Categorical-GLM group result: three ways of handling Subject 30 / Subject 33",
        fontsize=12, fontweight="bold", y=1.10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    png = OUT / "group_case_comparison.png"
    pdf = OUT / "group_case_comparison.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    tsv = OUT / "group_case_comparison.tsv"
    cols = list(rows[0].keys())
    with tsv.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in cols) + "\n")

    print(f"saved {png}")
    print(f"saved {pdf}")
    print(f"saved {tsv}")
    print("COMPARE_COMPLETE")


if __name__ == "__main__":
    main()
