#!/usr/bin/env python3
"""Publication-ready figure for the group univariate emotion contrasts.

Renders the 2 (source: Natural / AI) x 2 (valence contrast: Pleasant-Neutral /
Unpleasant-Neutral) design as a 2x2 grid of axial slice montages, FDR-thresholded,
on the MNI152 template.

Design choices that matter for a publishable figure:
  * One symmetric colour scale shared by all four panels, so panel-to-panel
    differences are real differences and not an artefact of per-panel scaling.
    The scale is driven by a high percentile of the pooled suprathreshold data
    rather than the global max, so one extreme voxel cannot flatten everything.
  * A single shared colourbar rather than four, for the same reason.
  * Identical slice positions in every panel, so anatomy is directly comparable
    across the grid.
  * Vector PDF alongside 300-dpi PNG.

Emits the figure plus a small TSV of the numbers a caption/results paragraph
needs (suprathreshold voxel counts, peak t, peak MNI coordinates).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.image import coord_transform

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,   # embed as TrueType so text stays editable in Illustrator
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

# (row, col) -> (contrast key, panel label)
GRID = {
    (0, 0): ("natural_pleasant_vs_neutral",   "Pleasant − Neutral"),
    (0, 1): ("natural_unpleasant_vs_neutral", "Unpleasant − Neutral"),
    (1, 0): ("ai_pleasant_vs_neutral",        "Pleasant − Neutral"),
    (1, 1): ("ai_unpleasant_vs_neutral",      "Unpleasant − Neutral"),
}
ROW_LABELS = ["Natural", "AI-generated"]

DEFAULT_CUTS = [-20, -12, -4, 4, 12, 20]


def peak_info(t_img, thresholded: np.ndarray):
    """Peak |t| among suprathreshold voxels, with its MNI coordinate."""
    if not np.any(thresholded != 0):
        return None
    data = np.where(thresholded != 0, thresholded, 0.0)
    idx = np.unravel_index(np.argmax(np.abs(data)), data.shape)
    x, y, z = coord_transform(idx[0], idx[1], idx[2], t_img.affine)
    return {"t": float(data[idx]), "mni": [round(x, 1), round(y, 1), round(z, 1)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--cuts", type=int, nargs="+", default=DEFAULT_CUTS)
    ap.add_argument("--vmax", type=float, default=None,
                    help="Override the shared colour-scale maximum (default: 99th pct of pooled suprathreshold |t|)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.group_dir / "group_summary.json").read_text())
    n_sub = summary["n_subjects"]
    fwhm = summary.get("fwhm_mm")  # absent for the categorical-GLM group stage (smoothing is pre-GLM there)

    # Load all four thresholded maps first so the colour scale can be shared.
    imgs, arrays = {}, {}
    for key, _ in GRID.values():
        p = args.group_dir / f"{key}_tstat_fdr{args.q:g}.nii.gz"
        img = nib.load(p)
        imgs[key] = img
        arrays[key] = np.asarray(img.dataobj)

    pooled = np.concatenate([a[a != 0].ravel() for a in arrays.values()]) if \
        any((a != 0).any() for a in arrays.values()) else np.array([0.0])
    vmax = args.vmax if args.vmax is not None else float(np.percentile(np.abs(pooled), 99))
    vmax = max(vmax, 1e-3)

    # Height is matched to the panel aspect (6 axial slices wide, 2 rows) so the
    # brains fill the figure instead of floating in whitespace.
    fig = plt.figure(figsize=(11.5, 4.3))
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.90, top=0.80, bottom=0.02,
                          hspace=0.02, wspace=0.02)

    rows = []
    for (r, c), (key, panel_label) in GRID.items():
        ax = fig.add_subplot(gs[r, c])
        arr = arrays[key]
        has_sig = bool((arr != 0).any())

        disp = plotting.plot_stat_map(
            imgs[key],
            display_mode="z",
            cut_coords=args.cuts,
            threshold=1e-6,          # maps are already FDR-thresholded; hide exact zeros
            vmax=vmax,
            colorbar=False,          # one shared bar instead
            cmap="cold_hot",
            annotate=False,
            black_bg=False,
            axes=ax,
        )
        disp.annotate(size=7)

        if not has_sig:
            ax.text(0.5, 0.5, "no voxels survive FDR", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, style="italic", color="0.35")

        # Column headers on the top row only -- repeating them on row 2 is
        # redundant and eats vertical space.
        if r == 0:
            ax.set_title(panel_label, fontsize=11, pad=8)

        if c == 0:
            ax.text(-0.045, 0.5, ROW_LABELS[r], transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=11.5, fontweight="bold")

        st = summary["contrasts"][key]
        pk = peak_info(imgs[key], arr)
        rows.append({
            "contrast": key,
            "source": ROW_LABELS[r],
            "valence_contrast": panel_label,
            "n_sig_voxels": st["n_significant_voxels"],
            "pct_group_mask": st["pct_of_group_mask"],
            "n_positive": st["n_positive"],
            "n_negative": st["n_negative"],
            "peak_t": pk["t"] if pk else "",
            "peak_mni_x": pk["mni"][0] if pk else "",
            "peak_mni_y": pk["mni"][1] if pk else "",
            "peak_mni_z": pk["mni"][2] if pk else "",
            "min_q": st["min_q"],
        })

    # Shared colourbar
    cax = fig.add_axes([0.917, 0.16, 0.012, 0.50])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-vmax, vmax=vmax),
                               cmap=plotting.cm.cold_hot)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("t  (FDR q < %.2f)" % args.q, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.6)

    fig.suptitle(
        f"Emotional picture viewing: group univariate contrasts (n = {n_sub})",
        fontsize=13, fontweight="bold", y=0.985)
    smoothing_note = f"{fwhm:g} mm FWHM post-GLM smoothing" if fwhm is not None else "8 mm FWHM pre-GLM smoothing"
    fig.text(0.5, 0.917,
             f"MNI152NLin6Asym native resolution · {smoothing_note} · "
             f"one-sample t-test, FDR q < {args.q:g}",
             ha="center", fontsize=8.5, color="0.3")

    png = args.out / "univariate_group_contrasts.png"
    pdf = args.out / "univariate_group_contrasts.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    tsv = args.out / "univariate_group_contrast_stats.tsv"
    cols = list(rows[0].keys())
    with tsv.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in cols) + "\n")

    print(f"shared colour scale: +/-{vmax:.2f} t")
    for row in rows:
        print(f"{row['contrast']}: {row['n_sig_voxels']} sig voxels, peak t = {row['peak_t']}")
    print(f"saved {png}")
    print(f"saved {pdf}")
    print(f"saved {tsv}")
    print("PLOT_COMPLETE")


if __name__ == "__main__":
    main()
