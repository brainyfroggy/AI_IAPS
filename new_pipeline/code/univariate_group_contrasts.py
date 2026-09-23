#!/usr/bin/env python3
"""Group-level univariate emotion contrasts from GLMsingle single-trial betas.

Stage A (per subject): average the 600 Type-D single-trial betas within each of
the 6 source x valence conditions, form 4 contrasts, apply post-GLM spatial
smoothing, and write one NIfTI per contrast.

Stage B (group):       one-sample t-test across subjects at each voxel, with
Benjamini-Hochberg FDR correction, restricted to voxels covered by every
subject's analysis mask.

Contrasts (all within source, so image-set/low-level differences between
Natural and AI stimuli cancel within each contrast):
    natural_pleasant_vs_neutral      AI_pleasant_vs_neutral
    natural_unpleasant_vs_neutral    AI_unpleasant_vs_neutral

Smoothing is applied HERE, post-GLM, not before the GLM. GLMsingle requires
unsmoothed input for its voxelwise HRF fitting and GLMdenoise steps, so the
betas on disk are unsmoothed by design; smoothing the condition-averaged
contrast is the correct place for it in this pipeline. Because Gaussian
convolution is linear and the mask-normalisation denominator does not depend on
trial, smoothing the contrast is mathematically identical to smoothing every
trial and then averaging -- so this ordering is a large compute saving, not an
approximation.

Stage A is resumable: a subject whose contrast files already exist is skipped
unless --overwrite is passed.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter

GLMSINGLE_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/glmsingle")

DEFAULT_SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

CONDITIONS = [(s, v) for s in ("natural", "ai") for v in ("pleasant", "neutral", "unpleasant")]

CONTRASTS = {
    "natural_pleasant_vs_neutral":   (("natural", "pleasant"),   ("natural", "neutral")),
    "natural_unpleasant_vs_neutral": (("natural", "unpleasant"), ("natural", "neutral")),
    "ai_pleasant_vs_neutral":        (("ai", "pleasant"),        ("ai", "neutral")),
    "ai_unpleasant_vs_neutral":      (("ai", "unpleasant"),      ("ai", "neutral")),
}

EXPECTED_TRIALS = 600
EXPECTED_PER_CONDITION = 100


# --------------------------------------------------------------------------- #
# Stage A: per-subject contrast maps
# --------------------------------------------------------------------------- #

def smooth_masked(values_at_flat, flat, mask, affine, fwhm_mm):
    """Mask-normalised Gaussian smoothing.

    Dividing by a smoothed copy of the mask corrects the edge dilution that
    plain convolution would introduce at the brain boundary (where the kernel
    otherwise averages in out-of-brain zeros). Identical implementation to the
    one validated in decode_roi_singletrial.py / condition_average_contrasts.py.
    """
    volume = np.zeros(mask.shape, dtype=np.float32)
    if fwhm_mm == 0:
        volume.ravel()[flat] = values_at_flat
        return volume
    sigma_vox = (fwhm_mm / math.sqrt(8 * math.log(2))) / nib.affines.voxel_sizes(affine)
    denom = gaussian_filter(mask.astype(np.float32), sigma=sigma_vox, mode="constant", cval=0.0)
    if np.any(denom[mask] <= np.finfo(np.float32).eps):
        raise RuntimeError("zero mask-normalised smoothing denominator")
    raw = np.zeros(mask.shape, dtype=np.float32)
    raw.ravel()[flat] = values_at_flat
    smoothed = gaussian_filter(raw, sigma=sigma_vox, mode="constant", cval=0.0)
    out = np.zeros(mask.shape, dtype=np.float32)
    out[mask] = smoothed[mask] / denom[mask]
    return out


def subject_contrasts(subject: int, fwhm: float, out_root: Path, overwrite: bool) -> dict:
    """Compute and write this subject's 4 smoothed contrast maps."""
    sub = f"sub-{subject:02d}"
    src = GLMSINGLE_ROOT / sub
    dst = out_root / "per_subject" / sub
    dst.mkdir(parents=True, exist_ok=True)

    targets = {c: dst / f"{sub}_contrast-{c}_fwhm-{fwhm:g}mm.nii.gz" for c in CONTRASTS}
    if not overwrite and all(p.exists() for p in targets.values()):
        return {"subject": subject, "status": "skipped_existing"}

    mask_img = nib.load(src / "analysis_mask.nii.gz")
    mask = np.asarray(mask_img.dataobj) > 0
    flat = np.load(src / "flat_mask_indices.npy").astype(np.int64)
    if not np.array_equal(flat, np.flatnonzero(mask.ravel())):
        raise RuntimeError(f"{sub}: flat_mask_indices inconsistent with analysis_mask")

    manifest = pd.read_csv(src / "trial_manifest.tsv", sep="\t").sort_values("beta_index")
    if manifest["beta_index"].astype(int).tolist() != list(range(EXPECTED_TRIALS)):
        raise RuntimeError(f"{sub}: trial_manifest beta_index is not exactly 0..599")
    manifest = manifest.reset_index(drop=True)
    manifest["source"] = manifest["source"].astype(str).str.lower()
    manifest["valence"] = manifest["valence"].astype(str).str.lower()

    cond_rows = {}
    for cond in CONDITIONS:
        idx = np.flatnonzero((manifest["source"] == cond[0]) & (manifest["valence"] == cond[1]))
        if idx.size != EXPECTED_PER_CONDITION:
            raise RuntimeError(f"{sub}: condition {cond} has {idx.size} trials, expected {EXPECTED_PER_CONDITION}")
        cond_rows[cond] = idx

    # Accumulate condition means blockwise so we never hold all 600 x n_voxels.
    n_vox = len(flat)
    cond_means = {c: np.empty(n_vox, dtype=np.float32) for c in CONDITIONS}
    with h5py.File(src / "glmsingle" / "TYPED_FITHRF_GLMDENOISE_RR.hdf5", "r") as h:
        ds = h["betasmd"]
        if ds.shape != (n_vox, 1, 1, EXPECTED_TRIALS):
            raise RuntimeError(f"{sub}: unexpected betasmd shape {ds.shape}")
        block = 8192
        for start in range(0, n_vox, block):
            stop = min(start + block, n_vox)
            chunk = np.asarray(ds[start:stop, 0, 0, :], dtype=np.float32)
            if not np.all(np.isfinite(chunk)):
                raise RuntimeError(f"{sub}: nonfinite betas in voxels {start}:{stop}")
            for cond, idx in cond_rows.items():
                cond_means[cond][start:stop] = chunk[:, idx].mean(axis=1)

    for name, (pos, neg) in CONTRASTS.items():
        diff = cond_means[pos] - cond_means[neg]
        vol = smooth_masked(diff, flat, mask, mask_img.affine, fwhm)
        img = nib.Nifti1Image(vol, mask_img.affine, header=mask_img.header.copy())
        img.header.set_data_dtype(np.float32)
        nib.save(img, targets[name])

    nib.save(nib.Nifti1Image(mask.astype(np.uint8), mask_img.affine),
             dst / f"{sub}_analysis-mask.nii.gz")
    return {"subject": subject, "status": "computed", "n_voxels": int(n_vox)}


# --------------------------------------------------------------------------- #
# Stage B: group one-sample t-test + FDR
# --------------------------------------------------------------------------- #

def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Return BH-adjusted p-values (q-values), same order as input."""
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    # enforce monotonicity from the largest p downward
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def group_stage(subjects, fwhm, out_root: Path, q_threshold: float,
                group_name: str = "group") -> dict:
    per_sub = out_root / "per_subject"
    grp = out_root / group_name
    grp.mkdir(parents=True, exist_ok=True)

    # Group mask = voxels present in EVERY subject's analysis mask.
    group_mask = None
    ref_img = None
    for s in subjects:
        sub = f"sub-{s:02d}"
        m_img = nib.load(per_sub / sub / f"{sub}_analysis-mask.nii.gz")
        m = np.asarray(m_img.dataobj) > 0
        if group_mask is None:
            group_mask, ref_img = m.copy(), m_img
        else:
            if m.shape != group_mask.shape or not np.allclose(m_img.affine, ref_img.affine, atol=1e-4):
                raise RuntimeError(f"{sub}: grid/affine mismatch against group reference")
            group_mask &= m
    n_group_vox = int(group_mask.sum())
    nib.save(nib.Nifti1Image(group_mask.astype(np.uint8), ref_img.affine),
             grp / "group_mask.nii.gz")

    summary = {
        "n_subjects": len(subjects),
        "subjects": subjects,
        "fwhm_mm": fwhm,
        "q_threshold": q_threshold,
        "group_mask_voxels": n_group_vox,
        "contrasts": {},
    }

    for name in CONTRASTS:
        stack = np.empty((len(subjects), n_group_vox), dtype=np.float32)
        for i, s in enumerate(subjects):
            sub = f"sub-{s:02d}"
            img = nib.load(per_sub / sub / f"{sub}_contrast-{name}_fwhm-{fwhm:g}mm.nii.gz")
            stack[i] = np.asarray(img.dataobj)[group_mask]
        if not np.all(np.isfinite(stack)):
            raise RuntimeError(f"{name}: nonfinite values in subject stack")

        t, p = stats.ttest_1samp(stack, popmean=0.0, axis=0)
        t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.nan_to_num(p, nan=1.0)
        q = benjamini_hochberg(p)
        sig = q < q_threshold

        def to_vol(vec, fill=0.0):
            v = np.full(group_mask.shape, fill, dtype=np.float32)
            v[group_mask] = vec
            return nib.Nifti1Image(v, ref_img.affine)

        nib.save(to_vol(stack.mean(axis=0)), grp / f"{name}_mean.nii.gz")
        nib.save(to_vol(t), grp / f"{name}_tstat.nii.gz")
        nib.save(to_vol(q, fill=1.0), grp / f"{name}_qval.nii.gz")
        nib.save(to_vol(np.where(sig, t, 0.0)), grp / f"{name}_tstat_fdr{q_threshold:g}.nii.gz")

        tsig = t[sig]
        summary["contrasts"][name] = {
            "n_significant_voxels": int(sig.sum()),
            "pct_of_group_mask": round(100.0 * sig.sum() / n_group_vox, 3),
            "n_positive": int((tsig > 0).sum()),
            "n_negative": int((tsig < 0).sum()),
            "max_t": float(t.max()),
            "min_t": float(t.min()),
            "min_q": float(q.min()),
        }
        print(f"{name}: {int(sig.sum())} sig voxels ({summary['contrasts'][name]['pct_of_group_mask']}%), "
              f"t range [{t.min():.2f}, {t.max():.2f}], min q = {q.min():.3g}", flush=True)

    (grp / "group_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fwhm", type=float, default=8.0)
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-subject-stage", action="store_true")
    ap.add_argument("--group-name", default="group",
                    help="Subdirectory under --out for this group result, so alternate "
                         "subject sets (e.g. a motion-based exclusion) can be written "
                         "side by side without clobbering each other.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"subjects (n={len(args.subjects)}): {args.subjects}", flush=True)
    print(f"smoothing FWHM: {args.fwhm} mm | FDR q < {args.q}", flush=True)

    if not args.skip_subject_stage:
        print("--- stage A: per-subject contrasts ---", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(subject_contrasts, s, args.fwhm, args.out, args.overwrite): s
                       for s in args.subjects}
            for fut in as_completed(futures):
                s = futures[fut]
                res = fut.result()  # re-raises with subject context preserved in the message
                print(f"  sub-{s:02d}: {res['status']}", flush=True)

    print(f"--- stage B: group statistics -> {args.group_name}/ ---", flush=True)
    group_stage(args.subjects, args.fwhm, args.out, args.q, args.group_name)
    print("UNIVARIATE_GROUP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
