#!/usr/bin/env python3
"""Group-level stage for the condition-level ('categorical') first-level GLM.

Same statistics as univariate_group_contrasts.py's group_stage() (one-sample
t-test, Benjamini-Hochberg FDR, group mask = intersection of per-subject
masks), just re-pointed at categorical_glm_fmriprep.py's per-subject output
directory and file-naming convention, so the two estimation methods
(GLMsingle-averaged vs. condition-level GLM) can be compared on identical
group-level machinery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats

CONTRASTS = [
    "natural_pleasant_vs_neutral",
    "natural_unpleasant_vs_neutral",
    "ai_pleasant_vs_neutral",
    "ai_unpleasant_vs_neutral",
]

DEFAULT_SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glm-root", type=Path, required=True,
                    help="categorical_glm root containing per_subject/")
    ap.add_argument("--group-name", default="group")
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--subject-suffix", action="append", default=[],
                    metavar="SUBJECT:SUFFIX",
                    help="Override the per-subject file suffix for one subject, e.g. "
                         "30:_runs1-7 to read sub-30's run-restricted refit instead of "
                         "its full-10-run output. Repeatable.")
    args = ap.parse_args()

    suffix_map = {}
    for item in args.subject_suffix:
        subj_str, suf = item.split(":", 1)
        suffix_map[int(subj_str)] = suf

    per_sub = args.glm_root / "per_subject"
    grp = args.glm_root / args.group_name
    grp.mkdir(parents=True, exist_ok=True)

    group_mask = None
    ref_img = None
    for s in args.subjects:
        sub = f"sub-{s:02d}"
        suffix = suffix_map.get(s, "")
        m_img = nib.load(per_sub / sub / f"{sub}_analysis-mask{suffix}.nii.gz")
        m = np.asarray(m_img.dataobj) > 0
        if group_mask is None:
            group_mask, ref_img = m.copy(), m_img
        else:
            if m.shape != group_mask.shape or not np.allclose(m_img.affine, ref_img.affine, atol=1e-4):
                raise RuntimeError(f"{sub}: grid/affine mismatch against group reference")
            group_mask &= m
    n_group_vox = int(group_mask.sum())
    nib.save(nib.Nifti1Image(group_mask.astype(np.uint8), ref_img.affine), grp / "group_mask.nii.gz")

    summary = {"n_subjects": len(args.subjects), "subjects": args.subjects,
               "subject_suffix_overrides": {str(k): v for k, v in suffix_map.items()},
               "q_threshold": args.q, "group_mask_voxels": n_group_vox, "contrasts": {}}

    for name in CONTRASTS:
        stack = np.empty((len(args.subjects), n_group_vox), dtype=np.float32)
        for i, s in enumerate(args.subjects):
            sub = f"sub-{s:02d}"
            suffix = suffix_map.get(s, "")
            img = nib.load(per_sub / sub / f"{sub}_contrast-{name}_categorical{suffix}.nii.gz")
            stack[i] = np.asarray(img.dataobj)[group_mask]
        if not np.all(np.isfinite(stack)):
            raise RuntimeError(f"{name}: nonfinite values in subject stack")

        t, p = stats.ttest_1samp(stack, popmean=0.0, axis=0)
        t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.nan_to_num(p, nan=1.0)
        q = benjamini_hochberg(p)
        sig = q < args.q

        def to_vol(vec, fill=0.0):
            v = np.full(group_mask.shape, fill, dtype=np.float32)
            v[group_mask] = vec
            return nib.Nifti1Image(v, ref_img.affine)

        nib.save(to_vol(stack.mean(axis=0)), grp / f"{name}_mean.nii.gz")
        nib.save(to_vol(t), grp / f"{name}_tstat.nii.gz")
        nib.save(to_vol(q, fill=1.0), grp / f"{name}_qval.nii.gz")
        nib.save(to_vol(np.where(sig, t, 0.0)), grp / f"{name}_tstat_fdr{args.q:g}.nii.gz")

        tsig = t[sig]
        summary["contrasts"][name] = {
            "n_significant_voxels": int(sig.sum()),
            "pct_of_group_mask": round(100.0 * sig.sum() / n_group_vox, 3),
            "n_positive": int((tsig > 0).sum()),
            "n_negative": int((tsig < 0).sum()),
            "max_t": float(t.max()), "min_t": float(t.min()), "min_q": float(q.min()),
        }
        print(f"{name}: {int(sig.sum())} sig voxels ({summary['contrasts'][name]['pct_of_group_mask']}%), "
              f"t range [{t.min():.2f}, {t.max():.2f}], min q = {q.min():.3g}", flush=True)

    (grp / "group_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("CATEGORICAL_GROUP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
