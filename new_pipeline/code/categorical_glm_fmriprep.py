#!/usr/bin/env python3
"""Condition-level ('categorical') first-level GLM on fMRIPrep-preprocessed BOLD.

This is the direct nilearn/fMRIPrep counterpart of the historical
GLM_AI_IAPS_fmriprep.m (SPM12) design, generalized from its original 3 pilot
subjects to the full cohort and re-pointed at this project's current fMRIPrep
derivatives (MNI152NLin6Asym, native resolution) instead of the older
MNI152NLin2009cAsym res-2 space.

Purpose: isolate *GLM estimation method* (condition-level vs. single-trial
GLMsingle) from *preprocessing pipeline* (fMRIPrep vs. the historical SPM
pipeline) as separate variables. The GLMsingle single-trial betas
(new_pipeline/glmsingle/) stay untouched -- they are the correct input for
decoding. This script produces a second, independent set of contrast maps
for univariate comparison against both the GLMsingle-averaged maps and the
historical SPM maps.

Design, matched to GLM_AI_IAPS_fmriprep.m:
  * 6 condition regressors: {natural,ai} x {pleasant,neutral,unpleasant},
    3 s boxcar, canonical (SPM double-gamma) HRF.
  * 8 mm FWHM smoothing applied to the BOLD *before* the GLM (this is the
    standard, and the reason it cannot be the unsmoothed GLMsingle betas --
    GLMsingle requires unsmoothed input for its own voxelwise HRF/denoising
    steps; this script exists specifically to use the conventional order).
  * 128 s high-pass filter, AR(1) noise model, 6 rigid-body motion regressors
    (fMRIPrep's trans_x/y/z, rot_x/y/z -- the same 6 parameters the historical
    script took from SPM realignment, just from a different preprocessing
    tool).
  * All 10 runs modeled jointly per subject (separate drift/AR terms per run,
    shared condition betas), matching the historical multi-session design.

Output per subject: 4 contrast maps (nat_pl-nt, nat_up-nt, ai_pl-nt, ai_up-nt),
directly comparable in structure to univariate_group_contrasts.py's output, so
the same group_stage()-style t-test + FDR applies unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import smooth_img

FMRIPREP_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/LAB_IAPS_AI/fmriprep_derivatives/unified_glmsingle_28")
GLMSINGLE_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/glmsingle")  # source of trial_manifest.tsv onsets

TR = 1.8
STIM_DURATION = 3.0
FWHM_MM = 8.0
HIGH_PASS_HZ = 1.0 / 128.0
MOTION_COLUMNS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
N_RUNS = 10

CONTRASTS = {
    "natural_pleasant_vs_neutral":   "natural_pleasant - natural_neutral",
    "natural_unpleasant_vs_neutral": "natural_unpleasant - natural_neutral",
    "ai_pleasant_vs_neutral":        "ai_pleasant - ai_neutral",
    "ai_unpleasant_vs_neutral":      "ai_unpleasant - ai_neutral",
}

DEFAULT_SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]


def find_session_for_run(subject: int, run: int) -> tuple[int, str]:
    """Return (session, run-label-within-session) for a global run 1..10.

    Every subject in this cohort splits runs 1-6 into session 1 and 7-10 into
    session 2 EXCEPT the handful of genuinely irregular pilot/mixed-SDC
    subjects. Rather than hand-encode that here (risking exactly the kind of
    silent mismatch this project has repeatedly had to root-cause), the
    session is discovered directly from which session folder actually
    contains a file for this run -- fail-closed if it's in neither or both.
    """
    sub = f"sub-{subject:02d}"
    hits = []
    for ses in (1, 2):
        p = (FMRIPREP_ROOT / f"Sub{subject:02d}" / sub / f"ses-{ses:02d}" / "func" /
             f"{sub}_ses-{ses:02d}_task-iaps_run-{run:02d}_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz")
        if p.exists():
            hits.append(ses)
    if len(hits) != 1:
        raise RuntimeError(f"sub-{subject:02d} run-{run:02d}: found in {len(hits)} sessions, expected exactly 1")
    return hits[0]


def bold_path(subject: int, run: int, session: int) -> Path:
    sub = f"sub-{subject:02d}"
    return (FMRIPREP_ROOT / f"Sub{subject:02d}" / sub / f"ses-{session:02d}" / "func" /
            f"{sub}_ses-{session:02d}_task-iaps_run-{run:02d}_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz")


def mask_path(subject: int, run: int, session: int) -> Path:
    sub = f"sub-{subject:02d}"
    return (FMRIPREP_ROOT / f"Sub{subject:02d}" / sub / f"ses-{session:02d}" / "func" /
            f"{sub}_ses-{session:02d}_task-iaps_run-{run:02d}_space-MNI152NLin6Asym_desc-brain_mask.nii.gz")


def confounds_path(subject: int, run: int, session: int) -> Path:
    sub = f"sub-{subject:02d}"
    return (FMRIPREP_ROOT / f"Sub{subject:02d}" / sub / f"ses-{session:02d}" / "func" /
            f"{sub}_ses-{session:02d}_task-iaps_run-{run:02d}_desc-confounds_timeseries.tsv")


def condition_name(source: str, valence: str) -> str:
    return f"{source}_{valence}"


def build_events(manifest: pd.DataFrame, run: int) -> pd.DataFrame:
    rows = manifest[manifest["run"] == run].sort_values("onset")
    if rows.empty:
        raise RuntimeError(f"run {run}: no trials in manifest")
    return pd.DataFrame({
        "onset": rows["onset"].to_numpy(float),
        "duration": np.full(len(rows), STIM_DURATION),
        "trial_type": [condition_name(s, v) for s, v in zip(rows["source"], rows["valence"])],
    })


def subject_glm(subject: int, out_root: Path, overwrite: bool,
                runs: list[int] | None = None, suffix: str = "") -> dict:
    """Fit one subject's categorical GLM.

    `runs` restricts which of the subject's 10 runs enter the design (default:
    all 10). `suffix` is appended to every output filename so a run-restricted
    refit (e.g. dropping motion-corrupted runs for one subject) is written
    alongside the full-data result instead of overwriting it -- both stay on
    disk, so the choice of which to use for a given analysis is explicit and
    auditable rather than a silent overwrite.
    """
    sub = f"sub-{subject:02d}"
    runs = runs if runs is not None else list(range(1, N_RUNS + 1))
    dst = out_root / "per_subject" / sub
    dst.mkdir(parents=True, exist_ok=True)
    targets = {c: dst / f"{sub}_contrast-{c}_categorical{suffix}.nii.gz" for c in CONTRASTS}
    if not overwrite and all(p.exists() for p in targets.values()):
        return {"subject": subject, "status": "skipped_existing"}

    manifest = pd.read_csv(GLMSINGLE_ROOT / sub / "trial_manifest.tsv", sep="\t")
    manifest["source"] = manifest["source"].astype(str).str.lower()
    manifest["valence"] = manifest["valence"].astype(str).str.lower()

    run_imgs, run_events, run_confounds = [], [], []
    masks = []
    for run in runs:
        session = find_session_for_run(subject, run)
        bold = bold_path(subject, run, session)
        m = mask_path(subject, run, session)
        conf = confounds_path(subject, run, session)
        if not (bold.exists() and m.exists() and conf.exists()):
            raise RuntimeError(f"{sub} run-{run:02d}: missing bold/mask/confounds under ses-{session:02d}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # nilearn's own smoothing-precision notices
            smoothed = smooth_img(str(bold), fwhm=FWHM_MM)
        run_imgs.append(smoothed)
        masks.append(np.asarray(nib.load(m).dataobj) > 0)

        run_events.append(build_events(manifest, run))

        conf_df = pd.read_csv(conf, sep="\t")
        motion = conf_df[MOTION_COLUMNS].copy()
        if motion.isna().any().any():
            # fMRIPrep leaves the first-row derivative columns NaN; the raw
            # 6 rigid-body parameters used here should never be NaN. Fail
            # closed rather than silently zero-filling real data.
            raise RuntimeError(f"{sub} run-{run:02d}: NaN in raw motion columns {MOTION_COLUMNS}")
        run_confounds.append(motion.reset_index(drop=True))

    ref_shape = masks[0].shape
    if not all(m.shape == ref_shape for m in masks):
        raise RuntimeError(f"{sub}: run masks have inconsistent shapes")
    subject_mask = np.logical_and.reduce(masks)
    mask_img = nib.Nifti1Image(subject_mask.astype(np.uint8), nib.load(mask_path(subject, runs[0], find_session_for_run(subject, runs[0]))).affine)

    model = FirstLevelModel(
        t_r=TR,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=HIGH_PASS_HZ,
        noise_model="ar1",
        mask_img=mask_img,
        smoothing_fwhm=None,   # already smoothed above; avoid double smoothing
        minimize_memory=False,  # need in-memory design/labels for contrast computation
        n_jobs=1,
    )
    model = model.fit(run_imgs, events=run_events, confounds=run_confounds)

    for name, expr in CONTRASTS.items():
        z = model.compute_contrast(expr, output_type="effect_size")  # beta-scale contrast, not z-scored
        nib.save(z, targets[name])

    nib.save(mask_img, dst / f"{sub}_analysis-mask{suffix}.nii.gz")
    return {"subject": subject, "status": "computed", "n_mask_voxels": int(subject_mask.sum()), "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--runs", type=int, nargs="+", default=None,
                    help="Restrict to these run numbers (1-10), applied to every subject in "
                         "--subjects. Intended for single-subject reruns (e.g. dropping "
                         "motion-corrupted runs), not full-cohort use.")
    ap.add_argument("--suffix", default="",
                    help="Appended to output filenames so a run-restricted refit doesn't "
                         "overwrite the full-data result (e.g. --suffix _runs1-7).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"subjects (n={len(args.subjects)}): {args.subjects}", flush=True)
    print(f"HRF=spm  high_pass={HIGH_PASS_HZ:.6f}Hz(128s)  noise=ar1  smoothing={FWHM_MM}mm  motion=6 raw params", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(subject_glm, s, args.out, args.overwrite, args.runs, args.suffix): s
                   for s in args.subjects}
        for fut in as_completed(futures):
            s = futures[fut]
            res = fut.result()
            print(f"  sub-{s:02d}: {res['status']}", flush=True)

    print("CATEGORICAL_GLM_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
