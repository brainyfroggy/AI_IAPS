#!/usr/bin/env python3
"""Pilot: does AR(1) serial-correlation prewhitening (which SPM applies, GLMsingle
does not) improve single-trial beta estimation and ERP decoding accuracy?

This is a controlled ablation, NOT a reproduction of GLMsingle or SPM: both arms use
a plain OLS LSA fit with GLMsingle's OWN per-trial design matrices
(DESIGNINFO.npy's designSINGLE -- one canonical-HRF-convolved column per trial,
globally indexed 0..599, mostly zero outside that trial's run) and GLMsingle's own
confound regressors (via build_nuisance, imported fresh from the frozen
pilot_raw_pipeline_sub4_6/code/run_glmsingle.py so the exact same motion/spike
columns are used) plus a degree-3 polynomial per-run trend (matching GLMsingle's
maxpolydeg=[3]*10). Neither arm has GLMsingle's per-voxel HRF library, GLMdenoise,
or ridge regularization -- those are deliberately stripped out so the ONLY
difference between arm A and arm B is:

  Arm A (no whitening): plain OLS on [trial columns | poly | nuisance].
  Arm B (AR(1) whitened): estimate one pooled AR(1) coefficient per run from arm
    A's residuals (median per-voxel lag-1 autocorrelation), apply the standard
    Prais-Winsten whitening transform to both X and y, refit OLS on the whitened
    data. This approximates SPM's SPM.xVi.form='AR(1)+w' step (a full ReML
    estimate) with a simpler, honestly-labeled method-of-moments version -- it is
    NOT a literal reimplementation of SPM's algorithm.

Restricted to the Kastner ROI voxel union (not whole-brain) for tractable pilot
runtime, and to 0mm (no post-hoc smoothing) so the smoothing factor -- already
tested and shown not to matter -- doesn't confound this comparison.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import nibabel as nib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER, roi_indices  # noqa: E402
from glmsingle_typeb_erp_pilot import decode_one  # noqa: E402

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")
KASTNER = ROOT / ".." / "Decoding" / "full_cohort_raw_pipeline_28" / "resources" / "kastner"
PILOT_RUNNER = ROOT / ".." / "Decoding" / "pilot_raw_pipeline_sub4_6" / "code" / "run_glmsingle.py"


def _fresh_import(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def poly_basis(n_time: int, degree: int = 3) -> np.ndarray:
    t = np.linspace(-1, 1, n_time)
    cols = [np.ones(n_time)] + [t ** d for d in range(1, degree + 1)]
    basis = np.column_stack(cols)
    return basis / np.linalg.norm(basis, axis=0, keepdims=True)


def prais_winsten_whiten(X: np.ndarray, y: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    Xw = X.copy()
    yw = y.copy()
    Xw[1:] = X[1:] - rho * X[:-1]
    yw[1:] = y[1:] - rho * y[:-1]
    scale = np.sqrt(1 - rho ** 2)
    Xw[0] = X[0] * scale
    yw[0] = y[0] * scale
    return Xw, yw


def fit_run(X: np.ndarray, y: np.ndarray, n_trial_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS fit; returns (trial_betas (n_trial_cols, n_voxels), residuals (n_time, n_voxels))."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta[:n_trial_cols], resid


def estimate_ar1(resid: np.ndarray) -> float:
    num = np.sum(resid[1:] * resid[:-1], axis=0)
    den = np.sum(resid[:-1] ** 2, axis=0)
    valid = den > 1e-8
    per_voxel_rho = num[valid] / den[valid]
    rho = float(np.median(per_voxel_rho))
    return float(np.clip(rho, -0.9, 0.9))


def process_subject(subject: int, pilot: ModuleType, roi_positions: dict, union: np.ndarray) -> dict:
    input_root = ROOT / "glmsingle" / f"sub-{subject:02d}"
    provenance = json.loads((input_root / "provenance.json").read_text())
    bold_files = [Path(p) for p in provenance["bold_files"]]

    confound_files = []
    for bold in bold_files:
        name = bold.name.split("_space-")[0] + "_desc-confounds_timeseries.tsv"
        confound = bold.parent / name
        if not confound.exists():
            raise RuntimeError(f"missing confounds: {confound}")
        confound_files.append(confound)

    designinfo = np.load(input_root / "glmsingle" / "DESIGNINFO.npy", allow_pickle=True).item()
    design_single = designinfo["designSINGLE"]  # 10 x (n_time, 600)

    coords = np.column_stack(np.unravel_index(union, nib.load(bold_files[0]).shape[:3]))

    n_trials = 600
    betas_noAR = np.full((n_trials, len(union)), np.nan, dtype=np.float32)
    betas_AR1 = np.full((n_trials, len(union)), np.nan, dtype=np.float32)
    rhos = []

    for run_idx, (bold_path, confound_path) in enumerate(zip(bold_files, confound_files)):
        image = nib.load(bold_path)
        volume = image.get_fdata(dtype=np.float32, caching="unchanged")
        y = np.asarray(volume[coords[:, 0], coords[:, 1], coords[:, 2], :], dtype=np.float64).T  # (n_time, n_voxels)
        del volume

        n_time = y.shape[0]
        design_run = design_single[run_idx]
        trial_cols = np.flatnonzero(np.any(design_run != 0, axis=0))
        if len(trial_cols) == 0:
            raise RuntimeError(f"sub{subject} run{run_idx+1}: no active trial columns found")

        nuisance, _names, _qc = pilot.build_nuisance(confound_path, n_time)
        poly = poly_basis(n_time, degree=3)
        X = np.concatenate([design_run[:, trial_cols], poly, nuisance], axis=1)

        beta_trials, resid = fit_run(X, y, len(trial_cols))
        betas_noAR[trial_cols] = beta_trials.astype(np.float32)

        rho = estimate_ar1(resid)
        rhos.append(rho)
        Xw, yw = prais_winsten_whiten(X, y, rho)
        beta_trials_ar1, _ = fit_run(Xw, yw, len(trial_cols))
        betas_AR1[trial_cols] = beta_trials_ar1.astype(np.float32)

    manifest = pd.read_csv(input_root / "trial_manifest.tsv", sep="\t").sort_values("beta_index")
    return {
        "betas_noAR": betas_noAR, "betas_AR1": betas_AR1,
        "manifest": manifest, "rhos": rhos,
    }


def main() -> None:
    pilot = _fresh_import(PILOT_RUNNER, "ai_iaps_pilot_glmsingle_ar1")
    atlas = KASTNER / "kastner.nii.gz"
    labels = KASTNER / "kastner.nii.txt"

    pilot_subjects = [1, 2, 4]
    all_rows = []
    for s in pilot_subjects:
        input_root = ROOT / "glmsingle" / f"sub-{s:02d}"
        mask_image = nib.load(input_root / "analysis_mask.nii.gz")
        mask = np.asarray(mask_image.dataobj) > 0
        union, positions, _counts = roi_indices(atlas, labels, mask_image, mask)

        result = process_subject(s, pilot, positions, union)
        print(f"sub-{s:02d}: median AR(1) rho per run = {np.round(result['rhos'], 3).tolist()}", flush=True)

        for variant, betas in [("no_whitening", result["betas_noAR"]), ("ar1_whitened", result["betas_AR1"])]:
            df = decode_one(betas, positions, result["manifest"])
            df["subject"] = f"Sub{s:02d}"
            df["variant"] = variant
            all_rows.append(df)
            print(f"  sub-{s:02d} {variant}: mean accuracy {df['accuracy'].mean():.4f}", flush=True)

    result_df = pd.concat(all_rows, ignore_index=True)
    out = ROOT / "glmsingle_ar1_pilot_results.csv"
    result_df.to_csv(out, index=False)
    print(f"\nsaved {out}")

    pivot = result_df.pivot_table(index=["subject", "roi", "contrast"], columns="variant", values="accuracy")
    pivot["ar1_minus_noAR"] = pivot["ar1_whitened"] - pivot["no_whitening"]
    print(f"\nmean(AR1 - no_whitening) across all cells: {pivot['ar1_minus_noAR'].mean()*100:+.2f} pts")
    print(f"AR1 wins: {(pivot['ar1_minus_noAR'] > 0).sum()}/{len(pivot)} cells")
    print(f"no_whitening mean accuracy: {pivot['no_whitening'].mean()*100:.2f}%")
    print(f"ar1_whitened  mean accuracy: {pivot['ar1_whitened'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
