#!/usr/bin/env python3
"""Test: is GLMsingle's estimation specifically compressing the natural-pleasant vs
neutral signal (relative to SPM), more than it compresses ai-pleasant vs neutral or
either unpleasant contrast?

Uses the RAW (pre-z-score) cached betas already built for the ERP permutation work
(stelzer_permutation/cache/ for GLMsingle 8mm Type-D, spm_erp_permutation/cache/ for
SPM LSS) -- no new beta estimation, just a mass-univariate readout of what's already
on disk. For each subject x ROI x {natural, ai}: compute Cohen's d for pleasant vs
neutral and unpleasant vs neutral, using per-trial ROI-mean beta as the scalar
(average over voxels, one value per trial, standard univariate approach). Compares
GLMsingle's d against SPM's d for the SAME subject x ROI x condition cell, since
absolute beta scale differs arbitrarily between pipelines (SPM's design-matrix units
vs GLMsingle's percent-BOLD-change) -- only the *within-pipeline standardized* effect
size is comparable across pipelines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GLM_CACHE = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/stelzer_permutation/cache")
SPM_CACHE = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/spm_erp_permutation/cache")
ROI_ORDER = ("V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "V3a", "V3b",
             "IPS", "LO1", "LO2", "hMT", "VO1", "VO2", "PHC1", "PHC2")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd < 1e-10:
        return np.nan
    return float((a.mean() - b.mean()) / pooled_sd)


def load_cache(path: Path) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    npz = np.load(path)
    data = npz["data"]
    source = npz["source"]
    valence = npz["valence"]
    positions = {k[len("positions_"):]: npz[k] for k in npz.files if k.startswith("positions_")}
    return data, positions, source, valence


def process_pipeline(cache_dir: Path, pipeline: str) -> pd.DataFrame:
    files = sorted(cache_dir.glob("sub-*.npz"))
    rows = []
    for f in files:
        subject = f.stem
        data, positions, source, valence = load_cache(f)
        for roi in ROI_ORDER:
            if roi not in positions:
                continue
            roi_data = data[:, positions[roi]]
            finite = np.all(np.isfinite(roi_data), axis=0)
            roi_data = roi_data[:, finite]
            if roi_data.shape[1] < 5:
                continue
            trial_mean = roi_data.mean(axis=1)  # (600,) -- one scalar per trial, avg over voxels

            for src in ("natural", "ai"):
                for pole in ("pleasant", "unpleasant"):
                    pos_idx = np.flatnonzero((source == src) & (valence == pole))
                    neu_idx = np.flatnonzero((source == src) & (valence == "neutral"))
                    d = cohens_d(trial_mean[pos_idx], trial_mean[neu_idx])
                    rows.append({
                        "pipeline": pipeline, "subject": subject, "roi": roi,
                        "source": src, "pole": pole, "cohens_d": d,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    glm_df = process_pipeline(GLM_CACHE, "glmsingle")
    spm_df = process_pipeline(SPM_CACHE, "spm")
    print(f"glmsingle rows: {len(glm_df)} ({glm_df['subject'].nunique()} subjects)")
    print(f"spm rows: {len(spm_df)} ({spm_df['subject'].nunique()} subjects)")

    # subject id formats already match (sub-01.npz style in both caches) -- confirm overlap
    common_subjects = sorted(set(glm_df["subject"]) & set(spm_df["subject"]))
    print(f"common subjects: {len(common_subjects)}")

    glm_df = glm_df[glm_df["subject"].isin(common_subjects)]
    spm_df = spm_df[spm_df["subject"].isin(common_subjects)]

    merged = glm_df.merge(spm_df, on=["subject", "roi", "source", "pole"], suffixes=("_glm", "_spm"))
    merged["abs_d_glm"] = merged["cohens_d_glm"].abs()
    merged["abs_d_spm"] = merged["cohens_d_spm"].abs()
    merged["ratio_glm_over_spm"] = merged["abs_d_glm"] / merged["abs_d_spm"]

    out = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/condition_compression_test.csv")
    merged.to_csv(out, index=False)
    print(f"\nsaved {out} ({len(merged)} rows)")

    summary = merged.groupby(["source", "pole"]).agg(
        mean_abs_d_glm=("abs_d_glm", "mean"),
        mean_abs_d_spm=("abs_d_spm", "mean"),
        median_ratio=("ratio_glm_over_spm", "median"),
        mean_ratio=("ratio_glm_over_spm", lambda x: np.nanmean(np.clip(x, 0, 5))),
    ).round(4)
    print("\n=== summary: |Cohen's d| by condition, GLMsingle vs SPM ===")
    print("(ratio < 1 means GLMsingle's effect size is SMALLER/more compressed than SPM's)")
    print(summary.to_string())

    print("\nCONDITION_COMPRESSION_TEST_COMPLETE")


if __name__ == "__main__":
    main()
