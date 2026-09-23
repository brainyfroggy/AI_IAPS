#!/usr/bin/env python3
"""Decompose ROI decoding into GLOBAL AMPLITUDE vs SPATIAL PATTERN, for either pipeline.

Motivating puzzle: SPM LS-A betas give higher within-source natural-pleasant decoding
(uniformly across every ROI, including ones with no plausible valence tuning) and better
cross-source generalisation, yet worse within-source decoding overall (with whole
conditions falling significantly below chance). GLMsingle Type-D shows the opposite
profile.

Hypothesis: GLMsingle's GLMdenoise + ridge shrink globally-shared variance, whereas SPM
LS-A retains it. If the source-transferable valence signal is largely a global amplitude
effect rather than a fine spatial pattern, that single difference predicts all three
observations at once.

Four feature variants, identical folds/classifier, only the per-trial transform differs
(all transforms are within-trial, so none leak across the train/test split):
  raw         : voxel pattern as-is (what the main analyses used)
  global_only : ONE feature -- the mean across ROI voxels for that trial
  centered    : pattern minus its own trial mean (global removed, spatial kept)
  patternz    : pattern z-scored within trial (global mean AND scale removed)

Also reports Cohen's d of the per-trial ROI mean between conditions (a decoding-free
measure of how much raw global amplitude separates the conditions).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER, CONTRASTS, load_glmsingle, validate_manifest  # noqa: E402
import decode_roi_kebostyle_spm as spmmod  # noqa: E402

VARIANTS = ("raw", "global_only", "centered", "patternz")


def transform(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "raw":
        return x
    m = x.mean(axis=1, keepdims=True)
    if kind == "global_only":
        return m
    if kind == "centered":
        return x - m
    if kind == "patternz":
        s = x.std(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return (x - m) / s
    raise ValueError(kind)


def fit_acc(xtr, ytr, xte, yte) -> float:
    mean = xtr.mean(axis=0, dtype=np.float64)
    std = xtr.std(axis=0, ddof=0, dtype=np.float64)
    ok = np.isfinite(mean) & np.isfinite(std) & (std > 1e-9)
    if ok.sum() == 0:
        return np.nan
    a = ((xtr[:, ok] - mean[ok]) / std[ok]).astype(np.float32)
    b = ((xte[:, ok] - mean[ok]) / std[ok]).astype(np.float32)
    clf = SVC(kernel="linear", C=1.0, cache_size=512)
    clf.fit(a, ytr)
    return float(np.mean(clf.predict(b) == yte))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def load_glmsingle_pipeline(subject: str, atlas: Path, labels: Path):
    root = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/glmsingle")
    sub_dir = root / f"sub-{int(subject.replace('Sub','')):02d}"
    data, positions, _counts, manifest, _ref = load_glmsingle(sub_dir, atlas, labels, 8.0)
    manifest = validate_manifest(manifest)
    return data, positions, manifest


def load_spm_pipeline(subject: str, atlas: Path, labels: Path):
    beta_dir = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/GLM_singletrial/betas")
    label_file = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/GLM_singletrial/beta_groups.csv")
    onset_dir = Path("/mnt/n/Experimental_Data/yujunchen/projects/LAB_IAPS_AI/DataRecording")
    runs = spmmod.DEFAULT_RUNS
    beta_labels, stim_to_cat = spmmod.build_stimulus_lookup(label_file, onset_dir, runs)
    bl = spmmod.make_subject_beta_table(subject, beta_labels, stim_to_cat, onset_dir, runs)
    import nibabel as nib
    first = beta_dir / subject / str(bl["beta_file"].iloc[0])
    if not first.exists() and Path(str(first) + ".gz").exists():
        first = Path(str(first) + ".gz")
    ref = nib.load(str(first))
    positions, union_flat = spmmod.build_kastner_roi_columns(atlas, labels, ref)
    data = spmmod.load_subject_union_betas(subject, bl, beta_dir, union_flat)
    src, val = zip(*(spmmod.category_to_source_valence(c) for c in bl["category"]))
    manifest = pd.DataFrame({"source": src, "valence": val})
    return data, positions, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=["glmsingle", "spm"], required=True)
    ap.add_argument("--subject", required=True, help="Sub01 style for glmsingle, Sub1 style for spm")
    ap.add_argument("--atlas", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/full_cohort_raw_pipeline_28/resources/kastner/kastner.nii.gz"))
    ap.add_argument("--atlas-labels", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/full_cohort_raw_pipeline_28/resources/kastner/kastner.nii.txt"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.pipeline == "glmsingle":
        data, positions, manifest = load_glmsingle_pipeline(args.subject, args.atlas, args.atlas_labels)
    else:
        data, positions, manifest = load_spm_pipeline(args.subject, args.atlas, args.atlas_labels)

    src = manifest["source"].astype(str).str.lower().to_numpy()
    val = manifest["valence"].astype(str).str.lower().to_numpy()
    rng = np.random.default_rng(args.seed)

    rows = []
    for roi in ROI_ORDER:
        if roi not in positions:
            continue
        x_roi = data[:, positions[roi]].astype(np.float64)
        x_roi = x_roi[:, np.all(np.isfinite(x_roi), axis=0)]
        if x_roi.shape[1] < 10:
            continue
        feats = {v: transform(x_roi, v).astype(np.float32) for v in VARIANTS}
        gmean = x_roi.mean(axis=1)

        for name, kind, tsrc, ttest, pos in CONTRASTS:
            tr_idx = np.flatnonzero((src == tsrc) & np.isin(val, [pos, "neutral"]))
            tr_y = (val[tr_idx] == pos).astype(np.int8)
            if kind == "within":
                d_glob = cohens_d(gmean[tr_idx][tr_y == 1], gmean[tr_idx][tr_y == 0])
            else:
                te_idx = np.flatnonzero((src == ttest) & np.isin(val, [pos, "neutral"]))
                te_y = (val[te_idx] == pos).astype(np.int8)
                d_glob = cohens_d(gmean[te_idx][te_y == 1], gmean[te_idx][te_y == 0])

            for variant in VARIANTS:
                f = feats[variant]
                accs = []
                for rep in range(args.n_repeats):
                    if kind == "within":
                        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                              random_state=int(rng.integers(0, 2**31 - 1)))
                        for a, b in skf.split(tr_idx, tr_y):
                            accs.append(fit_acc(f[tr_idx[a]], tr_y[a], f[tr_idx[b]], tr_y[b]))
                    else:
                        s1 = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                             random_state=int(rng.integers(0, 2**31 - 1)))
                        s2 = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                             random_state=int(rng.integers(0, 2**31 - 1)))
                        trs = [a for a, _ in s1.split(tr_idx, tr_y)]
                        tes = [b for _, b in s2.split(te_idx, te_y)]
                        for a, b in zip(trs, tes):
                            accs.append(fit_acc(f[tr_idx[a]], tr_y[a], f[te_idx[b]], te_y[b]))
                rows.append({"pipeline": args.pipeline, "subject": args.subject, "roi": roi,
                             "contrast": name, "contrast_kind": kind, "variant": variant,
                             "n_voxels": int(x_roi.shape[1]),
                             "accuracy": float(np.nanmean(accs)), "d_global": d_glob})
        print(f"  {args.subject} {roi} done", flush=True)

    pd.DataFrame(rows).to_csv(args.output / f"{args.pipeline}_{args.subject}_globaldiag.csv", index=False)
    print(f"GLOBALDIAG_COMPLETE {args.pipeline} {args.subject}")


if __name__ == "__main__":
    main()
