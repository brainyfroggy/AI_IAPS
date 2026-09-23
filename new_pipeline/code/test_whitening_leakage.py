#!/usr/bin/env python3
"""Decisive test of whether the large whitening 'gain' in high-voxel ROIs (esp. IPS) is a
real SNR improvement or circularity from estimating the noise covariance on all trials.

decode_roi_random10fold_whitened.py estimates Sigma once per subject per ROI from
condition-demeaned residuals over ALL 600 trials, then applies W = Sigma^-1/2 to everything.
The condition means used for demeaning are computed from all trials INCLUDING the test
trials of every later fold, so W carries sample-specific information about the test data.
When n_voxels > n_trials the sample covariance is singular, Ledoit-Wolf shrinkage is tiny,
and inverting it amplifies near-null directions enormously -- exactly the directions the
condition-demeaning emptied out.

Three arms, identical folds and classifier, run per ROI:
  none      : no whitening (baseline)
  all_trials: W estimated once from all 600 trials      (what the batch did)
  train_only: W re-estimated inside each training fold  (leak-free)

If all_trials >> train_only ~= none, the gain is circular.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER, CONTRASTS, load_glmsingle, validate_manifest  # noqa: E402


def whitening_from(residuals: np.ndarray) -> np.ndarray:
    lw = LedoitWolf().fit(residuals)
    eigval, eigvec = np.linalg.eigh(lw.covariance_)
    eigval = np.clip(eigval, 1e-8, None)
    return ((eigvec * (eigval ** -0.5)) @ eigvec.T).astype(np.float32)


def residuals_of(x: np.ndarray, cond: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Condition-demeaned residuals using ONLY the trials in `rows`."""
    out = np.empty((len(rows), x.shape[1]), dtype=np.float64)
    sub_cond = cond[rows]
    for c in np.unique(sub_cond):
        m = sub_cond == c
        blk = x[rows[m]]
        out[m] = blk - blk.mean(axis=0, keepdims=True)
    return out


def fit_acc(xtr, ytr, xte, yte) -> float:
    mean = xtr.mean(axis=0, dtype=np.float64)
    std = xtr.std(axis=0, ddof=0, dtype=np.float64)
    ok = np.isfinite(mean) & np.isfinite(std) & (std > 1e-7)
    a = ((xtr[:, ok] - mean[ok]) / std[ok]).astype(np.float32)
    b = ((xte[:, ok] - mean[ok]) / std[ok]).astype(np.float32)
    clf = SVC(kernel="linear", C=1.0, cache_size=1024)
    clf.fit(a, ytr)
    return float(np.mean(clf.predict(b) == yte))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--atlas-labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rois", nargs="+", default=["IPS", "V1v", "hMT", "LO2"])
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data, positions, counts, manifest, _ = load_glmsingle(args.input_root, args.atlas, args.atlas_labels, 8.0)
    manifest = validate_manifest(manifest)
    src = manifest["source"].astype(str).str.lower().to_numpy()
    val = manifest["valence"].astype(str).str.lower().to_numpy()
    cond = np.array([f"{a}_{b}" for a, b in zip(src, val)])
    rng = np.random.default_rng(args.seed)

    rows = []
    within = [c for c in CONTRASTS if c[1] == "within"]
    for roi in args.rois:
        x = data[:, positions[roi]]
        x = x[:, np.all(np.isfinite(x), axis=0)].astype(np.float64)
        n_vox = x.shape[1]

        # arm B: W estimated once from all trials (what the batch did)
        w_all = whitening_from(residuals_of(x, cond, np.arange(len(x))))
        x_all = (x @ w_all).astype(np.float32)

        for name, _kind, tsrc, _tt, pos in within:
            idx = np.flatnonzero((src == tsrc) & np.isin(val, [pos, "neutral"]))
            y = (val[idx] == pos).astype(np.int8)
            acc = {"none": [], "all_trials": [], "train_only": []}
            for rep in range(args.n_repeats):
                skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                      random_state=int(rng.integers(0, 2**31 - 1)))
                for tr, te in skf.split(idx, y):
                    gtr, gte = idx[tr], idx[te]
                    acc["none"].append(fit_acc(x[gtr], y[tr], x[gte], y[te]))
                    acc["all_trials"].append(fit_acc(x_all[gtr], y[tr], x_all[gte], y[te]))
                    # leak-free: covariance from training trials of THIS fold only
                    w_tr = whitening_from(residuals_of(x, cond, gtr))
                    acc["train_only"].append(
                        fit_acc((x[gtr] @ w_tr).astype(np.float32), y[tr],
                                (x[gte] @ w_tr).astype(np.float32), y[te]))
            for arm, v in acc.items():
                rows.append({"subject": args.subject, "roi": roi, "n_voxels": n_vox,
                             "contrast": name, "arm": arm, "accuracy": float(np.mean(v))})
        print(f"  {args.subject} {roi} (n_vox={n_vox}) done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output / f"{args.subject}_leakage_test.csv", index=False)
    print(f"WHITENING_LEAKAGE_TEST_COMPLETE {args.subject}")


if __name__ == "__main__":
    main()
