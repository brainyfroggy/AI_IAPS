#!/usr/bin/env python3
"""Random-10-fold decoding (Bo et al. 2021 scheme, 30 repeats) with multivariate noise
normalization ("whitening") applied per ROI before classification.

Design (documented explicitly since it involves real choices):

  - Noise covariance is estimated from CONDITION-DEMEANED single-trial residuals:
    for each of the 6 conditions (source x valence), subtract that condition's mean
    pattern from every trial of that condition, pool the 600 resulting residual
    vectors, and estimate their voxel x voxel covariance.
  - GLMsingle Type-D does not persist full residual time series (would be enormous),
    so this condition-demeaned-trial approach is the standard practical substitute
    used when GLM residuals aren't directly available (see e.g. Walther et al. 2016,
    NeuroImage, on multivariate noise normalization / "multivariate noise
    normalisation using the residuals").
  - Estimated with Ledoit-Wolf shrinkage (sklearn.covariance.LedoitWolf), which is
    necessary, not optional: several ROIs (e.g. IPS, ~1000 voxels) have more voxels
    than the 600 available trials, so the raw sample covariance is singular.
  - Computed ONCE per subject per ROI, from all 600 trials, OUTSIDE the classification
    CV loop -- not re-estimated per fold/repetition. This mirrors how the Kastner
    atlas and 8mm smoothing are also fixed, task-independent preprocessing steps
    applied once before any train/test split. The alternative (re-estimating the
    covariance inside every one of the ~30,000 fold iterations per subject) is
    both computationally prohibitive (repeated O(n_voxels^3) eigendecompositions)
    and estimates a quantity -- voxel noise correlation structure -- that is a
    property of the measurement, not of which specific trials land in a given
    fold, so this is not deprived data leakage for classification purposes (no
    valence/source labels are used at any point in this step, only the 6 condition
    groupings needed to demean).
  - The resulting whitening matrix W (= Sigma^-1/2) is applied once to the full
    600-trial ROI pattern; the classification loop below (fold construction, per-
    voxel z-scoring on training folds only, SVM) is otherwise byte-for-byte the
    same procedure as decode_roi_random10fold.py.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from sklearn import __version__ as sklearn_version
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import (  # noqa: E402
    ROI_ORDER,
    CONTRASTS,
    load_glmsingle,
    validate_manifest,
)

N_REPEATS = 30
N_FOLDS = 10


def whitening_transform(roi_data: np.ndarray, condition: np.ndarray) -> tuple[np.ndarray, dict]:
    """Estimate Sigma^-1/2 from condition-demeaned trial residuals (Ledoit-Wolf shrinkage)."""
    residuals = np.empty_like(roi_data, dtype=np.float64)
    for cond in np.unique(condition):
        idx = condition == cond
        residuals[idx] = roi_data[idx] - roi_data[idx].mean(axis=0, keepdims=True)
    lw = LedoitWolf().fit(residuals)
    cov = lw.covariance_
    eigval, eigvec = np.linalg.eigh(cov)
    eigval_clipped = np.clip(eigval, 1e-8, None)
    w = (eigvec * (eigval_clipped ** -0.5)) @ eigvec.T
    diag = {
        "shrinkage": float(lw.shrinkage_),
        "min_eigval_raw": float(eigval.min()),
        "max_eigval_raw": float(eigval.max()),
        "n_negative_or_tiny_eigval_clipped": int((eigval < 1e-8).sum()),
    }
    return w.astype(np.float32), diag


def fit_predict_accuracy(x_train, y_train, x_test, y_test) -> float:
    mean = x_train.mean(axis=0, dtype=np.float64)
    std = x_train.std(axis=0, ddof=0, dtype=np.float64)
    valid = np.isfinite(mean) & np.isfinite(std) & (std > 1e-7)
    if int(valid.sum()) < 10:
        raise RuntimeError("too few train-variable features")
    xtr = ((x_train[:, valid] - mean[valid]) / std[valid]).astype(np.float32)
    xte = ((x_test[:, valid] - mean[valid]) / std[valid]).astype(np.float32)
    clf = SVC(kernel="linear", C=1.0, cache_size=2048)
    clf.fit(xtr, y_train)
    pred = clf.predict(xte)
    return float(np.mean(pred == y_test))


def decode_random_kfold_whitened(
    data: np.ndarray,
    positions: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    common: dict,
    n_repeats: int = N_REPEATS,
    n_folds: int = N_FOLDS,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_values = manifest["source"].astype(str).str.lower().to_numpy()
    valence_values = manifest["valence"].astype(str).str.lower().to_numpy()
    condition_values = np.char.add(np.char.add(source_values, "_"), valence_values)
    rng = np.random.default_rng(seed)

    subject_rows: list[dict] = []
    repetition_rows: list[dict] = []
    whitening_rows: list[dict] = []

    for roi in ROI_ORDER:
        roi_data = data[:, positions[roi]]
        finite_global = np.all(np.isfinite(roi_data), axis=0)
        roi_data = roi_data[:, finite_global]
        if roi_data.shape[1] < 10:
            raise RuntimeError(f"{roi}: fewer than ten finite features")

        w, diag = whitening_transform(roi_data.astype(np.float64), condition_values)
        roi_data = (roi_data.astype(np.float64) @ w).astype(np.float32)
        whitening_rows.append({**common, "roi": roi, "n_voxels": roi_data.shape[1], **diag})

        for name, kind, train_source, test_source, positive in CONTRASTS:
            rep_means: list[float] = []

            if kind == "within":
                pool_idx = np.flatnonzero(
                    (source_values == train_source) & np.isin(valence_values, [positive, "neutral"])
                )
                pool_y = (valence_values[pool_idx] == positive).astype(np.int8)
                if len(pool_idx) != 200 or pool_y.sum() != 100:
                    raise RuntimeError(f"{name}: expected 200 trials, 100/100 balanced; got {len(pool_idx)}")

                for rep in range(n_repeats):
                    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
                    fold_accs = []
                    for train_i, test_i in skf.split(pool_idx, pool_y):
                        acc = fit_predict_accuracy(
                            roi_data[pool_idx[train_i]], pool_y[train_i],
                            roi_data[pool_idx[test_i]], pool_y[test_i],
                        )
                        fold_accs.append(acc)
                    rep_mean = float(np.mean(fold_accs))
                    rep_means.append(rep_mean)
                    repetition_rows.append({
                        **common, "roi": roi, "contrast": name, "contrast_kind": kind,
                        "repetition": rep, "mean_fold_accuracy": rep_mean,
                    })

            else:  # cross
                train_pool_idx = np.flatnonzero(
                    (source_values == train_source) & np.isin(valence_values, [positive, "neutral"])
                )
                train_pool_y = (valence_values[train_pool_idx] == positive).astype(np.int8)
                test_pool_idx = np.flatnonzero(
                    (source_values == test_source) & np.isin(valence_values, [positive, "neutral"])
                )
                test_pool_y = (valence_values[test_pool_idx] == positive).astype(np.int8)
                if len(train_pool_idx) != 200 or len(test_pool_idx) != 200:
                    raise RuntimeError(f"{name}: expected 200/200 trials; got {len(train_pool_idx)}/{len(test_pool_idx)}")

                for rep in range(n_repeats):
                    skf_train = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
                    skf_test = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
                    train_splits = [tr for tr, _ in skf_train.split(train_pool_idx, train_pool_y)]
                    test_splits = [te for _, te in skf_test.split(test_pool_idx, test_pool_y)]
                    fold_accs = []
                    for train_i, test_i in zip(train_splits, test_splits):
                        acc = fit_predict_accuracy(
                            roi_data[train_pool_idx[train_i]], train_pool_y[train_i],
                            roi_data[test_pool_idx[test_i]], test_pool_y[test_i],
                        )
                        fold_accs.append(acc)
                    rep_mean = float(np.mean(fold_accs))
                    rep_means.append(rep_mean)
                    repetition_rows.append({
                        **common, "roi": roi, "contrast": name, "contrast_kind": kind,
                        "repetition": rep, "mean_fold_accuracy": rep_mean,
                    })

            subject_rows.append({
                **common, "roi": roi, "contrast": name, "contrast_kind": kind,
                "train_source": train_source, "test_source": test_source, "positive_valence": positive,
                "n_repeats": n_repeats, "n_folds": n_folds,
                "accuracy": float(np.mean(rep_means)),
                "std_across_repeats": float(np.std(rep_means, ddof=1)),
            })

    return pd.DataFrame(subject_rows), pd.DataFrame(repetition_rows), pd.DataFrame(whitening_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--smoothing-mm", type=float, choices=[0, 3, 5, 8], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--atlas-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    data, positions, counts, manifest, reference = load_glmsingle(
        args.input_root, args.atlas, args.atlas_labels, args.smoothing_mm
    )
    manifest = validate_manifest(manifest)

    common = {
        "subject": args.subject,
        "estimator": "glmsingle_typed",
        "branch": args.branch,
        "smoothing_fwhm_mm": args.smoothing_mm,
        "smoothing_location": "post-GLM beta",
        "trial_unit": "single trial",
        "cv": "random 10-fold x30 repeats (Bo et al. 2021 scheme)",
        "classifier": "linear SVC C=1",
        "normalization": "multivariate noise whitening (Ledoit-Wolf shrinkage, condition-demeaned residuals) then per-voxel mean/std fit on outer-training trials only",
    }
    subject, repetitions, whitening = decode_random_kfold_whitened(
        data, positions, manifest, common, n_repeats=args.n_repeats, n_folds=args.n_folds, seed=args.seed
    )
    subject.to_csv(args.output / "subject_results.csv", index=False)
    repetitions.to_csv(args.output / "repetition_results.csv", index=False)
    whitening.to_csv(args.output / "whitening_diagnostics.csv", index=False)
    pd.DataFrame(counts).assign(**common).to_csv(args.output / "roi_counts.csv", index=False)
    provenance = {
        **common,
        "input_root": str(args.input_root),
        "atlas": str(args.atlas),
        "atlas_labels": str(args.atlas_labels),
        "reference_shape": list(reference.shape[:3]),
        "reference_affine": np.asarray(reference.affine).tolist(),
        "n_union_features": int(data.shape[1]),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "nibabel": nib.__version__,
        "scikit_learn": sklearn_version,
        "reference_paper_cv": "Bo et al. 2021, Cerebral Cortex 31:3047-3063, doi:10.1093/cercor/bhaa411",
        "reference_method_whitening": "Walther et al. 2016, NeuroImage -- multivariate noise normalization",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"RANDOM_10FOLD_WHITENED_COMPLETE {args.subject}")


if __name__ == "__main__":
    main()
