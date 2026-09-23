#!/usr/bin/env python3
"""Random 10-fold decoding in the 17 Kastner/Wang ROIs, repeated 100x, matching
Bo et al. 2021 (Cerebral Cortex)'s cross-validation scheme exactly:

  "all the data was divided into 10 equal subdatasets, nine of which comprised
  training data ... and the remaining one ... testing ... The decoding accuracies
  of 10 such procedures were averaged. To further ensure the stability of the
  decoding result, we repeated 10-fold partition 100 times ... and averaged the
  accuracies to yield the decoding accuracy for each ROI."

This replaces decode_roi_singletrial.py's leave-one-run-out CV with this random
resampling scheme. Everything else is identical: same 17 ROIs, same beta loading
(GLMsingle Type-D, MNI native-res, 8mm post-GLM mask-normalized smoothing), same
8 contrasts (4 within-source, 4 cross-source), same linear SVC(C=1).

CV design for the 8 contrasts:
  - within (train_source == test_source): standard stratified 10-fold CV on the
    pooled 200 trials (100 positive-valence + 100 neutral) for that source --
    directly matches Bo et al.'s single-dataset design.
  - cross (train_source != test_source): Bo et al. never had this factor (no
    natural/AI split). Generalized here by independently stratified-10-folding
    BOTH source's 200 trials each repetition, pairing fold i's 9-fold union
    (180 trials) from train_source with fold i's held-out fold (20 trials) from
    test_source -- keeps the same 180-train/20-test shape and reintroduces
    genuine per-repetition randomness on both sides, consistent with "within".

Reuses load_glmsingle/validate_manifest unchanged from decode_roi_singletrial.py.

Parallelized across (roi, contrast) -- 136 independent units for the usual 17 ROI x
8 contrast design -- since each unit's n_repeats x n_folds fits are independent of
every other unit. threadpool_limits(1) inside each worker prevents BLAS thread
oversubscription when --n-jobs > 1 processes run concurrently (same pattern as
kebo_decode_core.py). NOTE: parallelizing changes the RNG seeding from one shared
sequential stream (old serial version) to one independent per-(roi,contrast) stream
derived from --seed + subject + roi + contrast, so results from this version are not
bit-identical to the old serial version at the same --seed -- both are equally valid
draws from the same CV scheme, just not reproductions of each other.
"""

from __future__ import annotations

import argparse
import json
import platform
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import threadpoolctl
from joblib import Parallel, delayed
from sklearn import __version__ as sklearn_version
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

N_REPEATS = 100
N_FOLDS = 10
DEFAULT_N_JOBS = 1


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


def decode_one_roi_contrast(
    roi: str,
    roi_data: np.ndarray,
    source_values: np.ndarray,
    valence_values: np.ndarray,
    common: dict,
    contrast_spec: tuple,
    n_repeats: int,
    n_folds: int,
    seed_base: int,
) -> tuple[dict, list[dict]]:
    with threadpoolctl.threadpool_limits(limits=1):
        finite_global = np.all(np.isfinite(roi_data), axis=0)
        roi_data = roi_data[:, finite_global]
        if roi_data.shape[1] < 10:
            raise RuntimeError(f"{roi}: fewer than ten finite features")

        name, kind, train_source, test_source, positive = contrast_spec
        seed = seed_base + zlib.crc32(f"{roi}|{name}".encode()) % 1_000_000
        rng = np.random.default_rng(seed)

        rep_means: list[float] = []
        repetition_rows: list[dict] = []

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
            # No cross-validation: source B was never part of training regardless
            # of fold structure, so a single train-on-all/test-on-all split is the
            # correct (and deterministic) evaluation -- no repeats needed.
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

            acc = fit_predict_accuracy(
                roi_data[train_pool_idx], train_pool_y,
                roi_data[test_pool_idx], test_pool_y,
            )
            rep_means = [acc]

        if kind == "within":
            cv_desc = f"random 10-fold x{n_repeats} repeats (Bo et al. 2021 scheme)"
            row_n_repeats, row_n_folds = n_repeats, n_folds
            accuracy = float(np.mean(rep_means))
            std_across_repeats = float(np.std(rep_means, ddof=1))
        else:
            cv_desc = "single train/test split, no CV (train on all 200 source-A trials, test on all 200 source-B trials)"
            row_n_repeats, row_n_folds = 1, 1
            accuracy = float(rep_means[0])
            std_across_repeats = float("nan")

        subject_row = {
            **common, "roi": roi, "contrast": name, "contrast_kind": kind,
            "train_source": train_source, "test_source": test_source, "positive_valence": positive,
            "cv": cv_desc,
            "n_repeats": row_n_repeats, "n_folds": row_n_folds,
            "accuracy": accuracy,
            "std_across_repeats": std_across_repeats,
        }
        return subject_row, repetition_rows


def decode_random_kfold(
    data: np.ndarray,
    positions: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    common: dict,
    n_repeats: int = N_REPEATS,
    n_folds: int = N_FOLDS,
    seed: int = 0,
    n_jobs: int = DEFAULT_N_JOBS,
    contrasts: tuple = CONTRASTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_values = manifest["source"].astype(str).str.lower().to_numpy()
    valence_values = manifest["valence"].astype(str).str.lower().to_numpy()

    tasks = [(roi, contrast_spec) for roi in ROI_ORDER for contrast_spec in contrasts]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(decode_one_roi_contrast)(
            roi, data[:, positions[roi]], source_values, valence_values,
            common, contrast_spec, n_repeats, n_folds, seed,
        )
        for roi, contrast_spec in tasks
    )

    subject_rows = [r[0] for r in results]
    repetition_rows = [row for r in results for row in r[1]]
    return pd.DataFrame(subject_rows), pd.DataFrame(repetition_rows)


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
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel workers across (ROI, contrast)")
    parser.add_argument(
        "--contrasts", type=str, default=None,
        help="comma-separated subset of contrast names to run (default: all 8)",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    if args.contrasts:
        wanted = set(args.contrasts.split(","))
        contrasts = tuple(c for c in CONTRASTS if c[0] in wanted)
        missing = wanted - {c[0] for c in contrasts}
        if missing:
            raise RuntimeError(f"unknown contrast name(s): {missing}")
    else:
        contrasts = CONTRASTS

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
        "cv": f"random 10-fold x{args.n_repeats} repeats (Bo et al. 2021 scheme)",
        "classifier": "linear SVC C=1",
        "normalization": "per-voxel mean/std fit on outer-training trials only",
    }
    subject, repetitions = decode_random_kfold(
        data, positions, manifest, common, n_repeats=args.n_repeats, n_folds=args.n_folds,
        seed=args.seed, n_jobs=args.n_jobs, contrasts=contrasts,
    )
    subject.to_csv(args.output / "subject_results.csv", index=False)
    repetitions.to_csv(args.output / "repetition_results.csv", index=False)
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
        "reference_paper": "Bo et al. 2021, Cerebral Cortex 31:3047-3063, doi:10.1093/cercor/bhaa411",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"RANDOM_10FOLD_COMPLETE {args.subject}")


if __name__ == "__main__":
    main()
