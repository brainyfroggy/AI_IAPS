#!/usr/bin/env python3
"""Shared single-trial decoding core replicating Ke Bo's decoding recipe exactly, as used
in Decoding/decoding_test_on_kebodata.ipynb (Cell 24, `decoding_test_on_kebodata.ipynb`):

    StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + s*1000 + rep)
    for each fold: StandardScaler fit on train only, LinearSVC(max_iter=50000, dual='auto')
    20 repetitions, averaged.

That notebook only covers within-source pairs (Ke Bo's own dataset has no AI/source
factor). This module keeps that CV recipe byte-for-byte (same StratifiedKFold call,
same StandardScaler, same LinearSVC arguments, same random_state formula) and extends it
to cross-source contrasts the same way decode_roi_random10fold.py extended Bo et al. 2021's
scheme: independently fold both source pools and pair fold i's training union with fold i's
held-out test fold, using a distinct random_state offset to decorrelate the two foldings.

Operates on the same (data, positions, manifest) interface load_glmsingle() returns, so it
can run unmodified against either GLMsingle or SPM single-trial betas -- only the loader
differs between the two CLI wrappers that import this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import threadpoolctl
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

N_REPEATS = 20
N_FOLDS = 4
RANDOM_STATE = 42
CROSS_TEST_FOLD_OFFSET = 500_000
DEFAULT_N_JOBS = 1


def fit_score_fold(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = LinearSVC(max_iter=50000, dual="auto", random_state=RANDOM_STATE)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return float(accuracy_score(y_test, y_pred))


def subject_seed_component(subject_label: str) -> int:
    digits = "".join(ch for ch in subject_label if ch.isdigit())
    return int(digits) if digits else 0


def decode_one_roi_contrast(
    roi: str,
    roi_data: np.ndarray,
    source_values: np.ndarray,
    valence_values: np.ndarray,
    common: dict,
    contrast_spec: tuple,
    n_repeats: int,
    n_folds: int,
    subject_seed: int,
) -> dict:
    with threadpoolctl.threadpool_limits(limits=1):
        finite_global = np.all(np.isfinite(roi_data), axis=0)
        roi_data = roi_data[:, finite_global]
        if roi_data.shape[1] < 10:
            raise RuntimeError(f"{roi}: fewer than ten finite features")

        name, kind, train_source, test_source, positive = contrast_spec
        if True:
            rep_means: list[float] = []

            if kind == "within":
                pool_idx = np.flatnonzero(
                    (source_values == train_source) & np.isin(valence_values, [positive, "neutral"])
                )
                pool_y = (valence_values[pool_idx] == positive).astype(np.int8)
                if len(pool_idx) != 200 or pool_y.sum() != 100:
                    raise RuntimeError(f"{name}: expected 200 trials, 100/100 balanced; got {len(pool_idx)}")
                x = roi_data[pool_idx]
                y = pool_y

                for rep in range(n_repeats):
                    seed = RANDOM_STATE + subject_seed * 1000 + rep
                    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                    fold_accs = [
                        fit_score_fold(x[tr], y[tr], x[te], y[te])
                        for tr, te in skf.split(x, y)
                    ]
                    rep_means.append(float(np.mean(fold_accs)))

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
                x_train_pool = roi_data[train_pool_idx]
                x_test_pool = roi_data[test_pool_idx]

                for rep in range(n_repeats):
                    seed_train = RANDOM_STATE + subject_seed * 1000 + rep
                    seed_test = seed_train + CROSS_TEST_FOLD_OFFSET
                    skf_train = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_train)
                    skf_test = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_test)
                    train_splits = [tr for tr, _ in skf_train.split(x_train_pool, train_pool_y)]
                    test_splits = [te for _, te in skf_test.split(x_test_pool, test_pool_y)]
                    fold_accs = [
                        fit_score_fold(x_train_pool[tr], train_pool_y[tr], x_test_pool[te], test_pool_y[te])
                        for tr, te in zip(train_splits, test_splits)
                    ]
                    rep_means.append(float(np.mean(fold_accs)))

            return {
                **common, "roi": roi, "contrast": name, "contrast_kind": kind,
                "train_source": train_source, "test_source": test_source, "positive_valence": positive,
                "n_repeats": n_repeats, "n_folds": n_folds,
                "accuracy": float(np.mean(rep_means)),
                "std_across_repeats": float(np.std(rep_means, ddof=1)),
            }


def kebo_decode(
    data: np.ndarray,
    positions: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    common: dict,
    roi_order: tuple[str, ...],
    contrasts: tuple[tuple, ...],
    n_repeats: int = N_REPEATS,
    n_folds: int = N_FOLDS,
    n_jobs: int = DEFAULT_N_JOBS,
) -> pd.DataFrame:
    """Parallelizes across (roi, contrast) units -- 136 independent tasks for the usual
    17-ROI x 8-contrast design -- rather than by ROI alone, since ROI voxel counts vary
    widely (e.g. IPS ~1000 vs V3b ~100) and coarser chunking left large ROIs as stragglers
    holding up otherwise-idle workers. threadpool_limits(1) inside each worker prevents
    BLAS thread oversubscription when n_jobs > 1 processes run concurrently.
    """
    source_values = manifest["source"].astype(str).str.lower().to_numpy()
    valence_values = manifest["valence"].astype(str).str.lower().to_numpy()
    subject_seed = subject_seed_component(str(common.get("subject", "")))

    tasks = [(roi, contrast_spec) for roi in roi_order for contrast_spec in contrasts]
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(decode_one_roi_contrast)(
            roi, data[:, positions[roi]], source_values, valence_values,
            common, contrast_spec, n_repeats, n_folds, subject_seed,
        )
        for roi, contrast_spec in tasks
    )
    return pd.DataFrame(rows)
