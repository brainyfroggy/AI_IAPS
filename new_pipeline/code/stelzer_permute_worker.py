#!/usr/bin/env python3
"""Phase B of the Stelzer et al. 2013 / Bo et al. 2021 permutation replication.

For one (subject, ROI): 100 label shuffles x 8 contrasts, ONE single-pass 10-fold CV per
shuffle (not the 30-repeat average used for the observed statistic -- this is what makes
the replication "faithful": Bo et al.'s paper says each shuffled run generates ONE
chance-level accuracy). Labels are shuffled independently within each source's 200-trial
pool (for cross contrasts, train-source and test-source pools are shuffled independently
of each other), so class balance (100/100) and the natural/AI split both survive --
only the pattern<->valence link is destroyed.

Reads the Phase A cache (stelzer_cache_subject_matrices.py) instead of touching raw
NIfTI/HDF5 data, so this step is CPU-bound and safe to run at high concurrency.
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import threadpoolctl
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import CONTRASTS  # noqa: E402

N_PERMS = 100
N_FOLDS = 10


def fit_predict_accuracy(x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray) -> float:
    mean = x_train.mean(axis=0, dtype=np.float64)
    std = x_train.std(axis=0, ddof=0, dtype=np.float64)
    valid = np.isfinite(mean) & np.isfinite(std) & (std > 1e-7)
    if int(valid.sum()) < 10:
        raise RuntimeError("too few train-variable features")
    xtr = ((x_train[:, valid] - mean[valid]) / std[valid]).astype(np.float32)
    xte = ((x_test[:, valid] - mean[valid]) / std[valid]).astype(np.float32)
    clf = SVC(kernel="linear", C=1.0, cache_size=1024)
    clf.fit(xtr, y_train)
    pred = clf.predict(xte)
    return float(np.mean(pred == y_test))


def load_cache(cache_path: Path) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    npz = np.load(cache_path)
    data = npz["data"]
    source = npz["source"]
    valence = npz["valence"]
    positions = {k[len("positions_"):]: npz[k] for k in npz.files if k.startswith("positions_")}
    return data, positions, source, valence


def permute_within(roi_data: np.ndarray, pool_idx: np.ndarray, pool_y: np.ndarray,
                    rng: np.random.Generator, n_perms: int, n_folds: int) -> list[float]:
    nulls = []
    x = roi_data[pool_idx]
    for _ in range(n_perms):
        y_shuffled = rng.permutation(pool_y)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
        fold_accs = [fit_predict_accuracy(x[tr], y_shuffled[tr], x[te], y_shuffled[te])
                     for tr, te in skf.split(x, y_shuffled)]
        nulls.append(float(np.mean(fold_accs)))
    return nulls


def permute_cross(roi_data: np.ndarray, train_pool_idx: np.ndarray, train_pool_y: np.ndarray,
                   test_pool_idx: np.ndarray, test_pool_y: np.ndarray,
                   rng: np.random.Generator, n_perms: int, n_folds: int) -> list[float]:
    # No CV: each shuffle fits on all 200 (shuffled-label) train-pool trials and
    # scores once on all 200 (independently shuffled-label) test-pool trials --
    # mirrors the fixed no-fold design now used for the observed cross statistic.
    # n_folds is accepted (unused) only to keep the call signature uniform with
    # permute_within.
    del n_folds
    nulls = []
    x_train_pool = roi_data[train_pool_idx]
    x_test_pool = roi_data[test_pool_idx]
    for _ in range(n_perms):
        y_train_shuffled = rng.permutation(train_pool_y)
        y_test_shuffled = rng.permutation(test_pool_y)
        acc = fit_predict_accuracy(x_train_pool, y_train_shuffled, x_test_pool, y_test_shuffled)
        nulls.append(acc)
    return nulls


def process_subject_roi(subject: int, roi: str, cache_dir: Path, out_dir: Path,
                         n_perms: int = N_PERMS, n_folds: int = N_FOLDS, seed_base: int = 42,
                         contrasts: tuple = CONTRASTS) -> dict:
    out_path = out_dir / f"sub{subject:02d}_{roi}.csv"
    if out_path.exists():
        return {"subject": subject, "roi": roi, "status": "skipped_existing"}
    try:
        with threadpoolctl.threadpool_limits(limits=1):
            cache_path = cache_dir / f"sub-{subject:02d}.npz"
            data, positions, source, valence = load_cache(cache_path)
            roi_data = data[:, positions[roi]]
            finite = np.all(np.isfinite(roi_data), axis=0)
            roi_data = roi_data[:, finite].astype(np.float64)
            if roi_data.shape[1] < 10:
                raise RuntimeError(f"{roi}: fewer than ten finite features")

            rows = []
            for name, kind, train_source, test_source, positive in contrasts:
                seed = seed_base + subject * 100_000 + zlib.crc32(f"{roi}|{name}".encode()) % 1_000_000
                rng = np.random.default_rng(seed)

                if kind == "within":
                    pool_idx = np.flatnonzero(
                        (source == train_source) & np.isin(valence, [positive, "neutral"])
                    )
                    pool_y = (valence[pool_idx] == positive).astype(np.int8)
                    if len(pool_idx) != 200 or pool_y.sum() != 100:
                        raise RuntimeError(f"{name}: expected 200 trials, 100/100 balanced; got {len(pool_idx)}")
                    nulls = permute_within(roi_data, pool_idx, pool_y, rng, n_perms, n_folds)
                else:
                    train_pool_idx = np.flatnonzero(
                        (source == train_source) & np.isin(valence, [positive, "neutral"])
                    )
                    train_pool_y = (valence[train_pool_idx] == positive).astype(np.int8)
                    test_pool_idx = np.flatnonzero(
                        (source == test_source) & np.isin(valence, [positive, "neutral"])
                    )
                    test_pool_y = (valence[test_pool_idx] == positive).astype(np.int8)
                    if len(train_pool_idx) != 200 or len(test_pool_idx) != 200:
                        raise RuntimeError(f"{name}: expected 200/200 trials")
                    nulls = permute_cross(roi_data, train_pool_idx, train_pool_y,
                                          test_pool_idx, test_pool_y, rng, n_perms, n_folds)

                for perm_i, acc in enumerate(nulls):
                    rows.append({"subject": f"Sub{subject:02d}", "roi": roi, "contrast": name,
                                "permutation_index": perm_i, "null_accuracy": acc})

            pd.DataFrame(rows).to_csv(out_path, index=False)
        return {"subject": subject, "roi": roi, "status": "pass"}
    except Exception as exc:  # noqa: BLE001
        return {"subject": subject, "roi": roi, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/stelzer_permutation/cache"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/stelzer_permutation/null_pools"))
    ap.add_argument("--n-perms", type=int, default=N_PERMS)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--contrasts", type=str, default=None,
                    help="comma-separated subset of contrast names to run (default: all 8)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.contrasts:
        wanted = set(args.contrasts.split(","))
        contrasts = tuple(c for c in CONTRASTS if c[0] in wanted)
        missing = wanted - {c[0] for c in contrasts}
        if missing:
            raise RuntimeError(f"unknown contrast name(s): {missing}")
    else:
        contrasts = CONTRASTS
    res = process_subject_roi(args.subject, args.roi, args.cache_dir, args.out_dir,
                              args.n_perms, args.n_folds, contrasts=contrasts)
    print(res)
    if res["status"] == "pass":
        print(f"STELZER_PERMUTE_COMPLETE Sub{args.subject:02d} {args.roi}")


if __name__ == "__main__":
    main()
