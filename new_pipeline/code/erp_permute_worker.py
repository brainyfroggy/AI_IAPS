#!/usr/bin/env python3
"""ERP-style (trial-averaged) decoding + Stelzer/Bo permutation null, per (subject, ROI).

See docs/ERPSTYLE_PERMUTATION_PLAN.md. Computes the OBSERVED statistic and the 100-shuffle
null pool in ONE pass, because the voxel-source z-score is invariant under valence
permutation (it pools all 300 trials of a source and never reads valence labels), so it
can be computed once per (subject, ROI) and reused for the observed decode and all 100
shuffles alike. That also guarantees observed and null share bit-identical preprocessing,
which is what makes the empirical p-value interpretable at face value.

Both observed and null use --n-repeats (default 30). Matching them is deliberate: the
permutation p-value is P(null_draw >= observed), so a null that averaged MORE repeats than
the observed statistic would have tighter draws, too thin an upper tail, and p-values
biased small (anti-conservative).

Label shuffling happens at the SINGLE-TRIAL level -- the 200-trial (100 positive + 100
neutral) pool is repartitioned, then chunking/averaging is redone from the shuffled labels.
Shuffling already-averaged patterns instead would leave the averaging step computed from
true labels and would not be a null at all. For cross-source contrasts the train-source and
test-source pools are shuffled independently (same as stelzer_permute_worker.permute_cross).

Reads the existing stelzer_permutation/cache/ (Kastner ROIs, 8mm) -- the same matrices the
single-trial permutation used, so no re-smoothing or re-caching is needed.
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import threadpoolctl

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import CONTRASTS  # noqa: E402
from decode_roi_erpstyle import (  # noqa: E402
    compute_voxel_source_zscore,
    condition_label,
    decode_within_avg,
    decode_cross_avg,
)

N_PERMS = 100
N_REPEATS = 30
N_FOLDS = 4
N_AVG_GROUPS = 3


def load_cache(cache_path: Path):
    npz = np.load(cache_path)
    data = npz["data"]
    source = npz["source"]
    valence = npz["valence"]
    positions = {k[len("positions_"):]: npz[k] for k in npz.files if k.startswith("positions_")}
    return data, positions, source, valence


def shuffle_pair(d1: np.ndarray, d2: np.ndarray, rng: np.random.Generator):
    """Repartition the pooled 200 trials into two 100-trial groups at random."""
    pool = np.vstack([d1, d2])
    perm = rng.permutation(len(pool))
    return pool[perm[:len(d1)]], pool[perm[len(d1):]]


def process_subject_roi(subject: int, roi: str, cache_dir: Path, out_dir: Path,
                        n_perms: int = N_PERMS, n_repeats: int = N_REPEATS,
                        n_folds: int = N_FOLDS, n_avg_groups: int = N_AVG_GROUPS,
                        seed_base: int = 42, contrast_kind: str | None = None) -> dict:
    obs_path = out_dir / "observed" / f"sub{subject:02d}_{roi}.csv"
    null_path = out_dir / "null_pools" / f"sub{subject:02d}_{roi}.csv"
    if obs_path.exists() and null_path.exists():
        return {"subject": subject, "roi": roi, "status": "skipped_existing"}
    try:
        with threadpoolctl.threadpool_limits(limits=1):
            data, positions, source, valence = load_cache(cache_dir / f"sub-{subject:02d}.npz")
            roi_data = data[:, positions[roi]]
            finite = np.all(np.isfinite(roi_data), axis=0)
            roi_data = roi_data[:, finite].astype(np.float64)
            if roi_data.shape[1] < 10:
                raise RuntimeError(f"{roi}: fewer than ten finite features")

            # condition matrices, then the shuffle-invariant voxel-source z-score (once)
            cond_data = {}
            for src in ("natural", "ai"):
                for val in ("pleasant", "neutral", "unpleasant"):
                    idx = np.flatnonzero((source == src) & (valence == val))
                    if len(idx) != 100:
                        raise RuntimeError(f"{roi}/{src}/{val}: expected 100 trials, got {len(idx)}")
                    cond_data[condition_label(src, val)] = roi_data[idx]
            voxel_source_z = compute_voxel_source_zscore(cond_data)

            obs_rows, null_rows = [], []
            for name, kind, train_source, test_source, positive in CONTRASTS:
                if contrast_kind is not None and kind != contrast_kind:
                    continue
                seed = seed_base + subject * 100_000 + zlib.crc32(f"{roi}|{name}".encode()) % 1_000_000
                rng = np.random.default_rng(seed)

                if kind == "within":
                    d1 = voxel_source_z[condition_label(train_source, positive)]
                    d2 = voxel_source_z[condition_label(train_source, "neutral")]
                    obs = decode_within_avg(d1, d2, n_repeats, n_folds, n_avg_groups)
                    for perm_i in range(n_perms):
                        s1, s2 = shuffle_pair(d1, d2, rng)
                        null_rows.append({
                            "subject": f"Sub{subject:02d}", "roi": roi, "contrast": name,
                            "permutation_index": perm_i,
                            "null_accuracy": decode_within_avg(s1, s2, n_repeats, n_folds, n_avg_groups),
                        })
                else:
                    tr1 = voxel_source_z[condition_label(train_source, positive)]
                    tr2 = voxel_source_z[condition_label(train_source, "neutral")]
                    te1 = voxel_source_z[condition_label(test_source, positive)]
                    te2 = voxel_source_z[condition_label(test_source, "neutral")]
                    obs = decode_cross_avg(tr1, tr2, te1, te2, n_repeats, n_folds)
                    for perm_i in range(n_perms):
                        a1, a2 = shuffle_pair(tr1, tr2, rng)
                        b1, b2 = shuffle_pair(te1, te2, rng)
                        null_rows.append({
                            "subject": f"Sub{subject:02d}", "roi": roi, "contrast": name,
                            "permutation_index": perm_i,
                            "null_accuracy": decode_cross_avg(a1, a2, b1, b2, n_repeats, n_folds),
                        })

                obs_rows.append({
                    "subject": f"Sub{subject:02d}", "roi": roi, "contrast": name,
                    "contrast_kind": kind, "train_source": train_source,
                    "test_source": test_source, "positive_valence": positive,
                    "n_repeats": n_repeats, "n_folds": n_folds, "n_avg_groups": n_avg_groups,
                    "accuracy": obs,
                })

            obs_path.parent.mkdir(parents=True, exist_ok=True)
            null_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(obs_rows).to_csv(obs_path, index=False)
            pd.DataFrame(null_rows).to_csv(null_path, index=False)
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
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/erp_permutation"))
    ap.add_argument("--n-perms", type=int, default=N_PERMS)
    ap.add_argument("--n-repeats", type=int, default=N_REPEATS)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--contrast-kind", choices=["within", "cross"], default=None)
    args = ap.parse_args()
    res = process_subject_roi(args.subject, args.roi, args.cache_dir, args.out_dir,
                              args.n_perms, args.n_repeats, args.n_folds,
                              contrast_kind=args.contrast_kind)
    print(res)


if __name__ == "__main__":
    main()
