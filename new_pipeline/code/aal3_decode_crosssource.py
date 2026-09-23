#!/usr/bin/env python3
"""Phase 2 of the whole-brain AAL3 cross-source decoding plan.

For each of the 154 frozen AAL3 regions x 4 cross-source contrasts: fit ONE
linear SVC on all 200 train-source trials, score ONCE on all 200 test-source
trials. No folds, no repeats -- the corrected cross-source design (2026-08-21):
source B was never part of training regardless of fold structure, so a single
train-on-all/test-on-all split is the right (and deterministic) evaluation.

Task granularity is per-SUBJECT, not per-(region,contrast): unlike the
Kastner/Wang within-source work (10-fold x100 repeats = 1000 fits/unit, which
needed fine-grained parallelism for load balance), each subject's full
154 x 4 = 616 single fits here is cheap (~tens of seconds), so the natural
unit is "load this subject's cache once, do all its fits in-process" --
avoiding 154x redundant decompression of the same ~470MB cached array that
per-region task granularity would cost.
"""

from __future__ import annotations

import argparse
import json
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import threadpoolctl
from sklearn import __version__ as sklearn_version
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).parent))
from aal3_common import load_aal3_kept_regions  # noqa: E402

CROSS_CONTRASTS = (
    ("train_natural_pleasant_vs_neutral_test_ai", "natural", "ai", "pleasant"),
    ("train_ai_pleasant_vs_neutral_test_natural", "ai", "natural", "pleasant"),
    ("train_natural_unpleasant_vs_neutral_test_ai", "natural", "ai", "unpleasant"),
    ("train_ai_unpleasant_vs_neutral_test_natural", "ai", "natural", "unpleasant"),
)

DEFAULT_SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31, 33, 34]


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


def decode_one_subject(subject: int, cache_path: Path, region_names: list[str], common: dict) -> dict:
    try:
        with threadpoolctl.threadpool_limits(limits=1):
            npz = np.load(cache_path)
            data = npz["data"]
            source_values = npz["source"]
            valence_values = npz["valence"]

            rows = []
            for name in region_names:
                positions = npz[f"positions_{name}"]
                roi_data = data[:, positions]
                finite_global = np.all(np.isfinite(roi_data), axis=0)
                roi_data = roi_data[:, finite_global]
                if roi_data.shape[1] < 10:
                    raise RuntimeError(f"{name}: fewer than ten finite features")

                for cname, train_source, test_source, positive in CROSS_CONTRASTS:
                    train_pool_idx = np.flatnonzero(
                        (source_values == train_source) & np.isin(valence_values, [positive, "neutral"])
                    )
                    train_pool_y = (valence_values[train_pool_idx] == positive).astype(np.int8)
                    test_pool_idx = np.flatnonzero(
                        (source_values == test_source) & np.isin(valence_values, [positive, "neutral"])
                    )
                    test_pool_y = (valence_values[test_pool_idx] == positive).astype(np.int8)
                    if len(train_pool_idx) != 200 or len(test_pool_idx) != 200:
                        raise RuntimeError(
                            f"{cname}/{name}: expected 200/200 trials; "
                            f"got {len(train_pool_idx)}/{len(test_pool_idx)}"
                        )

                    acc = fit_predict_accuracy(
                        roi_data[train_pool_idx], train_pool_y,
                        roi_data[test_pool_idx], test_pool_y,
                    )
                    rows.append({
                        **common, "roi": name, "contrast": cname, "contrast_kind": "cross",
                        "train_source": train_source, "test_source": test_source,
                        "positive_valence": positive,
                        "cv": "single train/test split, no CV (train on all 200 source-A trials, "
                              "test on all 200 source-B trials)",
                        "n_features": int(roi_data.shape[1]),
                        "accuracy": acc,
                    })
        return {"subject": subject, "status": "pass", "rows": rows}
    except Exception as exc:  # noqa: BLE001
        return {"subject": subject, "status": "fail", "error": f"{type(exc).__name__}: {exc}", "rows": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--roi-manifest", type=Path, required=True)
    ap.add_argument("--smoothing-mm", type=float, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    kept_regions = load_aal3_kept_regions(args.roi_manifest)
    region_names = [name for _, name in kept_regions]
    print(f"kept regions: {len(region_names)}, subjects: {len(args.subjects)}", flush=True)

    common = {
        "estimator": "glmsingle_typed",
        "atlas": "AAL3v1 (170 raw ids, L/R separate, 154 kept)",
        "smoothing_fwhm_mm": args.smoothing_mm,
        "smoothing_location": "post-GLM beta",
        "trial_unit": "single trial",
        "classifier": "linear SVC C=1",
        "normalization": "per-voxel mean/std fit on train-source trials only",
    }

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(decode_one_subject, s, args.cache_dir / f"sub-{s:02d}.npz",
                        region_names, {**common, "subject": f"Sub{s:02d}"}): s
            for s in args.subjects
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"  sub-{res['subject']:02d}: {res['status']}"
                  + (f" ({res.get('error')})" if res["status"] == "fail" else ""), flush=True)

    n_fail = sum(1 for r in results if r["status"] == "fail")
    all_rows = [row for r in results for row in r["rows"]]
    subject_df = pd.DataFrame(all_rows)
    subject_df.to_csv(args.output / "subject_results.csv", index=False)

    provenance = {
        **common,
        "n_subjects": len(args.subjects),
        "n_regions": len(region_names),
        "n_contrasts": len(CROSS_CONTRASTS),
        "n_rows_expected": len(args.subjects) * len(region_names) * len(CROSS_CONTRASTS),
        "n_rows_actual": len(all_rows),
        "n_failed_subjects": n_fail,
        "roi_manifest": str(args.roi_manifest),
        "cache_dir": str(args.cache_dir),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"AAL3_DECODE_COMPLETE smoothing={args.smoothing_mm} subjects={len(args.subjects)} "
          f"failed={n_fail} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
