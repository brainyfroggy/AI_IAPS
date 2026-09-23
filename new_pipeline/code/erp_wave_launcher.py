#!/usr/bin/env python3
"""Dispatch ERP-style observed+permutation units across a process pool.

Calls process_subject_roi() in-process (not via subprocess) to avoid paying Python/sklearn
import overhead 510 times; each unit only reopens its subject's ~20MB cached .npz, so the
cost is almost entirely the SVM fits. threadpool_limits(1) inside each worker (set by the
worker itself) prevents BLAS oversubscription.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER  # noqa: E402
from erp_permute_worker import process_subject_roi, N_PERMS, N_REPEATS, N_FOLDS  # noqa: E402
from aal3_common import load_aal3_kept_regions  # noqa: E402

DEFAULT_SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31, 33, 34]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--rois", nargs="+", default=None)
    ap.add_argument("--roi-manifest", type=Path, default=None,
                    help="AAL3-style kept-regions CSV (id, roi_name, kept); overrides --rois")
    ap.add_argument("--contrast-kind", choices=["within", "cross"], default=None)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--n-perms", type=int, default=N_PERMS)
    ap.add_argument("--n-repeats", type=int, default=N_REPEATS)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/stelzer_permutation/cache"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/erp_permutation"))
    args = ap.parse_args()

    if args.roi_manifest is not None:
        rois = [name for _rid, name in load_aal3_kept_regions(args.roi_manifest)]
    elif args.rois is not None:
        rois = args.rois
    else:
        rois = list(ROI_ORDER)

    n_contrasts = 4 if args.contrast_kind else 8
    tasks = [(s, r) for s in args.subjects for r in rois]
    print(f"dispatching {len(tasks)} (subject, roi) units across {args.workers} workers "
          f"({args.n_perms} shuffles x {args.n_repeats} repeats x {n_contrasts} "
          f"{args.contrast_kind or 'all'} contrasts each)", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_subject_roi, s, r, args.cache_dir, args.out_dir,
                        args.n_perms, args.n_repeats, args.n_folds,
                        contrast_kind=args.contrast_kind): (s, r)
            for s, r in tasks
        }
        n_done = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            n_done += 1
            status = res["status"]
            marker = "" if status in ("pass", "skipped_existing") else f" ({res.get('error')})"
            if status == "fail" or n_done % 25 == 0 or n_done == len(tasks):
                print(f"  [{n_done}/{len(tasks)}] sub-{res['subject']:02d} {res['roi']}: {status}{marker}", flush=True)

    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"ERP_WAVE_COMPLETE total={len(results)} failed={n_fail}", flush=True)
    for r in results:
        if r["status"] == "fail":
            print(f"  FAILED sub-{r['subject']:02d} {r['roi']}: {r.get('error')}", flush=True)


if __name__ == "__main__":
    main()
