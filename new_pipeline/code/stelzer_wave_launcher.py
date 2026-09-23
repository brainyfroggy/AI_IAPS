#!/usr/bin/env python3
"""Phase B orchestration: dispatch all (subject, ROI) permutation tasks across a process
pool. Calls process_subject_roi() directly in-process (not via subprocess) to avoid
paying Python/sklearn import overhead 510 times -- each task only needs to reopen its
subject's small cached .npz, which is fast, so most of the cost is the actual CPU work.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER, CONTRASTS  # noqa: E402
from stelzer_permute_worker import process_subject_roi, N_PERMS, N_FOLDS  # noqa: E402

DEFAULT_SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31, 33, 34]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--rois", nargs="+", default=list(ROI_ORDER))
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--n-perms", type=int, default=N_PERMS)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/stelzer_permutation/cache"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/stelzer_permutation/null_pools"))
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

    tasks = [(s, r) for s in args.subjects for r in args.rois]
    print(f"dispatching {len(tasks)} (subject, roi) tasks across {args.workers} workers "
          f"({args.n_perms} perms x {args.n_folds} folds x {len(contrasts)} contrasts each)", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_subject_roi, s, r, args.cache_dir, args.out_dir,
                        args.n_perms, args.n_folds, 42, contrasts): (s, r)
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
    print(f"STELZER_WAVE_COMPLETE total={len(results)} failed={n_fail}", flush=True)
    if n_fail:
        for r in results:
            if r["status"] == "fail":
                print(f"  FAILED sub-{r['subject']:02d} {r['roi']}: {r.get('error')}", flush=True)


if __name__ == "__main__":
    main()
