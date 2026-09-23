#!/usr/bin/env python3
"""Dispatch run_glmsingle_fixedhrf_pilot.py's per-subject fixed-HRF GLMsingle rerun
across a process pool. Each subject is a fully independent GLMsingle fit (its own
BOLD load, design, GLM_single(...).fit() call), so this parallelizes the same way
glmsingle_wave_launcher.py does for the production run -- modest concurrency since
each worker holds a full run's masked BOLD (n_voxels x n_time, several GB) in
memory during the fit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_glmsingle_fixedhrf_pilot import (  # noqa: E402
    load_cohort, load_pilot_generalized, derive_all_subjects, run_fixedhrf,
    _fresh_import, GLMSINGLE_SPACE_SCRIPT, DEFAULT_COHORT_CONFIG,
)


def run_one(subject: int, cohort_config: Path, fmriprep_derivatives_root: Path,
            per_subject_bids_root: Path, branch: str, output_root: Path, n_pcs: int,
            want_glmdenoise: int = 0, want_fracridge: int = 0, want_library: int = 0) -> dict:
    out_dir = output_root / f"sub-{subject:02d}"
    if out_dir.exists():
        return {"subject": subject, "status": "skipped_existing"}
    try:
        cohort = load_cohort(cohort_config)
        pilot = load_pilot_generalized(cohort)
        space = _fresh_import(GLMSINGLE_SPACE_SCRIPT, f"ai_iaps_glmsingle_space_fixedhrf_s{subject}")
        label = f"Sub{subject:02d}"
        fmriprep_root = fmriprep_derivatives_root / label
        bids_root = per_subject_bids_root / label
        run_fixedhrf(subject, pilot, space, branch, fmriprep_root, bids_root, out_dir, n_pcs,
                     want_glmdenoise, want_fracridge, want_library)
        return {"subject": subject, "status": "pass"}
    except Exception as exc:  # noqa: BLE001
        return {"subject": subject, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--cohort-config", type=Path, default=DEFAULT_COHORT_CONFIG)
    ap.add_argument("--fmriprep-derivatives-root", type=Path, required=True)
    ap.add_argument("--per-subject-bids-root", type=Path, required=True)
    ap.add_argument("--branch", default="mni_res_native")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--n-pcs", type=int, default=10)
    ap.add_argument("--want-glmdenoise", type=int, default=0, choices=[0, 1])
    ap.add_argument("--want-fracridge", type=int, default=0, choices=[0, 1])
    ap.add_argument("--want-library", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    cohort = load_cohort(args.cohort_config)
    for s in args.subjects:
        if s not in derive_all_subjects(cohort):
            raise ValueError(f"Subject {s} not in cohort")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, s, args.cohort_config, args.fmriprep_derivatives_root,
                        args.per_subject_bids_root, args.branch, args.output_root, args.n_pcs,
                        args.want_glmdenoise, args.want_fracridge, args.want_library): s
            for s in args.subjects
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"  sub-{res['subject']:02d}: {res['status']}"
                  + (f" ({res.get('error')})" if res["status"] == "fail" else ""), flush=True)

    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"GLMSINGLE_FIXEDHRF_WAVE_COMPLETE total={len(results)} failed={n_fail}", flush=True)


if __name__ == "__main__":
    main()
