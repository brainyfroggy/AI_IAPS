#!/usr/bin/env python3
"""Run decode_roi_random10fold_whitened.py across the full cohort with bounded parallelism."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")
KASTNER = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/full_cohort_raw_pipeline_28/resources/kastner")
PYTHON = "/home/yujun/.cache/ai_iaps_pilot_venv/bin/python3"
SCRIPT = ROOT / "code" / "decode_roi_random10fold_whitened.py"

DEFAULT_SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31, 33]


def run_one(subject: int, n_repeats: int, out_root: str) -> dict:
    label = f"Sub{subject:02d}"
    out_dir = Path(out_root) / f"sub-{subject:02d}"
    if (out_dir / "subject_results.csv").exists():
        return {"subject": subject, "status": "skipped_existing"}
    cmd = [
        PYTHON, str(SCRIPT),
        "--subject", label,
        "--branch", "mni_res_native",
        "--smoothing-mm", "8",
        "--input-root", str(ROOT / "glmsingle" / f"sub-{subject:02d}"),
        "--atlas", str(KASTNER / "kastner.nii.gz"),
        "--atlas-labels", str(KASTNER / "kastner.nii.txt"),
        "--output", str(out_dir),
        "--n-repeats", str(n_repeats),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    status = "pass" if result.returncode == 0 else "fail"
    log_path = ROOT / "logs" / f"decode_roi_random10fold_whitened_sub{subject:02d}.log"
    log_path.write_text(result.stdout + "\n---STDERR---\n" + result.stderr)
    return {"subject": subject, "status": status, "log": str(log_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-repeats", type=int, default=30)
    ap.add_argument("--output-root", default=str(ROOT / "roi_decoding_random10fold_whitened"))
    args = ap.parse_args()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, s, args.n_repeats, args.output_root): s for s in args.subjects}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"  sub-{res['subject']:02d}: {res['status']}", flush=True)

    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"RANDOM10FOLD_WHITENED_WAVE_COMPLETE total={len(results)} failed={n_fail}", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
