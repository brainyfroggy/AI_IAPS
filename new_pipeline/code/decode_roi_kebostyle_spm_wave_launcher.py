#!/usr/bin/env python3
"""Run decode_roi_kebostyle_spm.py across the full SPM cohort with bounded parallelism.
Mirrors decode_roi_kebostyle_glmsingle_wave_launcher.py; subject labels are "SubN" (no
zero-padding) matching the SPM beta directory naming, per Decoding/PROJECT_KNOWLEDGE_DECODING.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")
PYTHON = "/home/yujun/.cache/ai_iaps_pilot_venv/bin/python3"
SCRIPT = ROOT / "code" / "decode_roi_kebostyle_spm.py"

DEFAULT_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18,
                    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]


def run_one(subject: int, inner_jobs: int, out_root: str) -> dict:
    label = f"Sub{subject}"
    out_dir = Path(out_root) / f"sub-{subject:02d}"
    if (out_dir / "subject_results.csv").exists():
        return {"subject": subject, "status": "skipped_existing"}
    cmd = [
        PYTHON, str(SCRIPT),
        "--subject", label,
        "--output", str(out_dir),
        "--n-jobs", str(inner_jobs),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    status = "pass" if result.returncode == 0 else "fail"
    log_path = ROOT / "logs" / f"decode_roi_kebostyle_spm_sub{subject:02d}.log"
    log_path.write_text(result.stdout + "\n---STDERR---\n" + result.stderr)
    return {"subject": subject, "status": status, "log": str(log_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--workers", type=int, default=2, help="subjects processed concurrently")
    ap.add_argument("--inner-jobs", type=int, default=10, help="ROI x contrast workers per subject")
    ap.add_argument("--output-root", default=str(ROOT / "roi_decoding_kebostyle_spm"))
    args = ap.parse_args()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, s, args.inner_jobs, args.output_root): s for s in args.subjects}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"  sub-{res['subject']:02d}: {res['status']}", flush=True)

    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"KEBOSTYLE_SPM_WAVE_COMPLETE total={len(results)} failed={n_fail}", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
