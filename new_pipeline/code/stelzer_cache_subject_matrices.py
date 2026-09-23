#!/usr/bin/env python3
"""Phase A of the Stelzer et al. 2013 / Bo et al. 2021 permutation test replication.

Loads + 8mm-smooths + ROI-extracts each subject's 600 GLMsingle single-trial betas ONCE
and caches the result to a compact .npz, so Phase B (100 shuffles x 17 ROIs x 8 contrasts
per subject) never has to touch the raw NIfTI/HDF5 data again. This is the memory-bound
step (~1.5GB/worker during volume loading), so run with modest concurrency (--workers 4).

Cache contents:
  data        : (600, n_union_features) float32 -- same union-of-ROIs matrix load_glmsingle produces
  positions_* : one int32 array per ROI, column indices into `data`
  source      : (600,) 'natural'/'ai'
  valence     : (600,) 'pleasant'/'neutral'/'unpleasant'
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER, load_glmsingle, validate_manifest  # noqa: E402

DEFAULT_SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31, 33, 34]


def cache_one(subject: int, glmsingle_root: Path, atlas: Path, atlas_labels: Path,
               out_dir: Path, smoothing_mm: float) -> dict:
    label = f"Sub{subject:02d}"
    out_path = out_dir / f"sub-{subject:02d}.npz"
    if out_path.exists():
        return {"subject": subject, "status": "skipped_existing"}
    try:
        input_root = glmsingle_root / f"sub-{subject:02d}"
        data, positions, _counts, manifest, _ref = load_glmsingle(
            input_root, atlas, atlas_labels, smoothing_mm
        )
        manifest = validate_manifest(manifest)
        payload = {
            "data": data.astype(np.float32),
            "source": np.array(manifest["source"].astype(str).str.lower().tolist(), dtype="<U10"),
            "valence": np.array(manifest["valence"].astype(str).str.lower().tolist(), dtype="<U10"),
        }
        for roi in ROI_ORDER:
            payload[f"positions_{roi}"] = positions[roi].astype(np.int32)
        np.savez_compressed(out_path, **payload)
        return {"subject": subject, "status": "pass", "n_features": int(data.shape[1])}
    except Exception as exc:  # noqa: BLE001
        return {"subject": subject, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--glmsingle-root", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/glmsingle"))
    ap.add_argument("--atlas", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/full_cohort_raw_pipeline_28/resources/kastner/kastner.nii.gz"))
    ap.add_argument("--atlas-labels", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/full_cohort_raw_pipeline_28/resources/kastner/kastner.nii.txt"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/stelzer_permutation/cache"))
    ap.add_argument("--smoothing-mm", type=float, default=8.0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(cache_one, s, args.glmsingle_root, args.atlas, args.atlas_labels,
                        args.out_dir, args.smoothing_mm): s
            for s in args.subjects
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"  sub-{res['subject']:02d}: {res['status']}"
                  + (f" ({res.get('error')})" if res["status"] == "fail" else ""), flush=True)

    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"STELZER_CACHE_COMPLETE total={len(results)} failed={n_fail}", flush=True)


if __name__ == "__main__":
    main()
