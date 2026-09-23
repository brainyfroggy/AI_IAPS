#!/usr/bin/env python3
"""Phase 1 of the whole-brain AAL3 cross-source decoding plan.

Loads + smooths (0mm or 8mm) + AAL3-extracts each subject's 600 GLMsingle
single-trial betas ONCE and caches the result to a compact .npz, so Phase 2
decode never touches raw HDF5/NIfTI data again. Mirrors
stelzer_cache_subject_matrices.py's pattern, generalized to the frozen
154-region AAL3 manifest via aal3_common.py instead of the Kastner atlas.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import validate_manifest  # noqa: E402
from aal3_common import load_aal3_kept_regions, load_aal3_glmsingle  # noqa: E402

DEFAULT_SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31, 33, 34]


def cache_one(subject: int, glmsingle_root: Path, atlas_native_labels: Path,
              kept_regions: list, out_dir: Path, smoothing_mm: float) -> dict:
    out_path = out_dir / f"sub-{subject:02d}.npz"
    if out_path.exists():
        return {"subject": subject, "status": "skipped_existing"}
    try:
        input_root = glmsingle_root / f"sub-{subject:02d}"
        data, positions, _counts, manifest, _ref = load_aal3_glmsingle(
            input_root, atlas_native_labels, kept_regions, smoothing_mm
        )
        manifest = validate_manifest(manifest)
        payload = {
            "data": data.astype(np.float32),
            "source": np.array(manifest["source"].astype(str).str.lower().tolist(), dtype="<U10"),
            "valence": np.array(manifest["valence"].astype(str).str.lower().tolist(), dtype="<U10"),
        }
        for _, name in kept_regions:
            payload[f"positions_{name}"] = positions[name].astype(np.int32)
        np.savez_compressed(out_path, **payload)
        return {"subject": subject, "status": "pass", "n_features": int(data.shape[1])}
    except Exception as exc:  # noqa: BLE001
        return {"subject": subject, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--glmsingle-root", type=Path, required=True)
    ap.add_argument("--atlas-native-labels", type=Path, required=True)
    ap.add_argument("--roi-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--smoothing-mm", type=float, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    kept_regions = load_aal3_kept_regions(args.roi_manifest)
    print(f"kept regions: {len(kept_regions)}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(cache_one, s, args.glmsingle_root, args.atlas_native_labels,
                        kept_regions, args.out_dir, args.smoothing_mm): s
            for s in args.subjects
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"  sub-{res['subject']:02d}: {res['status']}"
                  + (f" ({res.get('error')})" if res["status"] == "fail" else ""), flush=True)

    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"AAL3_CACHE_COMPLETE smoothing={args.smoothing_mm} total={len(results)} failed={n_fail}", flush=True)


if __name__ == "__main__":
    main()
