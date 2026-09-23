#!/usr/bin/env python3
"""Cache SPM LSS single-trial betas (Decoding/GLM_singletrial/betas) into the same
.npz schema stelzer_cache_subject_matrices.py uses for GLMsingle (data, source,
valence, positions_<roi>), so erp_permute_worker.py / erp_wave_launcher.py /
plot_bo_style_charts.py all run unchanged against SPM data just by pointing
--cache-dir here.

Reuses decode_roi_kebostyle_spm.py's beta-loading and Kastner ROI extraction
verbatim (build_stimulus_lookup / make_subject_beta_table / build_kastner_roi_columns
/ load_subject_union_betas / category_to_source_valence) -- that pipeline is already
validated (ran cleanly across n=28/30 subjects for the Ke Bo-style SPM decode). Only
the cache-serialization step here is new.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse

import nibabel as nib
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_kebostyle_spm import (  # noqa: E402
    ROI_ORDER,
    DEFAULT_RUNS,
    build_stimulus_lookup,
    make_subject_beta_table,
    build_kastner_roi_columns,
    load_subject_union_betas,
    category_to_source_valence,
)

DEFAULT_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]


def cache_one(subject: int, beta_dir: Path, label_file: Path, onset_base_dir: Path,
              atlas: Path, atlas_labels: Path, out_dir: Path) -> dict:
    label = f"Sub{subject}"
    out_path = out_dir / f"sub-{subject:02d}.npz"
    if out_path.exists():
        return {"subject": subject, "status": "skipped_existing"}
    try:
        runs = DEFAULT_RUNS
        beta_labels, stim_to_category = build_stimulus_lookup(label_file, onset_base_dir, runs)
        bl = make_subject_beta_table(label, beta_labels, stim_to_category, onset_base_dir, runs)

        first_beta = beta_dir / label / str(bl["beta_file"].iloc[0])
        if not first_beta.exists() and Path(str(first_beta) + ".gz").exists():
            first_beta = Path(str(first_beta) + ".gz")
        ref_img = nib.load(str(first_beta))

        positions, union_flat = build_kastner_roi_columns(atlas, atlas_labels, ref_img)
        data = load_subject_union_betas(label, bl, beta_dir, union_flat)

        source_vals, valence_vals = zip(*(category_to_source_valence(c) for c in bl["category"]))
        if len(bl) != 600:
            raise RuntimeError(f"expected 600 trials, got {len(bl)}")

        payload = {
            "data": data.astype(np.float32),
            "source": np.array(source_vals, dtype="<U10"),
            "valence": np.array(valence_vals, dtype="<U10"),
        }
        for roi in ROI_ORDER:
            if roi not in positions:
                raise RuntimeError(f"{roi}: missing from this subject's atlas positions")
            payload[f"positions_{roi}"] = positions[roi].astype(np.int32)
        np.savez_compressed(out_path, **payload)
        return {"subject": subject, "status": "pass", "n_features": int(data.shape[1])}
    except Exception as exc:  # noqa: BLE001
        return {"subject": subject, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--beta-dir", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/GLM_singletrial/betas"))
    ap.add_argument("--label-file", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/GLM_singletrial/beta_groups.csv"))
    ap.add_argument("--onset-base-dir", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/LAB_IAPS_AI/DataRecording"))
    ap.add_argument("--atlas", type=Path, default=Path("/mnt/c/MRIcroGL/Resources/atlas/kastner.nii.gz"))
    ap.add_argument("--atlas-labels", type=Path, default=Path("/mnt/c/MRIcroGL/Resources/atlas/kastner.nii.txt"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(cache_one, s, args.beta_dir, args.label_file, args.onset_base_dir,
                        args.atlas, args.atlas_labels, args.out_dir): s
            for s in args.subjects
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"  sub-{res['subject']:02d}: {res['status']}"
                  + (f" ({res.get('error')})" if res["status"] == "fail" else ""), flush=True)

    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"SPM_ERP_CACHE_COMPLETE total={len(results)} failed={n_fail}", flush=True)


if __name__ == "__main__":
    main()
