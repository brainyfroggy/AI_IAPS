#!/usr/bin/env python3
"""Univariate check: do unpleasant trials evoke weaker neural responses than
pleasant trials?

Motivation: unpleasant-vs-neutral decoding is consistently weaker than
pleasant-vs-neutral (within-source AND cross-source, Kastner AND AAL3). One
candidate explanation is simply lower evoked response / SNR for unpleasant
images. This script tests that univariately, independent of any classifier.

Design (verified): 120 unique images x 5 repetitions = 600 trials.
60 images per source (natural / AI), 20 per valence within each source.
All 30 subjects saw the identical image set (image_id is stable across
subjects), so image-level averaging across subjects is meaningful.

Aggregation, matching the requested figure:
  1. per trial   -> mean GLMsingle Type-D beta across the ROI set's voxels
  2. per subject -> average the 5 repetitions of each image  (600 -> 120)
  3. per image   -> mean across the 30 subjects, SEM across the 30 subjects

So error bars are BETWEEN-SUBJECT SEM (n=30), with repetition variance already
averaged out inside each subject.

Two ROI sets are computed from the existing 8mm caches:
  visual_kastner  - union of the 17 Kastner/Wang visual ROIs (where decoding
                    was strongest; the relevant region for image stimuli)
  wholebrain_aal3 - all 154 kept AAL3 regions (whole-brain gray matter)

Note on interpretation: decoding accuracy depends on pleasant/unpleasant vs
neutral *pattern separability*, not purely on response magnitude. A magnitude
difference is suggestive (lower SNR -> lower decodability), not proof of the
mechanism -- this is a diagnostic, not a decisive test.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

NEW_PIPELINE = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")
DEFAULT_SUBJECTS = [1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31, 33, 34]

ROI_SETS = {
    "visual_kastner": NEW_PIPELINE / "stelzer_permutation" / "cache",
    "wholebrain_aal3": NEW_PIPELINE / "aal3_crosssource" / "cache_smooth08",
}


def process_subject(subject: int, cache_path: Path, glmsingle_root: Path,
                    roi: str | None = None) -> dict:
    """Return image-level mean betas for one subject.

    roi=None averages over every voxel in the cache (the whole ROI set);
    passing a name restricts to that single ROI via its cached position index.
    """
    try:
        npz = np.load(cache_path)
        data = npz["data"]  # (600, n_voxels), row i == beta_index i
        if data.shape[0] != 600:
            raise RuntimeError(f"expected 600 trials, got {data.shape[0]}")
        if roi is not None:
            key = f"positions_{roi}"
            if key not in npz.files:
                raise RuntimeError(f"cache has no {key}")
            data = data[:, npz[key]]

        # per-trial mean beta across the ROI set's voxels
        trial_mean = data.mean(axis=1, dtype=np.float64)

        # each subject has its own randomized trial order -> use its own manifest
        manifest = pd.read_csv(glmsingle_root / f"sub-{subject:02d}" / "trial_manifest.tsv", sep="\t")
        manifest = manifest.sort_values("beta_index").reset_index(drop=True)
        if manifest["beta_index"].tolist() != list(range(600)):
            raise RuntimeError("manifest beta_index is not exactly 0..599")

        frame = pd.DataFrame({
            "image_id": manifest["image_id"].to_numpy(),
            "source": manifest["source"].astype(str).str.lower().to_numpy(),
            "valence": manifest["valence"].astype(str).str.lower().to_numpy(),
            "beta": trial_mean,
        })
        # step 2: average the 5 repetitions of each image within this subject
        img = frame.groupby(["image_id", "source", "valence"], as_index=False).agg(
            image_beta=("beta", "mean"),
            n_reps=("beta", "size"),
        )
        if not (img["n_reps"] == 5).all():
            raise RuntimeError("not every image has exactly 5 repetitions")
        if len(img) != 120:
            raise RuntimeError(f"expected 120 unique images, got {len(img)}")
        img.insert(0, "subject", f"Sub{subject:02d}")
        return {"subject": subject, "status": "pass", "frame": img}
    except Exception as exc:  # noqa: BLE001
        return {"subject": subject, "status": "fail", "error": f"{type(exc).__name__}: {exc}",
                "frame": pd.DataFrame()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--glmsingle-root", type=Path, default=NEW_PIPELINE / "glmsingle")
    ap.add_argument("--out-dir", type=Path, default=NEW_PIPELINE / "trial_beta_magnitude")
    ap.add_argument("--roi-sets", nargs="+", default=list(ROI_SETS))
    ap.add_argument("--roi", default=None,
                    help="restrict to a single named ROI within the cache (e.g. hMT)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for roi_set in args.roi_sets:
        cache_dir = ROI_SETS[roi_set]
        prefix = roi_set if args.roi is None else f"{roi_set}_{args.roi}"
        print(f"\n=== {prefix} (cache: {cache_dir}) ===", flush=True)
        missing = [s for s in args.subjects if not (cache_dir / f"sub-{s:02d}.npz").exists()]
        if missing:
            raise RuntimeError(f"{roi_set}: missing cache for subjects {missing}")

        frames = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_subject, s, cache_dir / f"sub-{s:02d}.npz",
                            args.glmsingle_root, args.roi): s
                for s in args.subjects
            }
            for fut in as_completed(futures):
                res = fut.result()
                if res["status"] == "fail":
                    raise RuntimeError(f"sub-{res['subject']:02d}: {res['error']}")
                frames.append(res["frame"])
                print(f"  sub-{res['subject']:02d}: pass", flush=True)

        subject_level = pd.concat(frames, ignore_index=True)
        subject_level.to_csv(args.out_dir / f"{prefix}_subject_image_betas.csv", index=False)

        # step 3: across-subject mean and SEM per image
        n_sub = subject_level["subject"].nunique()
        image_level = subject_level.groupby(["image_id", "source", "valence"], as_index=False).agg(
            mean_beta=("image_beta", "mean"),
            sd_beta=("image_beta", lambda x: x.std(ddof=1)),
            n_subjects=("image_beta", "size"),
        )
        image_level["sem_beta"] = image_level["sd_beta"] / np.sqrt(image_level["n_subjects"])
        if not (image_level["n_subjects"] == n_sub).all():
            raise RuntimeError("uneven subject count per image")
        image_level.to_csv(args.out_dir / f"{prefix}_image_level_betas.csv", index=False)

        # cell summary + paired tests across subjects
        from scipy import stats
        cell = subject_level.groupby(["subject", "source", "valence"], as_index=False)["image_beta"].mean()
        summary_rows = []
        for source in ["natural", "ai"]:
            piv = cell[cell["source"] == source].pivot(index="subject", columns="valence",
                                                        values="image_beta")
            for val in ["pleasant", "neutral", "unpleasant"]:
                summary_rows.append({
                    "source": source, "valence": val,
                    "mean_beta": float(piv[val].mean()),
                    "sem_beta": float(piv[val].std(ddof=1) / np.sqrt(len(piv))),
                })
            for a, b in [("pleasant", "neutral"), ("unpleasant", "neutral"), ("pleasant", "unpleasant")]:
                t, p = stats.ttest_rel(piv[a], piv[b])
                print(f"  [{source}] {a} vs {b}: mean diff = "
                      f"{(piv[a]-piv[b]).mean():+.4f}, t={t:.3f}, p={p:.4g}", flush=True)
        pd.DataFrame(summary_rows).to_csv(args.out_dir / f"{prefix}_cell_summary.csv", index=False)
        print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)

    print("\nTRIAL_BETA_MAGNITUDE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
