#!/usr/bin/env python3
"""Is SPM's systematic BELOW-chance within-AI decoding a real effect or a biased CV null?

The global-amplitude hypothesis was refuted (global_only decodes at ~51% in both
pipelines; centering changes nothing). What remains unexplained is an internal
inconsistency in the SPM results:

  train on AI, test on held-out AI     -> 48.0%  (14/17 ROIs SIGNIFICANTLY below chance)
  train on AI, test on Natural         -> 55.1%

Same training trials, same fitted classifier -- only the test set differs. A classifier
that had learned nothing from AI trials could not then generalise to Natural at 55%. So
the AI patterns evidently do carry valence information, yet held-out AI trials are
systematically MISclassified.

This script computes the empirical permutation null (labels shuffled within the pooled
trials, identical folds/classifier) for the affected contrasts. If the null sits at 50%,
the 48% is a real anti-correlation needing a substantive explanation. If the null itself
sits below 50%, the CV/normalisation is biased for these data and the below-chance
result is an artifact, not a finding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import ROI_ORDER  # noqa: E402
from diagnose_global_vs_pattern import load_glmsingle_pipeline, load_spm_pipeline, fit_acc  # noqa: E402

CHECK_CONTRASTS = [
    ("within_ai_unpleasant_vs_neutral", "ai", "unpleasant"),
    ("within_ai_pleasant_vs_neutral", "ai", "pleasant"),
    ("within_natural_pleasant_vs_neutral", "natural", "pleasant"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=["glmsingle", "spm"], required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--rois", nargs="+", default=["V1v", "hV4", "VO1", "PHC2", "hMT"])
    ap.add_argument("--n-perms", type=int, default=30)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--atlas", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/full_cohort_raw_pipeline_28/resources/kastner/kastner.nii.gz"))
    ap.add_argument("--atlas-labels", type=Path,
                    default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/full_cohort_raw_pipeline_28/resources/kastner/kastner.nii.txt"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    loader = load_glmsingle_pipeline if args.pipeline == "glmsingle" else load_spm_pipeline
    data, positions, manifest = loader(args.subject, args.atlas, args.atlas_labels)
    src = manifest["source"].astype(str).str.lower().to_numpy()
    val = manifest["valence"].astype(str).str.lower().to_numpy()
    rng = np.random.default_rng(args.seed)

    rows = []
    for roi in args.rois:
        if roi not in positions:
            continue
        x = data[:, positions[roi]].astype(np.float64)
        x = x[:, np.all(np.isfinite(x), axis=0)].astype(np.float32)
        if x.shape[1] < 10:
            continue

        for cname, csrc, cpos in CHECK_CONTRASTS:
            idx = np.flatnonzero((src == csrc) & np.isin(val, [cpos, "neutral"]))
            y = (val[idx] == cpos).astype(np.int8)
            xr = x[idx]

            # observed
            obs = []
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                  random_state=int(rng.integers(0, 2**31 - 1)))
            for a, b in skf.split(xr, y):
                obs.append(fit_acc(xr[a], y[a], xr[b], y[b]))
            observed = float(np.nanmean(obs))

            # permutation null: shuffle labels, identical fold machinery
            null = []
            for p in range(args.n_perms):
                yp = rng.permutation(y)
                skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                      random_state=int(rng.integers(0, 2**31 - 1)))
                accs = [fit_acc(xr[a], yp[a], xr[b], yp[b]) for a, b in skf.split(xr, yp)]
                null.append(float(np.nanmean(accs)))

            rows.append({
                "pipeline": args.pipeline, "subject": args.subject, "roi": roi,
                "contrast": cname, "n_voxels": int(x.shape[1]),
                "observed": observed, "null_mean": float(np.mean(null)),
                "null_sd": float(np.std(null, ddof=1)), "n_perms": args.n_perms,
            })
        print(f"  {args.subject} {roi} done", flush=True)

    pd.DataFrame(rows).to_csv(args.output / f"{args.pipeline}_{args.subject}_perm.csv", index=False)
    print(f"PERM_COMPLETE {args.pipeline} {args.subject}")


if __name__ == "__main__":
    main()
