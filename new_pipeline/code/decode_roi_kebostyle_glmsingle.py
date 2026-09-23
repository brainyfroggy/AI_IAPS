#!/usr/bin/env python3
"""Ke Bo-style single-trial decoding (StratifiedKFold(4) x 20 reps, StandardScaler,
LinearSVC -- see kebo_decode_core.py) applied to GLMsingle single-trial betas.

Reuses load_glmsingle/validate_manifest/ROI_ORDER/CONTRASTS unchanged from
decode_roi_singletrial.py, same as every other GLMsingle decode variant this session.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from sklearn import __version__ as sklearn_version

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import (  # noqa: E402
    ROI_ORDER,
    CONTRASTS,
    load_glmsingle,
    validate_manifest,
)
from kebo_decode_core import kebo_decode, N_REPEATS, N_FOLDS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--smoothing-mm", type=float, choices=[0, 3, 5, 8], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--atlas-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel workers across ROIs")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    data, positions, counts, manifest, reference = load_glmsingle(
        args.input_root, args.atlas, args.atlas_labels, args.smoothing_mm
    )
    manifest = validate_manifest(manifest)

    common = {
        "subject": args.subject,
        "beta_source": "glmsingle_typed",
        "branch": args.branch,
        "smoothing_fwhm_mm": args.smoothing_mm,
        "smoothing_location": "post-GLM beta",
        "trial_unit": "single trial",
        "cv": "Ke Bo scheme: StratifiedKFold(4) x 20 reps, StandardScaler(train-only), LinearSVC",
        "classifier": "LinearSVC(max_iter=50000, dual=auto)",
        "normalization": "StandardScaler fit on outer-training trials only",
    }
    results = kebo_decode(
        data, positions, manifest, common, ROI_ORDER, CONTRASTS,
        n_repeats=args.n_repeats, n_folds=args.n_folds, n_jobs=args.n_jobs,
    )
    results.to_csv(args.output / "subject_results.csv", index=False)
    pd.DataFrame(counts).assign(**common).to_csv(args.output / "roi_counts.csv", index=False)
    provenance = {
        **common,
        "input_root": str(args.input_root),
        "atlas": str(args.atlas),
        "atlas_labels": str(args.atlas_labels),
        "reference_shape": list(reference.shape[:3]),
        "reference_affine": np.asarray(reference.affine).tolist(),
        "n_union_features": int(data.shape[1]),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "nibabel": nib.__version__,
        "scikit_learn": sklearn_version,
        "reference_notebook": "Decoding/decoding_test_on_kebodata.ipynb (Cell 24)",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"KEBOSTYLE_GLMSINGLE_COMPLETE {args.subject}")


if __name__ == "__main__":
    main()
