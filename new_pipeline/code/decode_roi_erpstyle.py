#!/usr/bin/env python3
"""ERP-style (averaged-trial) decoding in the 17 Kastner/Wang ROIs, applied to GLMsingle
single-trial betas instead of the original SPM LSS betas.

This is a same-data ablation of the lab's earlier "voxel_trial_source_zscore" Avg(random)
decoding (see Decoding/decoding_multisub_parallel_avg_voxel_trial_source_zscore_AAL3_all.ipynb
and Decoding/PROJECT_KNOWLEDGE_DECODING.md), reimplemented verbatim against the current
GLMsingle Type-D / fMRIPrep pipeline's single-trial betas. Everything about the decoding
mechanics (z-scoring order, fold scheme, trial-averaging, classifier) is copied from that
notebook's decode_within_avg_voxel_trial_source / decode_cross_avg_voxel_trial_source /
compute_voxel_source_zscore / preprocess_voxel_trial_inside_fold functions; only the beta
source and ROI-loading plumbing (load_glmsingle, 8mm post-GLM mask-normalized smoothing) are
swapped for the new_pipeline's. This isolates how much of the ERP-style vs. random-10-fold
accuracy gap is attributable to trial-averaging + preprocessing order alone, vs. the
SPM-vs-GLMsingle estimation pipeline.

CV design (unchanged from the original notebook):
  - voxel-source z-score: computed ONCE per subject per ROI per source (natural / AI),
    pooling that source's 300 trials (100 pleasant + 100 neutral + 100 unpleasant): pattern
    z-score each trial across voxels, then z-score each voxel across the pooled trials.
    Unsupervised (no valence label used), so this is not leakage across the binary contrasts
    that reuse it, but it does pool across the train/test split used later.
  - within-source: KFold(4, shuffle=True), 20 repeats. Inside each fold: pattern z-score
    again, then per-voxel z-score fit on train only. Training trials are averaged into 3
    random-chunk patterns per condition; the held-out test trials are averaged into ONE
    pattern per condition. Linear SVC trained on the 6 averaged training patterns, scored on
    the 2 averaged test patterns.
  - cross-source: no held-out fold (the other source IS the test set). Per repeat: same
    z-scoring, then both train-source and test-source trials are split into 4 random chunks
    per condition and averaged; SVC trained on the 8 train-source chunk-patterns, scored on
    the 8 test-source chunk-patterns.

Reuses load_glmsingle/validate_manifest/ROI_ORDER/CONTRASTS unchanged from
decode_roi_singletrial.py.
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
from scipy.stats import zscore
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import KFold
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_roi_singletrial import (  # noqa: E402
    ROI_ORDER,
    CONTRASTS,
    load_glmsingle,
    validate_manifest,
)

N_REPEATS = 20
N_FOLDS = 4
N_AVG_GROUPS = 3

NATURAL_CATS = ("pleasant", "neutral", "unpleasant")
AI_CATS = ("pleasantAI", "neutralAI", "unpleasantAI")
ALL_CATS = NATURAL_CATS + AI_CATS


def safe_pattern_zscore(x: np.ndarray) -> np.ndarray:
    xz = zscore(x, axis=1)
    return np.nan_to_num(xz, nan=0.0, posinf=0.0, neginf=0.0)


def fit_trial_zscore(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std == 0] = 1
    return mean, std


def apply_trial_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def compute_voxel_source_zscore(cond_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pattern norm, then voxel-wise z-score pooled within each source (natural / AI)."""
    result: dict[str, np.ndarray] = {}
    for source_cats in (NATURAL_CATS, AI_CATS):
        present = [c for c in source_cats if c in cond_data]
        if not present:
            continue
        all_data = np.vstack([cond_data[c] for c in present])
        sizes = [len(cond_data[c]) for c in present]
        all_z = safe_pattern_zscore(all_data)
        m = np.mean(all_z, axis=0)
        s = np.std(all_z, axis=0)
        s[s == 0] = 1
        all_z = (all_z - m) / s
        idx = 0
        for c, sz in zip(present, sizes):
            result[c] = all_z[idx:idx + sz]
            idx += sz
    return result


def preprocess_voxel_trial_inside_fold(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_train = safe_pattern_zscore(x_train.copy())
    x_test = safe_pattern_zscore(x_test.copy())
    mean, std = fit_trial_zscore(x_train)
    return apply_trial_zscore(x_train, mean, std), apply_trial_zscore(x_test, mean, std)


def average_chunks(x: np.ndarray, chunks: list[np.ndarray]) -> np.ndarray:
    return np.array([np.mean(x[ch], axis=0) for ch in chunks])


def decode_within_avg(
    d1: np.ndarray, d2: np.ndarray,
    n_repeats: int = N_REPEATS, n_folds: int = N_FOLDS, n_avg_groups: int = N_AVG_GROUPS,
) -> float:
    accuracies = []
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42 + repeat)
        indices = np.arange(d1.shape[0])
        for train_idx, test_idx in kf.split(indices):
            train_all = np.vstack([d1[train_idx], d2[train_idx]])
            test_all = np.vstack([d1[test_idx], d2[test_idx]])
            train_all, test_all = preprocess_voxel_trial_inside_fold(train_all, test_all)

            n_train_1 = len(train_idx)
            n_test_1 = len(test_idx)
            d1_train_z, d2_train_z = train_all[:n_train_1], train_all[n_train_1:]
            d1_test_z, d2_test_z = test_all[:n_test_1], test_all[n_test_1:]

            train_chunks = np.array_split(np.arange(n_train_1), n_avg_groups)
            train_d1_avg = average_chunks(d1_train_z, train_chunks)
            train_d2_avg = average_chunks(d2_train_z, train_chunks)
            test_d1_avg = np.mean(d1_test_z, axis=0, keepdims=True)
            test_d2_avg = np.mean(d2_test_z, axis=0, keepdims=True)

            x_train = np.vstack([train_d1_avg, train_d2_avg])
            y_train = np.array([1] * len(train_d1_avg) + [0] * len(train_d2_avg))
            x_test = np.vstack([test_d1_avg, test_d2_avg])
            y_test = np.array([1, 0])

            clf = SVC(kernel="linear", C=1.0)
            clf.fit(x_train, y_train)
            accuracies.append(clf.score(x_test, y_test))
    return float(np.mean(accuracies))


def decode_cross_avg(
    train_d1: np.ndarray, train_d2: np.ndarray, test_d1: np.ndarray, test_d2: np.ndarray,
    n_repeats: int = N_REPEATS, n_folds: int = N_FOLDS,
) -> float:
    accuracies = []
    for repeat in range(n_repeats):
        train_all = np.vstack([train_d1, train_d2])
        test_all = np.vstack([test_d1, test_d2])
        train_all, test_all = preprocess_voxel_trial_inside_fold(train_all, test_all)

        n_tr1 = len(train_d1)
        n_te1 = len(test_d1)

        shuffle_idx = np.random.RandomState(42 + repeat).permutation(n_tr1)
        groups = np.array_split(shuffle_idx, n_folds)

        td1a = average_chunks(train_all[:n_tr1], groups)
        td2a = average_chunks(train_all[n_tr1:], groups)
        te1a = average_chunks(test_all[:n_te1], groups)
        te2a = average_chunks(test_all[n_te1:], groups)

        x_train = np.vstack([td1a, td2a])
        y_train = np.array([1] * len(td1a) + [0] * len(td2a))
        x_test = np.vstack([te1a, te2a])
        y_test = np.array([1] * len(te1a) + [0] * len(te2a))

        clf = SVC(kernel="linear", C=1.0)
        clf.fit(x_train, y_train)
        accuracies.append(clf.score(x_test, y_test))
    return float(np.mean(accuracies))


def condition_label(source: str, valence: str) -> str:
    if valence == "neutral":
        return "neutral" if source == "natural" else "neutralAI"
    return valence if source == "natural" else f"{valence}AI"


def decode_erpstyle(
    data: np.ndarray,
    positions: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    common: dict,
    n_repeats: int = N_REPEATS,
    n_folds: int = N_FOLDS,
    n_avg_groups: int = N_AVG_GROUPS,
) -> pd.DataFrame:
    source_values = manifest["source"].astype(str).str.lower().to_numpy()
    valence_values = manifest["valence"].astype(str).str.lower().to_numpy()

    rows: list[dict] = []

    for roi in ROI_ORDER:
        roi_data = data[:, positions[roi]]
        finite_global = np.all(np.isfinite(roi_data), axis=0)
        roi_data = roi_data[:, finite_global]
        if roi_data.shape[1] < 10:
            raise RuntimeError(f"{roi}: fewer than ten finite features")

        cond_data: dict[str, np.ndarray] = {}
        for source in ("natural", "ai"):
            for valence in ("pleasant", "neutral", "unpleasant"):
                idx = np.flatnonzero((source_values == source) & (valence_values == valence))
                if len(idx) != 100:
                    raise RuntimeError(f"{roi}/{source}/{valence}: expected 100 trials, got {len(idx)}")
                cond_data[condition_label(source, valence)] = roi_data[idx]

        voxel_source_z = compute_voxel_source_zscore(cond_data)

        for name, kind, train_source, test_source, positive in CONTRASTS:
            if kind == "within":
                d1 = voxel_source_z[condition_label(train_source, positive)]
                d2 = voxel_source_z[condition_label(train_source, "neutral")]
                acc = decode_within_avg(d1, d2, n_repeats, n_folds, n_avg_groups)
            else:
                train_d1 = voxel_source_z[condition_label(train_source, positive)]
                train_d2 = voxel_source_z[condition_label(train_source, "neutral")]
                test_d1 = voxel_source_z[condition_label(test_source, positive)]
                test_d2 = voxel_source_z[condition_label(test_source, "neutral")]
                acc = decode_cross_avg(train_d1, train_d2, test_d1, test_d2, n_repeats, n_folds)

            rows.append({
                **common, "roi": roi, "contrast": name, "contrast_kind": kind,
                "train_source": train_source, "test_source": test_source, "positive_valence": positive,
                "n_repeats": n_repeats, "n_folds": n_folds, "n_avg_groups": n_avg_groups,
                "accuracy": acc,
            })

    return pd.DataFrame(rows)


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
    parser.add_argument("--n-avg-groups", type=int, default=N_AVG_GROUPS)
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
        "estimator": "glmsingle_typed",
        "branch": args.branch,
        "smoothing_fwhm_mm": args.smoothing_mm,
        "smoothing_location": "post-GLM beta",
        "trial_unit": "averaged-trial (ERP-style, Avg(random))",
        "cv": "voxel_trial_source_zscore Avg(random): KFold(4) x 20 repeats, 3-group train averaging",
        "classifier": "linear SVC C=1",
        "normalization": "voxel-source z-score once per ROI, then pattern+trial z-score inside each fold",
    }
    results = decode_erpstyle(
        data, positions, manifest, common,
        n_repeats=args.n_repeats, n_folds=args.n_folds, n_avg_groups=args.n_avg_groups,
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
        "reference_notebook": "Decoding/decoding_multisub_parallel_avg_voxel_trial_source_zscore_AAL3_all.ipynb",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"ERPSTYLE_COMPLETE {args.subject}")


if __name__ == "__main__":
    main()
