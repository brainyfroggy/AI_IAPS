#!/usr/bin/env python3
"""Ke Bo-style single-trial decoding (StratifiedKFold(4) x 20 reps, StandardScaler,
LinearSVC -- see kebo_decode_core.py) applied to the lab's SPM LSS single-trial betas
(Decoding/GLM_singletrial/betas), instead of GLMsingle betas.

Beta-loading and Kastner ROI mask logic here mirrors
Decoding/kastner_singletrial_top50_decoding.py's build_stimulus_lookup /
make_subject_beta_table / build_kastner_roi_columns / load_subject_union_betas exactly
(same stimulus-to-category lookup via Sub27's presentation order as the reference, same
17-ROI Kastner/Wang grouping collapsing IPS0-5 into one IPS ROI, excluding FEF/MST/SPL1),
just reshaped into the (data, positions, manifest) interface kebo_decode_core.py expects
so the identical CV/classifier code runs against both beta sources unmodified.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import scipy.io
from nilearn.image import resample_img
from sklearn import __version__ as sklearn_version

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kebo_decode_core import kebo_decode, N_REPEATS, N_FOLDS  # noqa: E402

ROI_ORDER = (
    "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "V3a", "V3b",
    "IPS", "LO1", "LO2", "hMT", "VO1", "VO2", "PHC1", "PHC2",
)
KASTNER_ROI_GROUPS = {
    "V1v": ["V1v"], "V1d": ["V1d"],
    "V2v": ["V2v"], "V2d": ["V2d"],
    "V3v": ["V3v"], "V3d": ["V3d"],
    "hV4": ["hV4"],
    "V3a": ["V3a"], "V3b": ["V3b"],
    "IPS": ["IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5"],
    "LO1": ["LO1"], "LO2": ["LO2"],
    "hMT": ["hMT"],
    "VO1": ["VO1"], "VO2": ["VO2"],
    "PHC1": ["PHC1"], "PHC2": ["PHC2"],
}
CONTRASTS = (
    ("within_natural_pleasant_vs_neutral", "within", "natural", "natural", "pleasant"),
    ("within_ai_pleasant_vs_neutral", "within", "ai", "ai", "pleasant"),
    ("within_natural_unpleasant_vs_neutral", "within", "natural", "natural", "unpleasant"),
    ("within_ai_unpleasant_vs_neutral", "within", "ai", "ai", "unpleasant"),
    ("train_natural_pleasant_vs_neutral_test_ai", "cross", "natural", "ai", "pleasant"),
    ("train_ai_pleasant_vs_neutral_test_natural", "cross", "ai", "natural", "pleasant"),
    ("train_natural_unpleasant_vs_neutral_test_ai", "cross", "natural", "ai", "unpleasant"),
    ("train_ai_unpleasant_vs_neutral_test_natural", "cross", "ai", "natural", "unpleasant"),
)
ALL_CATEGORIES = ("pleasant", "neutral", "unpleasant", "pleasantAI", "neutralAI", "unpleasantAI")
DEFAULT_RUNS = tuple(f"Run{i:02d}" for i in range(1, 11))


def parse_mricrogl_label_file(label_file: Path) -> dict[int, str]:
    labels = {}
    with open(label_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                roi_id = int(float(parts[0]))
            except ValueError:
                continue
            roi_name = parts[1]
            if roi_id > 0 and roi_name != "*.*.*.*.*":
                labels[roi_id] = roi_name
    return labels


def load_stim_names_for_subject(sub: str, runs, onset_base_dir: Path) -> list[str]:
    sub_onset_dir = Path(onset_base_dir) / sub / "LogFiles"
    stim_names = []
    for run_name in runs:
        fpath = sub_onset_dir / f"{run_name}.mat"
        if not fpath.exists():
            alt_name = "Run" + str(int(run_name.replace("Run", "")))
            fpath = sub_onset_dir / f"{alt_name}.mat"
        if not fpath.exists():
            raise FileNotFoundError(f"Cannot find onset file for {sub}, {run_name}: {fpath}")
        mat = scipy.io.loadmat(str(fpath), squeeze_me=False)
        raw = mat["dataLog"]
        for row in range(1, raw.shape[0]):
            c = raw[row, 1]
            if c.size == 0:
                continue
            if str(c.flat[0]).strip() == "Stim on":
                stim_names.append(str(raw[row, 2].flat[0]).strip())
    return stim_names


def build_stimulus_lookup(label_file: Path, onset_base_dir: Path, runs) -> tuple[pd.DataFrame, dict[str, str]]:
    beta_labels = pd.read_csv(label_file, header=None, names=["beta_file", "category"])
    stim_template = load_stim_names_for_subject("Sub27", runs, onset_base_dir)
    if len(stim_template) != len(beta_labels):
        raise ValueError(f"Sub27 stimulus count ({len(stim_template)}) != beta count ({len(beta_labels)})")
    return beta_labels, dict(zip(stim_template, beta_labels["category"].tolist()))


def make_subject_beta_table(sub: str, beta_labels: pd.DataFrame, stim_to_category: dict[str, str], onset_base_dir: Path, runs) -> pd.DataFrame:
    stim_names = load_stim_names_for_subject(sub, runs, onset_base_dir)
    if len(stim_names) != len(beta_labels):
        raise ValueError(f"{sub}: stimulus count ({len(stim_names)}) != beta count ({len(beta_labels)})")
    missing = [s for s in stim_names if s not in stim_to_category]
    if missing:
        raise KeyError(f"{sub}: {len(missing)} stimuli missing from lookup. First few: {missing[:5]}")
    return pd.DataFrame({
        "beta_file": beta_labels["beta_file"].to_numpy(),
        "category": [stim_to_category[s] for s in stim_names],
        "stimulus": stim_names,
    })


def build_kastner_roi_columns(atlas_file: Path, atlas_label_file: Path, ref_img: nib.Nifti1Image) -> tuple[dict[str, np.ndarray], np.ndarray]:
    atlas_img = nib.load(str(atlas_file))
    atlas_data = np.rint(atlas_img.get_fdata()).astype(np.int32)
    labels = parse_mricrogl_label_file(atlas_label_file)
    label_to_id = {name: roi_id for roi_id, name in labels.items()}

    roi_flat: dict[str, np.ndarray] = {}
    for roi, members in KASTNER_ROI_GROUPS.items():
        missing = [m for m in members if m not in label_to_id]
        if missing:
            raise ValueError(f"Missing Kastner labels for {roi}: {missing}")
        member_ids = [label_to_id[m] for m in members]
        mask_data = np.isin(atlas_data, member_ids).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_data, atlas_img.affine)
        resampled = resample_img(
            mask_img, target_affine=ref_img.affine, target_shape=ref_img.shape,
            interpolation="nearest", force_resample=True, copy_header=True,
        )
        flat = np.flatnonzero(resampled.get_fdata().ravel() > 0).astype(np.int64)
        if len(flat) > 0:
            roi_flat[roi] = flat

    union_flat = np.unique(np.concatenate([roi_flat[r] for r in ROI_ORDER if r in roi_flat])).astype(np.int64)
    flat_to_col = {int(flat): i for i, flat in enumerate(union_flat)}
    positions = {
        roi: np.asarray([flat_to_col[int(flat)] for flat in roi_flat[roi]], dtype=np.int32)
        for roi in ROI_ORDER if roi in roi_flat
    }
    return positions, union_flat


def load_subject_union_betas(sub: str, bl: pd.DataFrame, beta_dir: Path, union_flat: np.ndarray) -> np.ndarray:
    sub_dir = Path(beta_dir) / sub
    x = np.empty((len(bl), len(union_flat)), dtype=np.float32)
    for i, beta_file in enumerate(bl["beta_file"].values):
        beta_path = sub_dir / str(beta_file)
        if not beta_path.exists() and Path(str(beta_path) + ".gz").exists():
            beta_path = Path(str(beta_path) + ".gz")
        if not beta_path.exists():
            raise FileNotFoundError(beta_path)
        x[i] = nib.load(str(beta_path)).get_fdata(dtype=np.float32).ravel()[union_flat]
    return x


def category_to_source_valence(category: str) -> tuple[str, str]:
    if category.endswith("AI"):
        return "ai", category[:-2]
    return "natural", category


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--beta-dir", type=Path, default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/GLM_singletrial/betas"))
    parser.add_argument("--label-file", type=Path, default=Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/GLM_singletrial/beta_groups.csv"))
    parser.add_argument("--onset-base-dir", type=Path, default=Path("/mnt/n/Experimental_Data/yujunchen/projects/LAB_IAPS_AI/DataRecording"))
    parser.add_argument("--atlas", type=Path, default=Path("/mnt/c/MRIcroGL/Resources/atlas/kastner.nii.gz"))
    parser.add_argument("--atlas-labels", type=Path, default=Path("/mnt/c/MRIcroGL/Resources/atlas/kastner.nii.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel workers across ROIs")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    runs = DEFAULT_RUNS
    beta_labels, stim_to_category = build_stimulus_lookup(args.label_file, args.onset_base_dir, runs)
    bl = make_subject_beta_table(args.subject, beta_labels, stim_to_category, args.onset_base_dir, runs)

    first_beta = Path(args.beta_dir) / args.subject / str(bl["beta_file"].iloc[0])
    if not first_beta.exists() and Path(str(first_beta) + ".gz").exists():
        first_beta = Path(str(first_beta) + ".gz")
    ref_img = nib.load(str(first_beta))

    positions, union_flat = build_kastner_roi_columns(args.atlas, args.atlas_labels, ref_img)
    data = load_subject_union_betas(args.subject, bl, args.beta_dir, union_flat)

    source_vals, valence_vals = zip(*(category_to_source_valence(c) for c in bl["category"]))
    manifest = pd.DataFrame({"source": source_vals, "valence": valence_vals})
    counts = manifest.value_counts(["source", "valence"]).rename("n").reset_index()

    common = {
        "subject": args.subject,
        "beta_source": "spm_lss",
        "smoothing_location": "unknown (inherited from SPM first-level model)",
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
    counts.assign(**common).to_csv(args.output / "roi_counts.csv", index=False)
    provenance = {
        **common,
        "beta_dir": str(args.beta_dir),
        "label_file": str(args.label_file),
        "atlas": str(args.atlas),
        "atlas_labels": str(args.atlas_labels),
        "reference_shape": list(ref_img.shape[:3]),
        "reference_affine": np.asarray(ref_img.affine).tolist(),
        "n_union_features": int(data.shape[1]),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "nibabel": nib.__version__,
        "scikit_learn": sklearn_version,
        "reference_notebook": "Decoding/decoding_test_on_kebodata.ipynb (Cell 24)",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"KEBOSTYLE_SPM_COMPLETE {args.subject}")


if __name__ == "__main__":
    main()
