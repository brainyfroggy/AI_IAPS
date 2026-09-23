#!/usr/bin/env python3
"""Pilot: rerun GLMsingle itself with wantlibrary=0 (ONE fixed/default HRF applied
to every voxel, no per-voxel library search) instead of wantlibrary=1 (GLMsingle's
normal per-voxel HRF selection from a ~20-shape library) -- everything else held
identical to the already-computed TYPEB_FITHRF.hdf5 (wantglmdenoise=0,
wantfracridge=0, wantlss=0, same nuisance regressors, same design, same data).

This is the direct, controlled test of the hypothesis raised comparing SPM (fixed
canonical HRF) against GLMsingle (per-voxel-fitted HRF): does turning OFF
GLMsingle's per-voxel HRF search -- forcing one fixed HRF everywhere, structurally
closer to what SPM does -- change ERP decoding accuracy relative to TYPEB
(wantlibrary=1, otherwise identical settings)?

Reuses run_glmsingle_wave.py's validated subject-generalization plumbing
(SESSION_INDICATORS/EXPECTED_RUN_N_VOLUMES derivation, branch-aware
discover_run_files patch) and run_glmsingle.py's data-prep helpers
(build_design, load_masked_data, intersection_mask, validate_run_n_volumes) via
fresh in-process imports -- neither frozen script is edited on disk. Only the
`params` dict passed to GLM_single differs from the frozen run().

Writes to a new glmsingle_fixedhrf_pilot/ tree, never touching the production
glmsingle/ output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import nibabel as nib
import numpy as np
import pandas as pd

NEW_PIPELINE_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")
PROJECT_AI_IAPS_ROOT = NEW_PIPELINE_ROOT.parent
DEFAULT_COHORT_CONFIG = NEW_PIPELINE_ROOT / "config" / "cohort_28.json"
PILOT_RUNNER = PROJECT_AI_IAPS_ROOT / "Decoding" / "pilot_raw_pipeline_sub4_6" / "code" / "run_glmsingle.py"
GLMSINGLE_SPACE_SCRIPT = PROJECT_AI_IAPS_ROOT / "Decoding" / "sub4_spatial_sensitivity_32" / "code" / "run_glmsingle_space.py"

sys.path.insert(0, str(NEW_PIPELINE_ROOT / "code"))
from run_glmsingle_wave import (  # noqa: E402
    load_cohort, derive_all_subjects, build_session_indicators_dict,
    build_expected_run_n_volumes_dict, patch_subject_dirs_dedup, _fresh_import,
)

SEED = 20260728
STIMDUR = 3.0
TR = 1.80
DEFAULT_FRACS = np.arange(1.0, 0.0, -0.05, dtype=np.float32)


def load_pilot_generalized(cohort: dict) -> ModuleType:
    pilot = _fresh_import(PILOT_RUNNER, "ai_iaps_pilot_glmsingle_fixedhrf")
    pilot.SESSION_INDICATORS = build_session_indicators_dict(cohort)
    pilot.EXPECTED_RUN_N_VOLUMES = build_expected_run_n_volumes_dict(cohort)
    patch_subject_dirs_dedup(pilot)
    return pilot


def run_fixedhrf(subject: int, pilot: ModuleType, space: ModuleType, branch: str,
                  fmriprep_root: Path, bids_root: Path, output: Path, n_pcs: int,
                  want_glmdenoise: int = 0, want_fracridge: int = 0, want_library: int = 0) -> None:
    np.random.seed(SEED)

    def replacement(root: Path, subj: int, _space: str):
        return space.discover_branch(root, subj, branch)

    pilot.discover_run_files = replacement

    records = pilot.discover_run_files(fmriprep_root, subject, branch)
    run_n_volumes = pilot.validate_run_n_volumes(records, subject)
    mask, reference = pilot.intersection_mask(records)
    flat_indices = np.flatnonzero(mask.ravel())

    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    scratch = output / "scratch"
    scratch.mkdir()

    mask_header = reference.header.copy()
    mask_header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, mask_header),
              output / "analysis_mask.nii.gz")
    np.save(output / "flat_mask_indices.npy", flat_indices)

    designs, trial_manifest, start_times = pilot.build_design(records, bids_root, subject)
    trial_manifest.to_csv(output / "trial_manifest.tsv", sep="\t", index=False)
    data, nuisance, nuisance_names, motion_qc = pilot.load_masked_data(records, flat_indices)
    pd.DataFrame(motion_qc).to_csv(output / "motion_qc.tsv", sep="\t", index=False)
    (output / "nuisance_columns.json").write_text(json.dumps(nuisance_names, indent=2) + "\n")

    # Output-slot selection matches GLMsingle's own wantfileoutputs convention
    # [TYPEA, TYPEB, TYPEC, TYPED]. IMPORTANT: the TYPEB slot (index 1) must
    # always be requested here -- GLMsingle's own vendored code has a bug
    # (glmsingle.py ~line 994, `HRFindex = np.ones(xyz)` with `xyz` undefined)
    # that only triggers when wantfileoutputs[1]==0 AND wantmemoryoutputs[1]==0
    # AND the HRF library has been reduced to size 1 (exactly what wantlibrary=0
    # does) -- a "short-circuit" code path GLMsingle's authors apparently never
    # exercised with library size 1. Requesting slot 1 avoids that path entirely
    # (confirmed: the wantlibrary=0/no-denoise/no-ridge pilot, which requested
    # slot 1, ran cleanly; the wantlibrary=0/+denoise/+ridge pilot, which
    # requested only slot 3, hit this exact crash).
    #
    # Slot D (index 3) is the RIDGE slot regardless of whether denoise is also
    # on: per GLMsingle's own docstring, "wantglmdenoise=0, wantfracridge=1"
    # still computes model type D, just "using 0 GLMdenoise regressors" -- so
    # ridge-only must also request slot 3, not fall through to slot B.
    if want_fracridge:
        wantfileoutputs = [0, 1, 0, 1]
        slot_name = "TYPED"
        slot_note = ("full Type-D treatment (denoise+ridge) but wantlibrary=0" if want_glmdenoise
                     else "ridge only (0 GLMdenoise regressors), wantlibrary=0")
    elif want_glmdenoise:
        wantfileoutputs = [0, 1, 1, 0]
        slot_name, slot_note = "TYPEC", "denoise only (no ridge), wantlibrary=0"
    else:
        wantfileoutputs = [0, 1, 0, 0]
        slot_name, slot_note = "TYPEB", "no denoise, no ridge, wantlibrary=0"

    params = {
        "wantlibrary": want_library,  # 0 = one fixed HRF for every voxel; 1 = per-voxel library search
        "wantglmdenoise": want_glmdenoise,
        "wantfracridge": want_fracridge,
        "wantlss": 0,
        "n_pcs": n_pcs,
        "pcstop": 1.05,
        "fracs": DEFAULT_FRACS,
        "wantautoscale": 1,
        "wantpercentbold": 1,
        "xvalscheme": np.arange(10, dtype=np.int64),
        "sessionindicator": np.asarray(pilot.SESSION_INDICATORS[subject], dtype=np.int64),
        "extra_regressors": nuisance,
        "maxpolydeg": [3] * 10,
        "chunklen": 10000,
        "wantfileoutputs": wantfileoutputs,
        "wantmemoryoutputs": [0, 0, 0, 0],
        "wanthdf5": 1,
        "brainexclude": False,
        "pcR2cutoffmask": 1,
        "seed": SEED,
    }
    provenance = {
        "subject": subject, "space": branch, "n_voxels": int(len(flat_indices)),
        "run_n_volumes": run_n_volumes,
        "wantlibrary": want_library, "wantglmdenoise": want_glmdenoise, "wantfracridge": want_fracridge,
        "output_slot": slot_name,
        "note": f"{slot_name}-equivalent slot, wantlibrary={want_library}: {slot_note}.",
        "vendor_patch_relevant": not want_glmdenoise,
        "vendor_patch": (
            "VENDOR_ROOT/glmsingle/glmsingle.py locally patched: when "
            "wantglmdenoise=0, pcregressors is now [None]*numruns instead of "
            "a bare [] placeholder. GLMsingle's own TYPE-D/ridge fitting path "
            "indexes pcregressors[run_i] unconditionally regardless of "
            "wantglmdenoise, which crashed (IndexError) specifically for the "
            "documented-supported wantglmdenoise=0, wantfracridge=1 "
            "combination. Safe: pcnum=0 in this branch, so "
            "_merge_extra_regressors's n_pc<=0 path never reads the "
            "placeholder values, only requires the list be indexable."
        ),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    previous = Path.cwd()
    os.chdir(scratch)
    try:
        pilot.GLM_single(params).fit(
            designs, data, STIMDUR, TR,
            outputdir=str(output / "glmsingle"), figuredir=None,
        )
    finally:
        os.chdir(previous)
    print(f"GLMSINGLE_FIXEDHRF_COMPLETE subject={subject}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--cohort-config", type=Path, default=DEFAULT_COHORT_CONFIG)
    ap.add_argument("--fmriprep-derivatives-root", type=Path, required=True,
                    help="e.g. LAB_IAPS_AI/fmriprep_derivatives/unified_glmsingle_28 (per-subject SubNN/ appended)")
    ap.add_argument("--per-subject-bids-root", type=Path, required=True,
                    help="e.g. LAB_IAPS_AI/bids_unified_glmsingle_28_per_subject (per-subject SubNN/ appended)")
    ap.add_argument("--branch", default="mni_res_native")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--n-pcs", type=int, default=10)
    ap.add_argument("--want-glmdenoise", type=int, default=0, choices=[0, 1])
    ap.add_argument("--want-fracridge", type=int, default=0, choices=[0, 1])
    ap.add_argument("--want-library", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    cohort = load_cohort(args.cohort_config)
    pilot = load_pilot_generalized(cohort)
    space = _fresh_import(GLMSINGLE_SPACE_SCRIPT, "ai_iaps_glmsingle_space_fixedhrf")

    for subject in args.subjects:
        if subject not in derive_all_subjects(cohort):
            raise ValueError(f"Subject {subject} not in {args.cohort_config}'s cohort.remaining_subjects")
        label = f"Sub{subject:02d}"
        fmriprep_root = args.fmriprep_derivatives_root / label
        bids_root = args.per_subject_bids_root / label
        output = args.output_root / f"sub-{subject:02d}"
        run_fixedhrf(subject, pilot, space, args.branch, fmriprep_root, bids_root, output, args.n_pcs,
                     args.want_glmdenoise, args.want_fracridge, args.want_library)


if __name__ == "__main__":
    main()
