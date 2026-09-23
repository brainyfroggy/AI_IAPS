#!/usr/bin/env python3
"""Rebuild Subject 30's BIDS tree with a corrected source file for run 6.

BACKGROUND (2026-08-11): the user identified from their own scan-session
notes that sub-30_ses-01_run-06_bold's approved source
(full_cohort_raw_pipeline_28/manifests/source_selection.tsv, currently
DEV_030/temp/DEV_030_cmrr_mbep2d_bold_1.80_RUN6_20250802172135_16.nii.gz)
is the raw series, which is missing frames (210 volumes vs the expected
218 -- this is exactly the pre-flagged shortfall in cohort_28.json's
flagged_runs for Sub30 run 6). The scanner also produced a MoCo
(motion-corrected) reconstruction of the SAME acquisition,
DEV_030/DEV_030_cmrr_mbep2d_bold_1.80_RUN6_20250802172135_17.nii.gz (NOT
in temp/), which recovered the full 218 volumes.

Independently verified before writing this script (not just trusting the
report):
  - _16.nii.gz (currently selected): shape (118,118,64,210), sha256
    46560823fab96c5465228d197111f75372add2a5988548216b2215ced83e40cb,
    matches source_selection.tsv's recorded hash exactly.
  - _17.nii.gz (proposed): shape (118,118,64,218), size 239095650,
    sha256 e49f3202bf43fae0efafa1c52284b7b61b6b4f67b357bc60f15b7f1074383cb3.
  - _17.json sidecar confirms it genuinely is the scanner's MoCo series:
    SeriesDescription="MoCoSeries", ImageTypeText includes "MOCO",
    ImageComments="Reference volume for motion correction.", SeriesNumber=17.
  - Run 10 (also flagged by the user) and runs 8/9 were spot-checked and
    already have their full, correct volume counts under the EXISTING
    manifest -- only run 6 needs a source-file correction.

IMPORTANT, worth being explicit about: full_cohort_raw_pipeline_28's own
candidate-discovery tooling (cohortlib.classify_source/is_moco, used by
audit_full_cohort_sources.py) deliberately EXCLUDES MoCo-series files from
its automated candidate lists -- build_full_cohort_bids.py's own docstring
says it "refuses ... MoCoSeries ... inputs". That exclusion lives in the
*candidate-discovery* path, which this script does not use (it does not
re-scan raw directories or call classify_source at all); audit_selection()
itself only validates an already-decided manifest row (approval_status,
hash/size identity, asset completeness), so it does not re-apply that
exclusion. This script is a deliberate, informed, single-run override of
that general default, made with direct knowledge of the specific
acquisition (the user's own scan-session notes) -- not a silent bypass.
Surfacing this plainly rather than treating audit_selection() passing as
implicit endorsement of using a MoCo series in general.

WHAT THIS SCRIPT DOES (frozen files never modified):
  1. Fresh-imports cohortlib.py (registered as "cohortlib" in sys.modules,
     since build_full_cohort_bids.py does `from cohortlib import ...`) and
     build_full_cohort_bids.py, both from full_cohort_raw_pipeline_28/code/,
     read-only, in-process -- same adapt-don't-reimplement pattern used by
     wave_orchestrator.py's load_v6_module() for the rest of this project.
  2. Reads the REAL frozen source_selection.tsv via cohortlib.read_selection()
     (never edited on disk).
  3. Overrides ONLY the sub-30_ses-01_run-06_bold row in memory with the
     verified _17 file's path/size/hash, leaving every other row (all other
     subjects, and Sub30's other 9 runs/T1w/fmaps) byte-identical to the
     frozen manifest.
  4. Writes the full corrected row set to a new_pipeline-owned file,
     new_pipeline/config/source_selection_sub30_corrected.tsv, via
     cohortlib.write_tsv() (never hand-transcribed).
  5. Runs audit_selection() (verify_hashes=True) against that corrected
     manifest, scoped to subject 30 only -- fails closed on any error.
  6. Runs build_bids() scoped to subject 30 only, into a fresh staging root
     under new_pipeline/_staging/ (never overwrites the existing shared
     BIDS tree directly; splicing sub-30/ into it is a separate, explicit
     step done by the caller after inspecting this script's output).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

FULL_COHORT_CODE = Path(
    "/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/full_cohort_raw_pipeline_28/code"
)
NEW_PIPELINE_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline")

FROZEN_CONFIG = FULL_COHORT_CODE.parent / "config" / "cohort.json"
FROZEN_SELECTION = FULL_COHORT_CODE.parent / "manifests" / "source_selection.tsv"
CORRECTED_SELECTION = NEW_PIPELINE_ROOT / "config" / "source_selection_sub30_corrected.tsv"
STAGING_OUTPUT = NEW_PIPELINE_ROOT / "_staging" / "sub30_run6_corrected_bids"

TARGET_ASSET_ID = "sub-30_ses-01_run-06_bold"
CORRECTED_FIELDS = {
    "source_relpath": "DEV_030/DEV_030_cmrr_mbep2d_bold_1.80_RUN6_20250802172135_17.nii.gz",
    "source_json_relpath": "DEV_030/DEV_030_cmrr_mbep2d_bold_1.80_RUN6_20250802172135_17.json",
    "source_size_bytes": "239095650",
    "source_sha256": "e49f3202bf43fae0efafa1c52284b7b61b6b4f67b357bc60f15b7f1074383cb3",
    "source_json_size_bytes": "3168",
    "source_json_sha256": "80dab092eedee4ab6069f63492f11979a5c4592920174cdc7a2b1be3bbff2305",
    "notes": (
        "CORRECTED 2026-08-11: original selection (_16, temp/) had 210 volumes "
        "(the pre-flagged shortfall in cohort_28.json flagged_runs). Substituted "
        "the scanner's MoCo reconstruction of the same acquisition (_17, "
        "SeriesNumber 17, NOT in temp/), which has the full 218 volumes. "
        "Identified from the user's own scan-session notes; independently "
        "verified via nibabel shape + sha256 before this substitution. "
        "candidate_count=1; candidates=DEV_030/DEV_030_cmrr_mbep2d_bold_1.80_RUN6_20250802172135_17.nii.gz"
    ),
}


def _fresh_import(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_frozen_modules() -> tuple[ModuleType, ModuleType]:
    # cohortlib MUST be registered under the exact name "cohortlib" before
    # build_full_cohort_bids is imported, since the latter does a plain
    # `from cohortlib import (...)` that Python resolves via sys.modules.
    cohortlib = _fresh_import(FULL_COHORT_CODE / "cohortlib.py", "cohortlib")
    build_bids_mod = _fresh_import(
        FULL_COHORT_CODE / "build_full_cohort_bids.py", "ai_iaps_build_full_cohort_bids"
    )
    return cohortlib, build_bids_mod


def build_corrected_selection(cohortlib: ModuleType) -> Path:
    rows = cohortlib.read_selection(FROZEN_SELECTION)
    matches = [row for row in rows if row["asset_id"] == TARGET_ASSET_ID]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {TARGET_ASSET_ID} row in the frozen manifest; found {len(matches)}"
        )
    target = matches[0]
    before = dict(target)
    target.update(CORRECTED_FIELDS)

    print(f"Overriding {TARGET_ASSET_ID}:")
    for key, new_value in CORRECTED_FIELDS.items():
        print(f"  {key}: {before.get(key)!r} -> {new_value!r}")

    CORRECTED_SELECTION.parent.mkdir(parents=True, exist_ok=True)
    cohortlib.write_tsv(CORRECTED_SELECTION, rows, cohortlib.SELECTION_COLUMNS)
    print(f"Wrote corrected manifest ({len(rows)} rows, all subjects) -> {CORRECTED_SELECTION}")
    return CORRECTED_SELECTION


def main() -> None:
    cohortlib, build_bids_mod = load_frozen_modules()
    selection_path = build_corrected_selection(cohortlib)

    config = cohortlib.load_config(FROZEN_CONFIG)
    raw_root = Path(config["paths"]["raw_root"])
    log_root = Path(config["paths"]["log_root"])
    stimuli_csv = Path(config["paths"]["stimuli_csv"])

    rows, issues = cohortlib.audit_selection(
        config,
        selection_path,
        raw_root,
        log_root,
        stimuli_csv,
        [30],
        require_approved=True,
        check_onsets=True,
        verify_hashes=True,
    )
    errors = [i for i in issues if i.severity == "error"]
    for issue in issues:
        print(f"{issue.severity.upper()} {issue.code} {issue.asset_id}: {issue.message}")
    if errors:
        raise SystemExit(f"FAIL-CLOSED: {len(errors)} audit errors; no BIDS output was created")
    print(f"AUDIT_PASS rows={len(rows)} subjects={sorted(set(int(r['subject']) for r in rows))}")

    if STAGING_OUTPUT.exists():
        raise SystemExit(
            f"Refusing to reuse an existing staging output; remove it first if intentional: {STAGING_OUTPUT}"
        )

    build_bids_mod.build_bids(
        config,
        rows,
        raw_root,
        log_root,
        stimuli_csv,
        STAGING_OUTPUT,
        [30],
        link_niftis=False,
        config_path=FROZEN_CONFIG,
        selection_path=selection_path,
    )
    print(f"BUILD_PASS: {STAGING_OUTPUT}")


if __name__ == "__main__":
    main()
