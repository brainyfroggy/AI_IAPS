#!/usr/bin/env python3
"""Subject-generalized GLMsingle runner for the new_pipeline 28-subject wave.

Resolves wave_orchestrator.py's KNOWN_ISSUE 3: sub4_spatial_sensitivity_32/code/
run_glmsingle_space.py hardcodes `subject=4` in the pilot_args.Namespace it
builds and exposes no --subject flag at all; it delegates to
pilot_raw_pipeline_sub4_6/code/run_glmsingle.py, which further hardcodes
SESSION_INDICATORS = {4: [...], 5: [...], 6: [...]} and
EXPECTED_RUN_N_VOLUMES = {5: [...]}, and restricts --subject to choices=[4,5,6].

Neither frozen/retained script is edited on disk. This wrapper imports each
fresh in-process (same importlib pattern run_glmsingle_space.py itself already
uses on the pilot script, and the same pattern wave_orchestrator.py uses on
fmriprep_sdc_workflow_v6.py) and monkeypatches exactly the two subject-keyed
dicts plus the effective subject value, deriving both dicts mechanically from
new_pipeline/config/cohort_28.json for all 28 subjects rather than hand-
entering new facts:

  - SESSION_INDICATORS[subject]: for each of the 10 experimental runs in
    order, which BIDS session (1-based) that run belongs to, per
    cohort_28.json's sessions[].experimental_runs -- this is exactly the same
    session/run assignment fmriprep_sdc_workflow_v6.subject_plan() already
    uses for BIDS-branch planning, just reshaped into GLMsingle's per-run
    array form.
  - EXPECTED_RUN_N_VOLUMES[subject]: from cohort_28.json's
    task.raw_bold_volume_qc block (modal 218 with per-subject flagged-run
    overrides, or a full per-subject override array for Sub5) -- itself a
    verbatim copy of the original cohort.json's flagged runs plus pilot
    config.json's already-published Sub5 volume table, not new data entry.

discover_branch() (fMRIPrep 25.1.3 filename discovery -- the one part of
run_glmsingle_space.py DESIGN_FREEZE.md correctly calls "already fixed and
validated") is reused directly by importing run_glmsingle_space.py fresh too,
so this wrapper does not reimplement or fork that logic either.

Still construct-only from wave_orchestrator.py's perspective: nothing in this
project executes this script for real without an explicit --execute flag one
level up. Running this file directly DOES perform a real GLMsingle fit --
treat it with the same care as fmriprep_sdc_workflow_v6.py's run-subject
--execute.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

NEW_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_AI_IAPS_ROOT = NEW_PIPELINE_ROOT.parent
DEFAULT_COHORT_CONFIG = NEW_PIPELINE_ROOT / "config" / "cohort_28.json"

PILOT_RUNNER = (
    PROJECT_AI_IAPS_ROOT / "Decoding" / "pilot_raw_pipeline_sub4_6" / "code" / "run_glmsingle.py"
)
GLMSINGLE_SPACE_SCRIPT = (
    PROJECT_AI_IAPS_ROOT / "Decoding" / "sub4_spatial_sensitivity_32" / "code" / "run_glmsingle_space.py"
)


def _fresh_import(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses/module-scope introspection safety, see wave_orchestrator.py
    spec.loader.exec_module(module)
    return module


def load_cohort(cohort_config: Path) -> dict:
    cohort = json.loads(cohort_config.read_text(encoding="utf-8"))
    if cohort.get("schema_version") != 2:
        raise ValueError(f"Unsupported cohort schema: {cohort.get('schema_version')}")
    return cohort


def derive_session_indicators(cohort: dict, subject: int) -> list[int]:
    entry = cohort["subjects"].get(str(subject))
    if entry is None:
        raise ValueError(f"No cohort_28.json entry for subject {subject}")
    by_run: dict[int, int] = {}
    for session_cfg in entry["sessions"]:
        session = int(session_cfg["session"])
        for run in session_cfg["experimental_runs"]:
            run = int(run)
            if run in by_run:
                raise ValueError(f"Run {run} assigned twice for subject {subject}")
            by_run[run] = session
    if sorted(by_run) != list(range(1, 11)):
        raise ValueError(f"Subject {subject} does not cover runs 1-10: {sorted(by_run)}")
    return [by_run[run] for run in range(1, 11)]


# Subjects processed through pilot_raw_pipeline_sub4_6/, a separate acquisition
# and audit process from full_cohort_raw_pipeline_28's 25-subject cohort.
# modal_expected_volumes/flagged_runs describe ONLY that other 25-subject
# cohort's raw-acquisition audit and must never be applied to these three by
# default -- see cohort_28.json's "pilot_cohort_subjects_note". Discovered
# 2026-08-06 the hard way: an earlier version of this function defaulted every
# subject without an override to modal_expected_volumes (218), which produced
# a false validation failure for Sub4 (real, empirically observed value: 224)
# -- the original pilot_raw_pipeline_sub4_6/code/run_glmsingle.py's own
# EXPECTED_RUN_N_VOLUMES dict correctly left Sub4/Sub6 out entirely (no check)
# rather than guessing, and this function now enforces that same discipline
# structurally instead of relying on cohort_28.json happening to omit them.
PILOT_COHORT_SUBJECTS = {4, 5, 6}


def derive_expected_run_n_volumes(cohort: dict, subject: int) -> tuple[int, ...] | None:
    # Must return a tuple, not a list: pilot_raw_pipeline_sub4_6/code/run_glmsingle.py's
    # validate_run_n_volumes() does `tuple(observed) != expected`, and its own
    # EXPECTED_RUN_N_VOLUMES dict is literally defined with tuples. A list
    # here would make that comparison always True (tuple != list in Python
    # regardless of contents) -- discovered 2026-08-06 when a first fix
    # attempt still failed with observed==expected printed identically.
    qc = cohort["task"]["raw_bold_volume_qc"]
    full_override = qc.get("full_override_by_subject", {}).get(str(subject))
    if full_override is not None:
        return tuple(int(value) for value in full_override)
    if subject in PILOT_COHORT_SUBJECTS:
        # No real evidence for this pilot-cohort subject yet -- skip the
        # check rather than inherit the other cohort's modal value.
        return None
    modal = int(qc["modal_expected_volumes"])
    volumes = [modal] * 10
    for flagged in qc.get("flagged_runs", []):
        if int(flagged["subject"]) == subject:
            volumes[int(flagged["run"]) - 1] = int(flagged["observed_volumes"])
    return tuple(volumes)


def derive_all_subjects(cohort: dict) -> list[int]:
    return sorted(int(item) for item in cohort["cohort"]["remaining_subjects"])


def build_session_indicators_dict(cohort: dict) -> dict[int, list[int]]:
    return {subject: derive_session_indicators(cohort, subject) for subject in derive_all_subjects(cohort)}


def build_expected_run_n_volumes_dict(cohort: dict) -> dict[int, tuple[int, ...]]:
    # Omit subjects with no derived expectation entirely (not {subject: None})
    # so pilot.validate_run_n_volumes()'s EXPECTED_RUN_N_VOLUMES.get(subject)
    # returns None and skips the check, matching the original dict's semantics
    # for subjects it deliberately left out.
    result = {}
    for subject in derive_all_subjects(cohort):
        volumes = derive_expected_run_n_volumes(cohort, subject)
        if volumes is not None:
            result[subject] = volumes
    return result


def patch_subject_dirs_dedup(pilot: ModuleType) -> None:
    # KNOWN_ISSUE (discovered 2026-08-07, Wave 1 Sub13): the frozen pilot
    # script's subject_dirs() returns
    #   [root / f"sub-{subject:02d}", root / f"sub-{subject}"]
    # For any subject >= 10 both format specifiers produce the identical
    # string (e.g. subject=13 -> "sub-13" both times), so the same Path
    # object is returned twice. discover_events() then globs per directory
    # and finds every events file duplicated, raising
    #   RuntimeError: Expected one events file for Sub13 Run01; found [same_path, same_path]
    # This is 100% deterministic -- it affects every two-digit subject
    # (11, 12, 13 now; 14-31 in later waves) and caused glmsingle_queue_daemon2.ps1
    # to loop relaunching Sub13 for hours since every retry failed identically.
    # Fixed here via monkeypatch (frozen script on disk left untouched) by
    # de-duplicating the candidate list before it reaches discover_events().
    original_subject_dirs = pilot.subject_dirs

    def deduped_subject_dirs(root: Path, subject: int) -> list[Path]:
        seen: list[Path] = []
        for path in original_subject_dirs(root, subject):
            if path not in seen:
                seen.append(path)
        return seen

    pilot.subject_dirs = deduped_subject_dirs


def load_pilot_generalized(cohort: dict) -> ModuleType:
    pilot = _fresh_import(PILOT_RUNNER, "ai_iaps_pilot_glmsingle_wave")
    pilot.SESSION_INDICATORS = build_session_indicators_dict(cohort)
    pilot.EXPECTED_RUN_N_VOLUMES = build_expected_run_n_volumes_dict(cohort)
    patch_subject_dirs_dedup(pilot)
    return pilot


def load_space_discovery() -> ModuleType:
    return _fresh_import(GLMSINGLE_SPACE_SCRIPT, "ai_iaps_glmsingle_space_wave")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-config", type=Path, default=DEFAULT_COHORT_CONFIG)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--fmriprep-root", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument(
        "--branch",
        default="mni_res_native",
        help="Spatial branch name, see run_glmsingle_space.py BRANCH_GLOBS (default: this wave's frozen target space).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunklen", type=int, default=10000)
    parser.add_argument("--n-pcs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort = load_cohort(args.cohort_config)
    if args.subject not in derive_all_subjects(cohort):
        raise ValueError(f"Subject {args.subject} is not in {args.cohort_config}'s cohort.remaining_subjects")

    pilot = load_pilot_generalized(cohort)
    space = load_space_discovery()

    def replacement(root: Path, subject: int, _space: str):
        return space.discover_branch(root, subject, args.branch)

    pilot.discover_run_files = replacement

    pilot_args = argparse.Namespace(
        subject=args.subject,
        fmriprep_root=args.fmriprep_root,
        bids_root=args.bids_root,
        space=args.branch,
        output=args.output,
        chunklen=args.chunklen,
        n_pcs=args.n_pcs,
        fixed_frac=None,
        fracs=None,
        max_voxels=None,
        overwrite=False,
    )
    pilot.run(pilot_args)


if __name__ == "__main__":
    main()
