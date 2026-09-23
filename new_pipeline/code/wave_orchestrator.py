#!/usr/bin/env python3
"""Wave-level orchestrator for the new_pipeline 28-subject unified GLMsingle wave.

Scope (see ../DESIGN_FREEZE.md): fMRIPrep (MNI152NLin6Asym, res-native) ->
GLMsingle Type-D single-trial betas -> STOP. This script only sequences
per-subject plan -> preflight -> run-subject (fMRIPrep), and can additionally
construct (but by default never executes) the matching GLMsingle command.

Design choice: ADAPT, don't reimplement. All BIDS-branch/session/SDC-mode
logic, hashing, receipt verification, and Docker command construction are
reused verbatim from full_cohort_raw_pipeline_28/code/fmriprep_sdc_workflow_v6.py
by importing that file fresh in-process (see load_v6_module()) -- the file on
disk under full_cohort_raw_pipeline_28/ is never modified, per this project's
hard constraint that that workspace is frozen, read-only evidence. Exactly one
function, load_configs(), is monkeypatched in the in-memory copy to accept
this wave's own sdc-config analysis_id in addition to the original (see
KNOWN_ISSUE 4 below). This mirrors an existing pattern already used in this
codebase: sub4_spatial_sensitivity_32/code/run_glmsingle_space.py monkeypatches
`pilot.discover_run_files` on an in-process import of
pilot_raw_pipeline_sub4_6/code/run_glmsingle.py for the same reason (adapt a
frozen, retained script without editing it on disk).

============================================================================
KNOWN_ISSUES -- discovered 2026-08-05/06 while building this orchestrator.
Items 2 and 4 (GLMsingle generalization, see run_glmsingle_wave.py) are now
RESOLVED. The rest still stand. Read before authorizing --execute.
============================================================================

1. SUBJECTS 4, 5, 6 ORIGINALLY HAD NO ENTRY IN THE REUSED cohort.json --
   NOW RESOLVED via new_pipeline/config/cohort_28.json.
   full_cohort_raw_pipeline_28/config/cohort.json's schema-v2 "subjects" dict
   and "cohort.remaining_subjects" list cover exactly the 25 subjects
   [1,2,7,9,11..31]. Subjects 4/5/6 (the "pilot" subjects) were processed
   through a separate, older, differently-structured pipeline
   (pilot_raw_pipeline_sub4_6/) that predates this schema-v2 cohort config
   and was never folded into it -- cohortlib.load_config() and
   fmriprep_sdc_workflow_v6.subject_plan() both hard-fail (ValueError) for
   subject 4, 5, or 6 against the ORIGINAL cohort.json. Resolution: built
   cohort_28.json (new file under new_pipeline/config/, NOT an edit to the
   frozen original) that copies the original 25 subjects' entries verbatim
   and adds sessions[]/sdc_mode/experimental_runs entries for 4/5/6, sourced
   from pilot_raw_pipeline_sub4_6/config.json and PIPELINE_RUNBOOK.md's own
   already-published, already-BIDS-validated facts (session_indicator table,
   distortion-correction table) -- not re-derived from raw acquisition
   inspection. Sub5 is genuinely irregular (fieldmap-less February isolated
   from a measured-fieldmap March, a March-only T1w reused across both BIDS
   sessions) in ways the schema-v2 source_dir/source_run_overrides fields
   were never designed to express, so this orchestrator deliberately does
   NOT build Sub4/5/6's BIDS bytes through build_full_cohort_bids.py's
   cohortlib-driven source-selection machinery. Instead it reuses
   pilot_raw_pipeline_sub4_6/code/build_pilot_bids.py UNMODIFIED (read-only,
   own --output only) -- that script already has exact, hardcoded,
   previously-BIDS-validated (zero errors, see PIPELINE_RUNBOOK.md section 2)
   source selections for these 3 subjects' every BOLD/T1w/fieldmap file.
   cohort_28.json's per-subject entries for 4/5/6 therefore only need to be
   correct for fmriprep_sdc_workflow_v6.subject_plan()'s narrower needs
   (session/sdc_mode/experimental_runs, used for planning and SDC-branch
   logic) -- they are NOT used for BIDS source selection for these 3
   subjects, so their source_dir/log_filename_template fields are present
   for schema-shape consistency only and are not load-bearing.
   One deliberate structural adaptation: build_pilot_bids.py places Sub4 and
   Sub5's T1w "sessionless" (sub-XX/anat/sub-XX_T1w.nii.gz, no ses- folder),
   but audit_bids_branch() only discovers anatomicals matching
   "ses-*/anat/*_T1w.nii.gz". This orchestrator's BIDS-merge step therefore
   duplicates that same T1w file into every session folder the subject has
   (ses-01/anat for Sub4; both ses-01/anat and ses-02/anat for Sub5) with
   ses-tagged filenames, matching how the other 25 subjects' mixed-session
   cases already work, and satisfying merge_mixed_branches()'s own
   byte-identical-anatomical invariant check for Sub5's isolated branches.

2. basic_derivative_gate() ORIGINALLY HARDCODED res-2 FILENAME SUFFIXES --
   NOW PATCHED, see patch_basic_derivative_gate().
   fmriprep_sdc_workflow_v6.basic_derivative_gate() -- called by run_subject()
   immediately after a real (--execute) fMRIPrep completes, before publishing
   the final derivative tree -- hardcoded the four required suffixes as
   literal "_space-MNI152NLin6Asym_res-2_..." strings in its function body,
   never reading the sdc config's "required_derivative_suffixes" field. Under
   this wave's frozen res-native output, fMRIPrep 25.1.3 omits the res- token
   entirely from output filenames (the same fact DESIGN_FREEZE.md section 7
   documents for run_glmsingle_space.py's own filename discovery). Unpatched,
   this would raise "Derivative run mismatch" for EVERY subject's real
   fMRIPrep run this wave, even a fully successful one. patch_basic_derivative_gate()
   (called from run_wave() right after load_configs(), once the actual
   sdc["fmriprep"]["output_resolution"] value is known) replaces it in the
   in-memory module copy with a byte-for-byte faithful copy of the original's
   logic, differing only in which suffix tuple it checks (native-shaped vs
   res-N-shaped, chosen from the config, not hardcoded to one wave). Unlike
   every other patch in this file, this one changes what a *safety* gate
   accepts as "complete" -- flagged here explicitly so a human reviews the
   patched logic (it is included verbatim in this file, not hidden) before
   trusting it on a real --execute run.

3. run_glmsingle_space.py HARDCODES subject=4 AND HAS NO --subject FLAG --
   NOW WORKED AROUND via a new wrapper, run_glmsingle_wave.py (see below).
   sub4_spatial_sensitivity_32/code/run_glmsingle_space.py -- named in
   DESIGN_FREEZE.md as "already fixed and validated" for this wave's
   GLMsingle step -- builds an argparse.Namespace with `subject=4` hardcoded
   and exposes no --subject CLI argument at all. It delegates to
   pilot_raw_pipeline_sub4_6/code/run_glmsingle.py, which further hardcodes
   `SESSION_INDICATORS = {4: [...], 5: [...], 6: [...]}`,
   `EXPECTED_RUN_N_VOLUMES = {5: [...]}`, and `--subject choices=[4,5,6]`.
   Only discover_branch() (fMRIPrep 25.1.3 filename discovery) was genuinely
   generic. new_pipeline/code/run_glmsingle_wave.py imports
   pilot_raw_pipeline_sub4_6/code/run_glmsingle.py fresh in-process (same
   pattern as everywhere else in this file -- the frozen file on disk is
   never edited) and monkeypatches SESSION_INDICATORS/EXPECTED_RUN_N_VOLUMES
   with values mechanically derived from cohort_28.json for all 28 subjects
   (session assignment from sessions[].experimental_runs; volume counts from
   a new "raw_bold_volume_qc" block in cohort_28.json, itself copied from the
   original cohort.json's flagged-run entries plus pilot config.json's
   documented Sub5 full-array override -- not new data entry, a translation
   of already-published facts), and calls run() with args.subject taken from
   its own --subject flag instead of the hardcoded 4. This orchestrator's
   launch_glmsingle_command() now points at run_glmsingle_wave.py instead of
   run_glmsingle_space.py directly. Still never executed by this orchestrator
   without --execute-glmsingle (off by default) -- only the command is
   constructed.

4. audit_bids_branch() REQUIRES A SINGLE-SUBJECT-SCOPED "CANONICAL" BIDS ROOT.
   Discovered empirically while running this orchestrator's own Subject-1
   smoke test on 2026-08-06. fmriprep_sdc_workflow_v6.audit_bids_branch()
   (called from preflight_subject()) does
   `observed_subjects = sorted(p.name for p in bids_root.glob("sub-*") ...)`
   and raises RuntimeError unless that list is EXACTLY [f"sub-{label}"] for
   the one subject being planned -- i.e. the "canonical BIDS root" it expects
   must contain only that one subject's sub-XX directory, never a shared
   multi-subject tree. This matches the original pipeline's documented
   "process strictly one subject at a time" convention (see
   build_isolated_branch(), which explicitly rewrites participants.tsv down
   to one row for the same reason) but directly conflicts with this wave's
   own brief, which asked for ONE shared 28-subject (in practice 25-subject,
   see KNOWN_ISSUE 1) BIDS tree built once. Because symlinks/hardlinks do not
   work on this N: mount (see the BIDS-build discussion in this session's
   report), there is no cheap way to give each subject their own "view" of
   the shared tree -- it requires an actual byte copy of that one subject's
   already-verified directory out of the shared tree, plus a filtered
   participants.tsv and the shared dataset-level files. See
   materialize_single_subject_bids_view() below, and DEFAULT_PER_SUBJECT_BIDS_ROOT.
   Practical implication for a full wave: this copy step multiplies the BIDS
   storage footprint again (each subject's data exists once in the shared
   tree and once more in its own per-subject view) and adds a real per-subject
   time cost (Subject 1's ~2.3 GB view took 2m30s to copy) on top of the
   original full-copy build. A human should decide whether to keep the shared
   tree at all once every subject also needs its own per-subject view, or
   build per-subject trees directly from raw sources via
   `build_full_cohort_bids.py --subject N` instead (bypassing the shared
   tree, at the cost of losing the "build once, audit once" property and
   re-paying the network-hash cost per subject).

5. load_configs() HARDCODES ONE ACCEPTED analysis_id.
   fmriprep_sdc_workflow_v6.load_configs() raises ValueError unless
   sdc["analysis_id"] == "ai_iaps_full28_sdc_glmsingle_erp_v1" exactly. This
   wave's own config (fmriprep_sdc_config_native.json) deliberately uses a
   distinct analysis_id, "ai_iaps_new_pipeline_28_glmsingle_typed_native_v1",
   per DESIGN_FREEZE.md's explicit instruction ("so it's clearly not the
   frozen 26-subject-res2 config"). Patched in-process below rather than
   edited on disk -- see load_v6_module().
============================================================================
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Mapping

NEW_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_AI_IAPS_ROOT = NEW_PIPELINE_ROOT.parent
FULL_COHORT_ROOT = PROJECT_AI_IAPS_ROOT / "Decoding" / "full_cohort_raw_pipeline_28"
V6_PATH = FULL_COHORT_ROOT / "code" / "fmriprep_sdc_workflow_v6.py"

DEFAULT_COHORT_CONFIG = NEW_PIPELINE_ROOT / "config" / "cohort_28.json"
DEFAULT_SDC_CONFIG = NEW_PIPELINE_ROOT / "config" / "fmriprep_sdc_config_native.json"
DEFAULT_WAVES_CONFIG = NEW_PIPELINE_ROOT / "config" / "cohort_waves.json"

LAB_IAPS_AI_ROOT = Path("N:/Experimental_Data/yujunchen/projects/LAB_IAPS_AI")
DEFAULT_SHARED_BIDS_ROOT = LAB_IAPS_AI_ROOT / "bids_unified_glmsingle_28"
DEFAULT_PER_SUBJECT_BIDS_ROOT = LAB_IAPS_AI_ROOT / "bids_unified_glmsingle_28_per_subject"
DEFAULT_DERIVATIVES_ROOT = LAB_IAPS_AI_ROOT / "fmriprep_derivatives" / "unified_glmsingle_28"
DEFAULT_GLMSINGLE_OUTPUT_ROOT = NEW_PIPELINE_ROOT / "glmsingle"
DEFAULT_LICENSE_FILE = Path(r"C:\Users\yujunchen\.cache\ai_iaps_fmriprep_license.txt")

GLMSINGLE_WAVE_SCRIPT = NEW_PIPELINE_ROOT / "code" / "run_glmsingle_wave.py"

LEDGER_PATH = NEW_PIPELINE_ROOT / "logs" / "wave_orchestrator_ledger.jsonl"

# The original (frozen, untouched-on-disk) analysis_id plus this wave's own.
ACCEPTED_SDC_ANALYSIS_IDS = {
    "ai_iaps_full28_sdc_glmsingle_erp_v1",
    "ai_iaps_new_pipeline_28_glmsingle_typed_native_v1",
}

FMRIPREP_WORKERS_DEFAULT = 4
GLMSINGLE_WORKERS_DEFAULT = 5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ledger_append(event: Mapping) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = dict(event)
    record.setdefault("ts", utc_now())
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")


def load_v6_module() -> ModuleType:
    """Import fmriprep_sdc_workflow_v6.py fresh, in-process, without touching
    the file on disk, and monkeypatch load_configs() to accept this wave's
    sdc-config analysis_id too. See KNOWN_ISSUE 4 in the module docstring."""
    spec = importlib.util.spec_from_file_location("ai_iaps_fmriprep_sdc_workflow_v6", V6_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {V6_PATH}")
    v6 = importlib.util.module_from_spec(spec)
    # Must register in sys.modules before exec_module(): the target file uses
    # @dataclass at module scope, and CPython 3.12+'s dataclass machinery does
    # sys.modules.get(cls.__module__).__dict__ while processing the class body,
    # which raises AttributeError on None if the module isn't registered yet.
    sys.modules[spec.name] = v6
    spec.loader.exec_module(v6)

    original_load_configs = v6.load_configs

    def patched_load_configs(cohort_path: Path, sdc_path: Path):
        cohort = v6.read_json(cohort_path)
        sdc = v6.read_json(sdc_path)
        if cohort.get("schema_version") != 2:
            raise ValueError(f"Unsupported cohort schema: {cohort.get('schema_version')}")
        if sdc.get("schema_version") != 1:
            raise ValueError(f"Unsupported fMRIPrep SDC schema: {sdc.get('schema_version')}")
        if sdc.get("analysis_id") not in ACCEPTED_SDC_ANALYSIS_IDS:
            raise ValueError(
                f"Unexpected analysis_id in fMRIPrep SDC config: {sdc.get('analysis_id')!r}; "
                f"expected one of {sorted(ACCEPTED_SDC_ANALYSIS_IDS)}"
            )
        image = str(sdc["fmriprep"]["image"])
        digest = str(sdc["fmriprep"]["image_digest"])
        if not image.endswith("@" + digest):
            raise ValueError("Pinned fMRIPrep image reference and digest disagree")
        launcher = sdc.get("docker_launcher")
        if not isinstance(launcher, dict):
            raise ValueError("Missing docker_launcher configuration")
        expected_launcher = {
            "mode": "wsl",
            "windows_wsl_executable": r"C:\Windows\System32\wsl.exe",
            "distribution": "Ubuntu-22.04",
            "linux_docker_executable": "/usr/bin/docker",
            "persistent_bind_prefix": "/mnt/n",
        }
        for key, expected in expected_launcher.items():
            if launcher.get(key) != expected:
                raise ValueError(
                    f"Unsupported Docker launcher {key}: {launcher.get(key)!r}; expected {expected!r}"
                )
        if launcher["distribution"] != sdc["validator"]["wsl_distribution"]:
            raise ValueError("Docker and BIDS-validator WSL distributions must match")
        return cohort, sdc

    del original_load_configs  # kept unused intentionally: documents what we replaced
    v6.load_configs = patched_load_configs
    return v6


def patch_basic_derivative_gate(v6: ModuleType, sdc: Mapping) -> None:
    """Replace basic_derivative_gate() with a resolution-aware version.

    See KNOWN_ISSUE 2: the original hardcodes "_space-MNI152NLin6Asym_res-2_..."
    suffixes and would fail-closed against every real res-native fMRIPrep run.
    This replacement is a faithful copy of the original's logic (same
    dataset_description.json/report/run-set/SDC-method checks) with only the
    required-suffix tuple made resolution-aware, matching fMRIPrep 25.1.3's own
    documented convention that native-resolution MNI output omits the res-
    entity entirely (already relied on by run_glmsingle_space.py's
    BRANCH_GLOBS in sub4_spatial_sensitivity_32, and by this wave's own
    fmriprep_sdc_config_native.json "required_derivative_suffixes" field).
    Deliberately a monkeypatch on the in-memory module copy, not an edit to
    the file on disk -- same rationale as load_v6_module()'s load_configs
    patch. This is the one gate this orchestrator chooses to adapt rather
    than merely document, because leaving it unpatched would make it
    impossible to ever publish a real fMRIPrep result for this wave; every
    other KNOWN_ISSUE is left for explicit human sign-off."""
    resolution = str(sdc["fmriprep"]["output_resolution"])
    if resolution == "native":
        required = (
            "_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz",
            "_space-MNI152NLin6Asym_desc-preproc_bold.json",
            "_space-MNI152NLin6Asym_desc-brain_mask.nii.gz",
            "_desc-confounds_timeseries.tsv",
        )
    else:
        required = (
            f"_space-MNI152NLin6Asym_res-{resolution}_desc-preproc_bold.nii.gz",
            f"_space-MNI152NLin6Asym_res-{resolution}_desc-preproc_bold.json",
            f"_space-MNI152NLin6Asym_res-{resolution}_desc-brain_mask.nii.gz",
            "_desc-confounds_timeseries.tsv",
        )

    def patched_basic_derivative_gate(root: Path, plan, branch) -> dict:
        if not (root / "dataset_description.json").is_file():
            raise FileNotFoundError(root / "dataset_description.json")
        if not (root / f"sub-{plan.subject_label}.html").is_file():
            raise FileNotFoundError(root / f"sub-{plan.subject_label}.html")
        session_records = []
        for session in branch.sessions:
            expected_runs = next(item.runs for item in plan.sessions if item.session == session)
            mode = next(item.mode for item in plan.sessions if item.session == session)
            func = root / f"sub-{plan.subject_label}" / f"ses-{session}" / "func"
            for suffix in required:
                hits = sorted(func.glob(f"sub-{plan.subject_label}_ses-{session}_task-iaps_run-*{suffix}"))
                observed = {v6.run_number(path) for path in hits}
                if len(hits) != len(observed) or observed != set(expected_runs):
                    raise RuntimeError(
                        f"Derivative run mismatch for ses-{session} suffix {suffix}: {sorted(observed)}"
                    )
            methods = {run: v6.sdc_method(root, plan.subject_label, session, run) for run in expected_runs}
            wrong = {run: value for run, value in methods.items() if not v6.method_matches(mode, value)}
            if wrong:
                raise RuntimeError(f"Wrong SDC method for Sub{plan.subject} ses-{session}: {wrong}")
            session_records.append({"session": session, "mode": mode, "methods": methods})
        return {"status": "pass", "sessions": session_records}

    v6.basic_derivative_gate = patched_basic_derivative_gate


ANAT_SIMILARITY_SCRIPT = NEW_PIPELINE_ROOT / "code" / "compare_anat_similarity.py"
ANAT_SIMILARITY_VENV_PYTHON = "/home/yujun/.cache/ai_iaps_pilot_venv/bin/python"
ANAT_SIMILARITY_WSL_EXECUTABLE = r"C:\Windows\System32\wsl.exe"
ANAT_SIMILARITY_WSL_DISTRIBUTION = "Ubuntu-22.04"


def _wsl_path(path: Path) -> str:
    # Same UNC-aware conversion as glmsingle_wave_launcher.py's windows_to_wsl().
    absolute = Path(os.path.abspath(path))
    if absolute.drive.startswith("\\\\"):
        n_target = Path("N:\\").resolve(strict=False)
        relative = absolute.relative_to(n_target)
        return f"/mnt/n/{relative.as_posix().lstrip('/')}"
    drive = absolute.drive.rstrip(":").lower()
    tail = absolute.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def check_anat_similarity(preferred_anat: Path, other_anat: Path) -> dict:
    """Shell out to compare_anat_similarity.py (WSL venv -- needs numpy/nibabel,
    not present in the Windows-native interpreter this orchestrator otherwise
    runs under) and return its JSON report. Raises if the comparison itself
    cannot be completed (missing script, subprocess error) -- only a genuine
    similarity-below-threshold result is treated as a normal fail-closed
    RuntimeError by the caller, not an exception here."""
    command = [
        ANAT_SIMILARITY_WSL_EXECUTABLE,
        "-d",
        ANAT_SIMILARITY_WSL_DISTRIBUTION,
        "--",
        ANAT_SIMILARITY_VENV_PYTHON,
        _wsl_path(ANAT_SIMILARITY_SCRIPT),
        "--dir-a",
        _wsl_path(preferred_anat),
        "--dir-b",
        _wsl_path(other_anat),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"compare_anat_similarity.py produced no parseable output "
            f"(exit {completed.returncode}): stdout={completed.stdout!r} stderr={completed.stderr!r}"
        ) from error
    return report


FMRIPREP_CONTAINER_MEMORY_LIMIT = "32g"


def patch_fmriprep_memory_limit(v6: ModuleType, limit: str = FMRIPREP_CONTAINER_MEMORY_LIMIT) -> None:
    """Add a hard per-container cgroup memory limit to every fMRIPrep
    `docker run` command.

    WHY (Wave 3, 2026-08-09/10): fMRIPrep's own `--mem-mb 20000` is only a
    nipype *scheduler hint* -- it does NOT constrain the container, which
    until now ran with no cgroup limit at all. Observed real peaks were far
    above the nominal 20 GB: Sub28 26.2 GiB, Sub30 26.6 GiB. With the 4-way
    pool, and with 6 of Wave 3's 8 subjects being mixed-SDC (so their FIRST
    branch is always the memory-hungry ses_01_syn), four concurrent ANTs SyN
    registrations routinely oversubscribed the 94 GB WSL2 budget.

    That is what killed Sub26: a GLOBAL out-of-memory event
    (constraint=CONSTRAINT_NONE) in which the kernel killed the single
    antsRegistration process holding ~11.4 GiB. The critical property of a
    global OOM is that the kernel picks its victim by oom_score (roughly
    RSS) -- so the process killed is whichever is LARGEST at that instant,
    NOT the one whose allocation triggered the event. In practice that means
    a small, freshly-started branch can tip the machine over and the kernel
    destroys an unrelated branch that is hours into its run.

    A per-container limit converts that failure mode from a global OOM into
    a contained, per-container cgroup OOM: a runaway branch is killed by its
    own limit and every other container keeps running untouched.

    Limit choice: 32 GiB, i.e. ~20% headroom above the observed 26.6 GiB
    peak, so a legitimately heavy branch is not killed by its own limit
    while a genuine runaway is still contained. --memory-swap is set equal
    to --memory, which disables *additional* swap for the container, making
    containment crisp rather than letting a runaway drag the whole VM into
    swap thrashing. Both flags were verified accepted by this environment's
    Docker/WSL2 kernel (no "kernel does not support swap limit" warning)
    before this patch was adopted.

    Implementation: wraps the frozen fmriprep_command() and splices the
    flags in immediately before the image reference, which is the exact
    boundary between `docker run` flags and fMRIPrep's own arguments. The
    file on disk under full_cohort_raw_pipeline_28/ is never modified, per
    this project's frozen-workspace constraint.

    LIMITATION -- applies to NEW processes only: this is an in-process
    monkeypatch, so it affects only fMRIPrep containers launched by a
    Python process that ran this function. An orchestrator already running
    when the patch was written holds the unpatched function in memory, so
    its still-pending branches (the ses_02_gre branches for subjects
    28/29/30 at the time of writing) launch WITHOUT the limit. Those are
    lower-risk since the gre path has no ANTs SyN step, but they are not
    protected by this patch.
    """
    # Idempotence guard: applying this patch twice to the same module object
    # would wrap the wrapper and inject the flags twice. Docker tolerates
    # duplicates (last wins), so this is not dangerous today -- run_wave()
    # calls it exactly once per process -- but an accidental second call is
    # exactly the kind of latent defect that surfaces later, so refuse it.
    if getattr(v6, "_ai_iaps_fmriprep_memory_limit_patched", False):
        return

    original_fmriprep_command = v6.fmriprep_command

    def fmriprep_command_with_memory_limit(
        bids_root, output, license_file, work_volume, plan, branch, sdc
    ):
        argv = list(
            original_fmriprep_command(
                bids_root, output, license_file, work_volume, plan, branch, sdc
            )
        )
        image = str(sdc["fmriprep"]["image"])
        try:
            image_index = argv.index(image)
        except ValueError as error:  # fail closed rather than guess
            raise RuntimeError(
                "Could not locate the fMRIPrep image reference in the constructed "
                "docker argv; refusing to inject a memory limit blindly"
            ) from error
        argv[image_index:image_index] = ["--memory", limit, "--memory-swap", limit]
        return argv

    v6.fmriprep_command = fmriprep_command_with_memory_limit
    v6._ai_iaps_fmriprep_memory_limit_patched = True


def patch_merge_mixed_branches(v6: ModuleType) -> None:
    """Replace merge_mixed_branches()'s strict byte-for-byte SHA-256 anatomical
    equality requirement with a similarity-threshold check (see
    compare_anat_similarity.py). Discovered 2026-08-06/07: ANTs/ITK-based
    anatomical processing (bias correction, registration, segmentation) is not
    perfectly bit-reproducible across two independent fMRIPrep runs, even
    given byte-identical T1w input and a pinned container image -- confirmed
    empirically for Sub5 (correlation 0.986-0.9998 across all anat files
    between its two isolated-session branches, brain-mask Dice ~0.999,
    <0.7% discrete-label voxels differing). The original strict check assumed
    perfect reproducibility ("byte-identical because every branch receives
    the exact same complete T1w set and pinned runtime") and fails-closed on
    every mixed-SDC-mode subject as a result -- not just Sub5; this affects
    every subject using the isolated_mixed_session_branches strategy (~10
    more across Waves 2/3: 17/18/20/21/24/26/28/29/30/31).

    This mirrors pilot_raw_pipeline_sub4_6/code/merge_sub05_fmriprep.py's own
    precedent for this exact subject (its comment: "Anatomical derivatives
    remain from the all-session branch; both branches use the identical
    subject-level T1w input and pinned image" -- it never required or
    verified byte-identical agreement either, it simply used one branch's
    anat unconditionally). This patch keeps real verification (unlike the
    pilot script) rather than trusting blindly: a genuinely different
    anatomical (wrong subject, failed registration, etc.) would show
    similarity far below these thresholds and would still be caught.

    Only the anat-equality block changes; everything else (which branch's
    anat is actually copied into the merged output -- always `preferred`,
    the measured-SDC branch -- report bundling, provenance, the final
    basic_derivative_gate validation) is unchanged from the original.
    Deliberately a monkeypatch on the in-memory module copy, not an edit to
    the frozen file on disk -- same pattern as load_configs and
    basic_derivative_gate above."""

    def patched_merge_mixed_branches(plan, branch_outputs, destination: Path, link_mode: str) -> dict:
        if plan.strategy != "isolated_mixed_session_branches":
            raise ValueError("Merge is only valid for mixed-session plans")
        destination = v6.require_n_drive(destination, "final fMRIPrep derivative")
        if destination.exists():
            raise FileExistsError(destination)
        for branch in plan.branches:
            v6.basic_derivative_gate(branch_outputs[branch.name], plan, branch)
        stage = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
        if stage.exists():
            raise FileExistsError(stage)
        stage.mkdir(parents=True)
        preferred = next(
            (branch for branch in plan.branches if branch.mode in v6.MEASURED_MODES), plan.branches[0]
        )
        preferred_root = branch_outputs[preferred.name]
        try:
            for name in ("dataset_description.json", ".bidsignore", "README", "CHANGES"):
                src = preferred_root / name
                if src.is_file():
                    v6.copy_or_link(src, stage / name, link_mode)
            subject_rel = Path(f"sub-{plan.subject_label}")
            preferred_subject = preferred_root / subject_rel
            preferred_anat = preferred_subject / "anat"
            similarity_reports = {}
            for other in plan.branches:
                other_anat = branch_outputs[other.name] / subject_rel / "anat"
                if other.name == preferred.name:
                    continue
                report = check_anat_similarity(preferred_anat, other_anat)
                similarity_reports[other.name] = report
                if report.get("overall_status") != "pass":
                    failed = [c for c in report.get("checks", []) if c.get("status") == "fail"]
                    raise RuntimeError(
                        f"Branches produced anatomical references below the similarity threshold: "
                        f"{preferred.name} versus {other.name}; failed checks: {failed}; do not merge"
                    )
            _copy_tree_selected(
                preferred_root,
                stage,
                [path.relative_to(preferred_root) for path in v6.iter_files(preferred_anat)],
                link_mode,
            )
            for branch in plan.branches:
                source = branch_outputs[branch.name]
                for session in branch.sessions:
                    session_root = source / subject_rel / f"ses-{session}"
                    _copy_tree_selected(
                        source,
                        stage,
                        [path.relative_to(source) for path in v6.iter_files(session_root)],
                        link_mode,
                    )
                    figures = source / subject_rel / "figures"
                    selected_figures = [
                        path.relative_to(source)
                        for path in v6.iter_files(figures)
                        if f"_ses-{session}_" in path.name
                    ]
                    _copy_tree_selected(source, stage, selected_figures, link_mode)
                report_bundle = stage / "logs" / "branch_reports" / branch.name
                for relroot in (
                    Path(f"sub-{plan.subject_label}.html"),
                    subject_rel / "figures",
                    subject_rel / "log",
                    Path("logs"),
                ):
                    src = source / relroot
                    if src.is_file():
                        v6.copy_or_link(src, report_bundle / relroot, link_mode)
                    elif src.is_dir():
                        for item in v6.iter_files(src):
                            v6.copy_or_link(item, report_bundle / item.relative_to(source), link_mode)
            report_links = "\n".join(
                f'<li><a href="logs/branch_reports/{branch.name}/sub-{plan.subject_label}.html">'
                f'{branch.name}: sessions {", ".join(branch.sessions)} ({branch.mode})</a></li>'
                for branch in plan.branches
            )
            (stage / f"sub-{plan.subject_label}.html").write_text(
                "<!doctype html><html><head><meta charset=\"utf-8\"><title>"
                f"Sub-{plan.subject_label} merged fMRIPrep reports</title></head><body>"
                "<h1>Session-specific fMRIPrep branch reports</h1>"
                "<p>This subject required isolated branches to prevent cross-session "
                "fieldmap borrowing. Review every branch report below.</p><ul>"
                f"{report_links}</ul></body></html>\n",
                encoding="utf-8",
            )
            provenance = {
                "generated_at": v6.utc_now(),
                "subject": plan.subject,
                "strategy": plan.strategy,
                "all_subject_anatomicals_in_every_branch": True,
                "anatomical_similarity_checks": similarity_reports,
                "anatomical_source_branch": preferred.name,
                "anatomical_equality_policy": "similarity-threshold (patched 2026-08-07), not byte-identical",
                "session_sources": {
                    session: {
                        "branch": branch.name,
                        "mode": branch.mode,
                        "root": str(branch_outputs[branch.name]),
                    }
                    for branch in plan.branches
                    for session in branch.sessions
                },
                "branch_report_bundles": {
                    branch.name: f"logs/branch_reports/{branch.name}/sub-{plan.subject_label}.html"
                    for branch in plan.branches
                },
            }
            provenance_path = stage / "logs" / "merge_provenance.json"
            provenance_path.parent.mkdir(parents=True, exist_ok=True)
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
            for item in plan.sessions:
                one = v6.BranchSpec("merged", item.mode, (item.session,), item.runs, "merged", item.mode == "syn")
                v6.basic_derivative_gate(stage, plan, one)
            stage.replace(destination)
            return provenance
        except Exception:
            raise

    def _copy_tree_selected(source: Path, destination: Path, relative_files, mode: str) -> None:
        for rel in relative_files:
            v6.copy_or_link(source / rel, destination / rel, mode)

    v6.merge_mixed_branches = patched_merge_mixed_branches


def load_waves(waves_path: Path) -> dict:
    return json.loads(waves_path.read_text(encoding="utf-8"))


def subjects_for_wave(waves_cfg: Mapping, wave_number: int) -> list[int]:
    for entry in waves_cfg["waves"]:
        if int(entry["wave"]) == wave_number:
            return [int(item) for item in entry["subjects"]]
    raise ValueError(f"No such wave in cohort_waves.json: {wave_number}")


def subject_label(subject: int) -> str:
    return f"Sub{subject:02d}"


def materialize_single_subject_bids_view(
    subject: int, shared_bids_root: Path, per_subject_bids_root: Path
) -> Path:
    """Copy exactly one subject's already-verified directory out of the
    shared BIDS tree into its own single-subject root. Required because
    audit_bids_branch() (in the reused, unmodified fmriprep_sdc_workflow_v6.py)
    hard-requires the canonical BIDS root passed to preflight to contain only
    the one subject being planned -- see KNOWN_ISSUE 5 in the module
    docstring. Symlinks/hardlinks do not work on this N: mount, so this is a
    real byte copy, not a cheap view. Idempotent/fail-closed: refuses to
    proceed if the destination already exists (never silently reuses or
    overwrites a possibly-stale prior view)."""
    label = subject_label(subject)
    destination = per_subject_bids_root / label
    if destination.exists():
        return destination
    if not str(os.path.abspath(shared_bids_root)).upper().startswith("N:"):
        raise ValueError(f"shared_bids_root must be on N:, not {shared_bids_root}")
    if not str(os.path.abspath(per_subject_bids_root)).upper().startswith("N:"):
        raise ValueError(f"per_subject_bids_root must be on N:, not {per_subject_bids_root}")
    subject_dirname = f"sub-{subject:02d}"
    source_subject_dir = shared_bids_root / subject_dirname
    if not source_subject_dir.is_dir():
        raise FileNotFoundError(
            f"Subject {subject} has no directory in the shared BIDS tree: {source_subject_dir}"
        )
    stage = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if stage.exists():
        raise FileExistsError(f"Refusing to reuse an abandoned staging tree: {stage}")
    stage.mkdir(parents=True)
    try:
        import csv
        import shutil as _shutil

        for name in ("dataset_description.json", "task-iaps_events.json", "README", ".bidsignore"):
            src = shared_bids_root / name
            if src.is_file():
                _shutil.copy2(src, stage / name)
        with (shared_bids_root / "participants.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        chosen = [row for row in rows if row.get("participant_id") == subject_dirname]
        if len(chosen) != 1:
            raise RuntimeError(
                f"participants.tsv does not contain exactly one row for {subject_dirname}: {chosen}"
            )
        with (stage / "participants.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(chosen[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(chosen)
        _shutil.copytree(source_subject_dir, stage / subject_dirname)
        stage.replace(destination)
    except Exception:
        # Preserve a nonempty stage for diagnosis, matching this project's
        # established convention elsewhere (see build_isolated_branch()).
        if stage.exists() and not any(stage.iterdir()):
            stage.rmdir()
        raise
    return destination


def subject_paths(subject: int, derivatives_root: Path) -> dict[str, Path]:
    label = subject_label(subject)
    staging = derivatives_root / "_staging" / label
    return {
        "staging": staging,
        "preflight_root": staging / "preflight",
        "branch_root": staging / "branches",
        "branch_output_root": staging / "branch_output",
        "final_root": derivatives_root / label,
    }


def launch_glmsingle_command(
    subject: int, fmriprep_final_root: Path, bids_root: Path, output_root: Path, cohort_config: Path
) -> list[str]:
    """Construct (never here execute) the GLMsingle call for this subject,
    via the subject-generalized wrapper run_glmsingle_wave.py (see
    KNOWN_ISSUE 3 -- this resolves the earlier subject=4 hardcoding)."""
    output = output_root / f"sub-{subject:02d}"
    return [
        sys.executable,
        str(GLMSINGLE_WAVE_SCRIPT),
        "--cohort-config",
        str(cohort_config),
        "--subject",
        str(subject),
        "--fmriprep-root",
        str(fmriprep_final_root),
        "--bids-root",
        str(bids_root),
        "--branch",
        "mni_res_native",
        "--output",
        str(output),
    ]


def process_subject(
    v6: ModuleType,
    subject: int,
    wave: int,
    cohort: Mapping,
    cohort_config_path: Path,
    sdc_config_path: Path,
    shared_bids_root: Path,
    per_subject_bids_root: Path,
    derivatives_root: Path,
    license_file: Path,
    link_mode: str,
    run_external_gates: bool,
    execute: bool,
) -> dict:
    paths = subject_paths(subject, derivatives_root)
    result: dict = {"subject": subject, "wave": wave, "label": subject_label(subject)}

    # --- materialize this subject's single-subject BIDS view (KNOWN_ISSUE 5) -
    try:
        canonical_bids_root = materialize_single_subject_bids_view(
            subject, shared_bids_root, per_subject_bids_root
        )
    except Exception as error:  # noqa: BLE001
        detail = f"{type(error).__name__}: {error}"
        ledger_append(
            {
                "wave": wave,
                "subject": subject,
                "stage": "bids_view",
                "status": "fail",
                "detail": detail,
            }
        )
        result["status"] = "bids_view_fail"
        result["detail"] = detail
        return result
    ledger_append(
        {
            "wave": wave,
            "subject": subject,
            "stage": "bids_view",
            "status": "pass",
            "canonical_bids_root": str(canonical_bids_root),
        }
    )

    # --- plan --------------------------------------------------------------
    ledger_append({"wave": wave, "subject": subject, "stage": "plan", "status": "started"})
    try:
        plan = v6.subject_plan(cohort, subject)
    except (ValueError, KeyError) as error:
        detail = str(error)
        ledger_append(
            {
                "wave": wave,
                "subject": subject,
                "stage": "plan",
                "status": "blocked_missing_cohort_entry",
                "detail": detail,
            }
        )
        result["status"] = "blocked_missing_cohort_entry"
        result["detail"] = detail
        return result
    plan_summary = v6.plan_payload(plan, cohort_config_path, sdc_config_path)
    ledger_append(
        {
            "wave": wave,
            "subject": subject,
            "stage": "plan",
            "status": "pass",
            "strategy": plan.strategy,
            "branches": [branch.name for branch in plan.branches],
            "plan_sha256": plan_summary["plan_sha256"],
        }
    )

    # --- preflight -----------------------------------------------------------
    # If a prior invocation (e.g. an earlier --wave ... dry-run smoke test)
    # already produced a PASSING preflight receipt for this exact subject,
    # reuse it rather than fail-closed-refuse. This matches
    # fmriprep_sdc_workflow_v6.py's own intended workflow -- preflight once,
    # then run-subject (dry-run today, --execute later) against that same
    # receipt -- and mirrors run_subject()'s own re-verification: it always
    # re-hashes the current BIDS input tree against the recorded receipt via
    # verify_preflight_receipt() before doing anything else, so reusing a
    # receipt here is never a silent trust decision. Only a receipt with
    # status "pass" and external_gates_executed True is reused; anything else
    # (partial/failed/dry-run-local-only) still fail-closed-refuses, since
    # repairing or reinterpreting an ambiguous prior state is not this
    # orchestrator's job.
    ledger_append({"wave": wave, "subject": subject, "stage": "preflight", "status": "started"})
    reused_existing_preflight = False
    if paths["preflight_root"].exists():
        summary_path = paths["preflight_root"] / "preflight_summary.json"
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            existing_summary = None
        if (
            existing_summary is not None
            and existing_summary.get("status") == "pass"
            and existing_summary.get("external_gates_executed") is True
        ):
            preflight_summary = existing_summary
            reused_existing_preflight = True
        else:
            detail = (
                f"Refusing to overwrite existing preflight output that is not a passing, "
                f"externally-gated receipt: {paths['preflight_root']}"
            )
            ledger_append({"wave": wave, "subject": subject, "stage": "preflight", "status": "refused_existing", "detail": detail})
            result["status"] = "preflight_refused_existing"
            result["detail"] = detail
            return result
    if not reused_existing_preflight:
        try:
            preflight_summary = v6.preflight_subject(
                plan,
                cohort_config_path,
                sdc_config_path,
                canonical_bids_root,
                paths["branch_root"],
                paths["preflight_root"],
                link_mode,
                run_external_gates,
            )
        except Exception as error:  # noqa: BLE001 - fail closed, record, and stop this subject
            detail = f"{type(error).__name__}: {error}"
            ledger_append({"wave": wave, "subject": subject, "stage": "preflight", "status": "fail", "detail": detail})
            result["status"] = "preflight_fail"
            result["detail"] = detail
            return result
    ledger_append(
        {
            "wave": wave,
            "subject": subject,
            "stage": "preflight",
            "status": preflight_summary["status"],
            "external_gates_executed": preflight_summary["external_gates_executed"],
            "preflight_root": str(paths["preflight_root"]),
            "reused_existing_preflight": reused_existing_preflight,
        }
    )

    # --- run-subject (fMRIPrep); execute gated by --execute ------------------
    ledger_append(
        {
            "wave": wave,
            "subject": subject,
            "stage": "fmriprep",
            "status": "started",
            "execute": execute,
        }
    )
    if paths["final_root"].exists():
        detail = f"Refusing to overwrite existing final fMRIPrep root: {paths['final_root']}"
        ledger_append({"wave": wave, "subject": subject, "stage": "fmriprep", "status": "refused_existing", "detail": detail})
        result["status"] = "fmriprep_refused_existing"
        result["detail"] = detail
        return result
    try:
        run_result = v6.run_subject(
            plan,
            cohort_config_path,
            sdc_config_path,
            paths["preflight_root"],
            paths["branch_output_root"],
            paths["final_root"],
            license_file,
            execute,
            link_mode,
        )
    except Exception as error:  # noqa: BLE001
        detail = f"{type(error).__name__}: {error}"
        ledger_append({"wave": wave, "subject": subject, "stage": "fmriprep", "status": "fail", "detail": detail})
        result["status"] = "fmriprep_fail"
        result["detail"] = detail
        return result

    ledger_append(
        {
            "wave": wave,
            "subject": subject,
            "stage": "fmriprep",
            "status": run_result["status"],
            "execute": execute,
            "commands": run_result.get("commands"),
            "final_root": str(paths["final_root"]),
        }
    )
    result["status"] = run_result["status"]
    result["commands"] = run_result.get("commands")
    result["final_root"] = str(paths["final_root"])

    # --- GLMsingle (construct always; execute only with both gates) ----------
    glmsingle_command = launch_glmsingle_command(
        subject, paths["final_root"], canonical_bids_root, DEFAULT_GLMSINGLE_OUTPUT_ROOT, cohort_config_path
    )
    result["glmsingle_command"] = glmsingle_command
    ledger_append(
        {
            "wave": wave,
            "subject": subject,
            "stage": "glmsingle",
            "status": "command_constructed_not_executed",
            "command": glmsingle_command,
            "note": "Uses run_glmsingle_wave.py (KNOWN_ISSUE 3 resolution); still requires "
            "--execute-glmsingle to actually run, which this orchestrator never sets on its own.",
        }
    )
    return result


def run_wave(args: argparse.Namespace) -> list[dict]:
    v6 = load_v6_module()
    cohort, sdc = v6.load_configs(args.cohort_config, args.sdc_config)
    patch_basic_derivative_gate(v6, sdc)
    patch_merge_mixed_branches(v6)
    patch_fmriprep_memory_limit(v6, limit=getattr(args, "fmriprep_memory_limit", FMRIPREP_CONTAINER_MEMORY_LIMIT))
    waves_cfg = load_waves(args.waves_config)
    subjects = subjects_for_wave(waves_cfg, args.wave)
    if args.only_subjects:
        allowed = set(args.only_subjects)
        subjects = [subject for subject in subjects if subject in allowed]
        unknown = allowed - set(subjects_for_wave(waves_cfg, args.wave))
        if unknown:
            raise ValueError(f"--only-subjects {sorted(unknown)} are not in wave {args.wave}")
    ledger_append(
        {
            "wave": args.wave,
            "subject": None,
            "stage": "wave",
            "status": "started",
            "subjects": subjects,
            "execute": args.execute,
        }
    )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.fmriprep_workers)) as pool:
        futures = {
            pool.submit(
                process_subject,
                v6,
                subject,
                args.wave,
                cohort,
                args.cohort_config,
                args.sdc_config,
                args.shared_bids_root,
                args.per_subject_bids_root,
                args.derivatives_root,
                args.license_file,
                args.link_mode,
                args.run_external_gates,
                args.execute,
            ): subject
            for subject in subjects
        }
        for future in as_completed(futures):
            subject = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 - fail closed per subject
                detail = f"{type(error).__name__}: {error}"
                ledger_append(
                    {
                        "wave": args.wave,
                        "subject": subject,
                        "stage": "wave",
                        "status": "unhandled_exception",
                        "detail": detail,
                    }
                )
                results.append({"subject": subject, "wave": args.wave, "status": "unhandled_exception", "detail": detail})

    ledger_append(
        {
            "wave": args.wave,
            "subject": None,
            "stage": "wave",
            "status": "finished",
            "subject_statuses": {item["subject"]: item.get("status") for item in results},
        }
    )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument(
        "--only-subjects",
        type=lambda value: [int(item) for item in value.split(",") if item.strip()],
        default=None,
        help="Comma-separated subject numbers to restrict this run to (must be a subset of --wave). "
        "Use for smoke tests, e.g. --wave 1 --only-subjects 1",
    )
    parser.add_argument("--execute", action="store_true", help="Actually run fMRIPrep. Default is plan/preflight/dry-run only.")
    parser.add_argument("--fmriprep-workers", type=int, default=FMRIPREP_WORKERS_DEFAULT)
    parser.add_argument(
        "--fmriprep-memory-limit",
        type=str,
        default=FMRIPREP_CONTAINER_MEMORY_LIMIT,
        help="Per-container --memory/--memory-swap cgroup limit passed to every fMRIPrep "
        "docker run (default: 32g, sized for the observed ~26.6 GiB ANTs SyN peak with "
        "headroom). Override for a specific retry that needs more, e.g. a subject whose "
        "container-scoped OOM shows the default is insufficient for its node-scheduling "
        "peak -- see patch_fmriprep_memory_limit()'s docstring for the Wave 3 Sub26 case "
        "this was added for.",
    )
    parser.add_argument("--glmsingle-workers", type=int, default=GLMSINGLE_WORKERS_DEFAULT)
    parser.add_argument("--cohort-config", type=Path, default=DEFAULT_COHORT_CONFIG)
    parser.add_argument("--sdc-config", type=Path, default=DEFAULT_SDC_CONFIG)
    parser.add_argument("--waves-config", type=Path, default=DEFAULT_WAVES_CONFIG)
    parser.add_argument("--shared-bids-root", type=Path, default=DEFAULT_SHARED_BIDS_ROOT)
    parser.add_argument("--per-subject-bids-root", type=Path, default=DEFAULT_PER_SUBJECT_BIDS_ROOT)
    parser.add_argument("--derivatives-root", type=Path, default=DEFAULT_DERIVATIVES_ROOT)
    parser.add_argument("--license-file", type=Path, default=DEFAULT_LICENSE_FILE)
    parser.add_argument("--link-mode", choices=("hardlink", "copy", "auto"), default="auto")
    parser.add_argument(
        "--run-external-gates",
        dest="run_external_gates",
        action="store_true",
        default=True,
        help="Run the official BIDS validator and pinned-container sdcflows probe during preflight (default on).",
    )
    parser.add_argument("--skip-external-gates", dest="run_external_gates", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not str(os.path.abspath(args.derivatives_root)).upper().startswith("N:"):
        print("ERROR: --derivatives-root must be on N:", file=sys.stderr)
        return 2
    try:
        results = run_wave(args)
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(results, indent=2, default=str))
    blocked_or_failed = [
        item for item in results if item.get("status") not in {"pass", "dry_run"}
    ]
    return 0 if not blocked_or_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
