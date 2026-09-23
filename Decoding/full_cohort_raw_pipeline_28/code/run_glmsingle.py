#!/usr/bin/env python3
"""Run the frozen GLMsingle Type-D model for any included cohort subject.

The numerical implementation is intentionally delegated to the immutable
Sub4--6 pilot wrapper.  This full-cohort entry point removes the pilot's
hard-coded subject choices, derives session indicators from BIDS, cross-checks
them against ``config/cohort.json``, and fails closed unless the pinned vendor,
ordered patches, ridge grid, nuisance-PC count, TR, duration, and seed match
``ANALYSIS_FREEZE.json``.

Production outputs are never overwritten or resumed.  ``--validate-only`` is
the safe preflight mode and does not estimate a GLM.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np

from cohort_analysis_common import (
    COHORT_ROOT,
    DEFAULT_COHORT_CONFIG,
    DEFAULT_FREEZE,
    atomic_write_json,
    load_cohort_config,
    load_freeze,
    resolve_session_indicators,
    sha256_file,
    validate_subject_in_freeze,
)


PILOT_SCRIPT = (
    COHORT_ROOT.parent
    / "pilot_raw_pipeline_sub4_6"
    / "code"
    / "run_glmsingle.py"
)
VENDORED_PYTHON_DEPENDENCIES = (
    COHORT_ROOT
    / "vendor"
    / "python"
)
FRACRIDGE_VERSION = "2.0"
FRACRIDGE_RECORD = (
    VENDORED_PYTHON_DEPENDENCIES
    / f"fracridge-{FRACRIDGE_VERSION}.dist-info"
    / "RECORD"
)
# This is the RECORD from the verified local installation used by the pilot.
# Pinning it also pins every wheel-payload path, SHA-256 digest, and size below.
FRACRIDGE_RECORD_SHA256 = (
    "a9ec96dc55b4d215d4dd02caacf2d710e235c650d58c5165af22ed83551b359c"
)


def validate_fracridge_vendor() -> dict:
    """Fail closed unless the restored fracridge 2.0 payload is exact."""
    if not FRACRIDGE_RECORD.is_file():
        raise FileNotFoundError(
            f"Pinned fracridge RECORD is missing: {FRACRIDGE_RECORD}"
        )
    record_sha256 = sha256_file(FRACRIDGE_RECORD)
    if record_sha256 != FRACRIDGE_RECORD_SHA256:
        raise RuntimeError(
            "Pinned fracridge RECORD hash mismatch: "
            f"expected {FRACRIDGE_RECORD_SHA256}, observed {record_sha256}"
        )

    vendor_root = VENDORED_PYTHON_DEPENDENCIES.resolve()
    verified_files: list[dict] = []
    with FRACRIDGE_RECORD.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    for row in rows:
        if len(row) != 3:
            raise RuntimeError(f"Malformed fracridge RECORD row: {row!r}")
        relative, record_digest, record_size = row
        if not record_digest:
            # RECORD itself and interpreter-generated __pycache__ entries are
            # intentionally unhashed in the installed distribution metadata.
            continue
        try:
            algorithm, expected_digest = record_digest.split("=", 1)
        except ValueError as error:
            raise RuntimeError(
                f"Malformed fracridge digest for {relative}: {record_digest}"
            ) from error
        if algorithm != "sha256":
            raise RuntimeError(
                f"Unsupported fracridge RECORD algorithm for {relative}: {algorithm}"
            )
        path = (VENDORED_PYTHON_DEPENDENCIES / Path(relative)).resolve()
        if not path.is_relative_to(vendor_root):
            raise RuntimeError(f"Unsafe fracridge RECORD path: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Pinned fracridge payload is missing: {path}")
        payload = path.read_bytes()
        observed_digest = (
            base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
            .decode("ascii")
            .rstrip("=")
        )
        if observed_digest != expected_digest:
            raise RuntimeError(f"Pinned fracridge payload hash mismatch: {relative}")
        if record_size and len(payload) != int(record_size):
            raise RuntimeError(f"Pinned fracridge payload size mismatch: {relative}")
        verified_files.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if len(verified_files) != 12:
        raise RuntimeError(
            "Pinned fracridge RECORD must contain exactly 12 hashed payload files; "
            f"observed {len(verified_files)}"
        )

    dependency_path = str(VENDORED_PYTHON_DEPENDENCIES)
    if dependency_path not in sys.path:
        sys.path.insert(0, dependency_path)
    existing = sys.modules.get("fracridge")
    if existing is not None:
        existing_file = Path(existing.__file__).resolve()
        if not existing_file.is_relative_to(vendor_root):
            raise RuntimeError(
                "fracridge was imported from outside the pinned vendor before "
                f"preflight: {existing_file}"
            )
    fracridge = importlib.import_module("fracridge")
    module_path = Path(fracridge.__file__).resolve()
    if not module_path.is_relative_to(vendor_root):
        raise RuntimeError(f"fracridge resolved outside the pinned vendor: {module_path}")
    if getattr(fracridge, "__version__", None) != FRACRIDGE_VERSION:
        raise RuntimeError(
            "Pinned fracridge module version mismatch: "
            f"{getattr(fracridge, '__version__', None)!r}"
        )
    distribution = importlib.metadata.distribution("fracridge")
    distribution_path = Path(distribution._path).resolve()
    expected_distribution_path = FRACRIDGE_RECORD.parent.resolve()
    if distribution_path != expected_distribution_path:
        raise RuntimeError(
            "fracridge metadata resolved outside the pinned vendor: "
            f"{distribution_path}"
        )
    if distribution.version != FRACRIDGE_VERSION:
        raise RuntimeError(
            f"Pinned fracridge metadata version mismatch: {distribution.version}"
        )
    return {
        "validated": True,
        "package": "fracridge",
        "version": FRACRIDGE_VERSION,
        "vendor_root": str(vendor_root),
        "module_path": str(module_path),
        "distribution_path": str(distribution_path),
        "record_path": str(FRACRIDGE_RECORD.resolve()),
        "record_sha256": record_sha256,
        "verified_hashed_file_count": len(verified_files),
        "verified_files": verified_files,
    }


def load_pilot_module(path: Path = PILOT_SCRIPT) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Immutable pilot GLMsingle wrapper is missing: {path}")
    # The pilot launch environment supplied fracridge through PYTHONPATH.  Make
    # that pinned, project-local dependency explicit so batch jobs do not rely
    # on an interactive shell's environment.
    if not VENDORED_PYTHON_DEPENDENCIES.is_dir():
        raise FileNotFoundError(
            "Pinned GLMsingle Python dependencies are missing: "
            f"{VENDORED_PYTHON_DEPENDENCIES}"
        )
    fracridge_vendor_validation = validate_fracridge_vendor()
    module_name = "ai_iaps_immutable_pilot_run_glmsingle"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import immutable pilot wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        if error.name == "fracridge":
            raise RuntimeError(
                "GLMsingle runtime dependency fracridge is unavailable. The "
                "project vendor directory currently contains fracridge 2.0 "
                "metadata but must also contain/import the package code before "
                "production launch. Do not run GLMsingle until this gate passes."
            ) from error
        raise
    module._full_cohort_fracridge_vendor_validation = fracridge_vendor_validation
    return module


def validate_pinned_contract(pilot: ModuleType, freeze: dict) -> dict:
    frozen = freeze["glmsingle"]
    observed_patches = [spec[2] for spec in pilot.GLMSINGLE_PATCH_SPECS]
    checks = {
        "commit": pilot.GLMSINGLE_COMMIT == frozen["commit"],
        "ordered_patch_sha256": observed_patches == frozen["ordered_patch_sha256"],
        "fractional_ridge_grid": np.allclose(
            np.asarray(pilot.DEFAULT_FRACS, dtype=float),
            np.asarray(frozen["fractional_ridge_grid"], dtype=float),
            rtol=0,
            atol=1e-12,
        ),
        "tr_seconds": float(pilot.TR) == float(freeze["acquisition"]["tr_seconds"]),
        "stimulus_duration_seconds": float(pilot.STIMDUR)
        == float(freeze["acquisition"]["stimulus_duration_seconds"]),
        "seed": int(pilot.SEED)
        == int(freeze["random_seed_for_matched_nonhistorical_code"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Pilot/freeze GLMsingle contract mismatch: {failed}")
    return checks


def enforce_production_options(args: argparse.Namespace, freeze: dict) -> None:
    if args.nonproduction:
        return
    if args.n_pcs != int(freeze["glmsingle"]["candidate_noise_pcs"]):
        raise ValueError("Production --n-pcs must match ANALYSIS_FREEZE.json")
    if args.fixed_frac is not None or args.fracs is not None:
        raise ValueError("Production must use the frozen full 20-value ridge grid")
    if args.max_voxels is not None:
        raise ValueError("Production cannot use --max-voxels")
    if args.overwrite:
        raise ValueError("Production output overwrite is forbidden")
    if args.output is not None:
        unsafe_tokens = {"failed", "smoke", "coarse", "test"}
        path_parts = {part.lower() for part in args.output.parts}
        unsafe_hits = {
            token
            for token in unsafe_tokens
            if any(token in part for part in path_parts)
        }
        if unsafe_hits:
            raise ValueError(
                "Production destination contains a nonproduction path component: "
                f"{sorted(unsafe_hits)}"
            )


def prepare(
    args: argparse.Namespace,
) -> tuple[ModuleType, dict, dict, list[dict], list[int], str, dict]:
    freeze = load_freeze(args.freeze)
    validate_subject_in_freeze(args.subject, freeze)
    cohort_config = load_cohort_config(args.cohort_config)
    if args.subject not in {
        int(value) for value in cohort_config["cohort"]["included_subjects"]
    }:
        raise ValueError(f"Sub{args.subject} is absent from cohort.json")
    enforce_production_options(args, freeze)

    pilot = load_pilot_module(args.pilot_script)
    fracridge_vendor = pilot._full_cohort_fracridge_vendor_validation
    contract = validate_pinned_contract(pilot, freeze)
    vendor = pilot.validate_glmsingle_vendor()
    records = pilot.discover_run_files(args.fmriprep_root, args.subject, args.space)
    run_n_volumes = pilot.validate_run_n_volumes(records, args.subject)
    indicators, indicator_source = resolve_session_indicators(
        [record["bold"] for record in records], args.subject, cohort_config
    )
    if len(indicators) != int(freeze["acquisition"]["expected_runs_per_subject"]):
        raise RuntimeError("Session-indicator length differs from frozen run count")
    # The immutable implementation reads this mapping inside its run() call.
    pilot.SESSION_INDICATORS[args.subject] = indicators
    expected_volumes = (
        cohort_config.get("subjects", {})
        .get(str(args.subject), {})
        .get("expected_run_n_volumes")
    )
    if expected_volumes is not None:
        expected_volumes = tuple(int(value) for value in expected_volumes)
        if len(expected_volumes) != 10:
            raise ValueError("expected_run_n_volumes must contain 10 values")
        pilot.EXPECTED_RUN_N_VOLUMES[args.subject] = expected_volumes
        # Re-run after installing the external acquisition invariant.
        run_n_volumes = pilot.validate_run_n_volumes(records, args.subject)

    preflight = {
        "status": "validated_not_run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": args.subject,
        "freeze_path": str(args.freeze.resolve()),
        "freeze_sha256": sha256_file(args.freeze),
        "cohort_config_path": str(args.cohort_config.resolve()),
        "cohort_config_sha256": sha256_file(args.cohort_config),
        "pilot_script": str(args.pilot_script.resolve()),
        "pilot_script_sha256": sha256_file(args.pilot_script),
        "pinned_contract": contract,
        "fracridge_vendor_validation": fracridge_vendor,
        "vendor_validation": vendor,
        "session_indicators": indicators,
        "session_indicator_source": indicator_source,
        "run_n_volumes": run_n_volumes,
        "bold_files": [str(record["bold"].resolve()) for record in records],
    }
    return (
        pilot,
        freeze,
        cohort_config,
        records,
        indicators,
        indicator_source,
        preflight,
    )


def run(args: argparse.Namespace) -> None:
    pilot, _, _, _, indicators, indicator_source, preflight = prepare(args)
    if args.preflight_json is not None:
        atomic_write_json(args.preflight_json, preflight)
    if args.validate_only:
        print(json.dumps(preflight, indent=2), flush=True)
        return
    if args.output is None:
        raise ValueError("--output is required unless --validate-only is used")

    delegated = argparse.Namespace(
        subject=args.subject,
        fmriprep_root=args.fmriprep_root,
        bids_root=args.bids_root,
        space=args.space,
        output=args.output,
        chunklen=args.chunklen,
        n_pcs=args.n_pcs,
        fixed_frac=args.fixed_frac,
        fracs=args.fracs,
        max_voxels=args.max_voxels,
        overwrite=args.overwrite,
    )
    pilot.run(delegated)

    provenance_path = args.output / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["full_cohort_entry_point"] = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "freeze_path": preflight["freeze_path"],
        "freeze_sha256": preflight["freeze_sha256"],
        "cohort_config_path": preflight["cohort_config_path"],
        "cohort_config_sha256": preflight["cohort_config_sha256"],
        "session_indicators": indicators,
        "session_indicator_source": indicator_source,
        "production_contract_enforced": not args.nonproduction,
        "fracridge_vendor_validation": preflight[
            "fracridge_vendor_validation"
        ],
    }
    atomic_write_json(provenance_path, provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--fmriprep-root", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--space", default="MNI152NLin6Asym")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--cohort-config", type=Path, default=DEFAULT_COHORT_CONFIG)
    parser.add_argument("--pilot-script", type=Path, default=PILOT_SCRIPT)
    parser.add_argument("--preflight-json", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--chunklen", type=int, default=10000)
    parser.add_argument("--n-pcs", type=int, default=10)
    parser.add_argument("--fixed-frac", type=float)
    parser.add_argument("--fracs", type=float, nargs="+")
    parser.add_argument("--max-voxels", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--nonproduction",
        action="store_true",
        help="Permit explicit smoke-test overrides; never use for cohort results.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if not args.validate_only and args.output is None:
        build_parser().error("--output is required unless --validate-only is used")
    return args


if __name__ == "__main__":
    run(parse_args())
