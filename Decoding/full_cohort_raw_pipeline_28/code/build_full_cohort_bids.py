#!/usr/bin/env python3
"""Build a session-correct BIDS tree from an approved, exact source manifest.

This command performs a complete audit before creating the output directory.
It contains no source globs and refuses proposed, ambiguous, missing, MoCoSeries,
cross-session, onset-mismatched, or source-size-changed inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from cohortlib import (
    SELECTION_COLUMNS,
    audit_selection,
    bids_bold_relpath,
    extract_events,
    iter_expected_assets,
    iter_expected_runs,
    load_config,
    load_stimuli,
    read_json,
    safe_relative_path,
    sha256_file,
    sidecar_for_nifti,
    stimuli_identity,
    verify_file_identity,
    write_tsv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "cohort.json"
DEFAULT_SELECTION = ROOT / "manifests" / "source_selection.tsv"
DEFAULT_OUTPUT = ROOT / "bids_remaining25"


def _subjects_arg(values: list[int] | None) -> Iterable[int] | None:
    return values if values else None


GENERATED_FILE_COLUMNS = [
    "asset_id",
    "file_type",
    "bids_relpath",
    "size_bytes",
    "sha256",
]


def _manifest_path(root: Path, row: Mapping[str, str], field: str) -> Path:
    path, problem = safe_relative_path(root, row[field])
    if problem or path is None:
        raise RuntimeError(f"Unsafe {field} for {row['asset_id']}: {problem}")
    return path


def _source_path(raw_root: Path, row: Mapping[str, str]) -> Path:
    return _manifest_path(raw_root, row, "source_relpath")


def _write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _copy_frozen_nifti(
    source: Path,
    destination: Path,
    expected_size: int | str,
    expected_sha256: str,
    *,
    link_niftis: bool,
) -> None:
    if link_niftis:
        verify_file_identity(
            source,
            expected_size,
            expected_sha256,
            label="source NIfTI immediately before symlinking",
        )
        destination.symlink_to(source)
        return
    digest = hashlib.sha256()
    copied_size = 0
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        while chunk := source_handle.read(8 * 1024 * 1024):
            destination_handle.write(chunk)
            digest.update(chunk)
            copied_size += len(chunk)
    if copied_size != int(expected_size) or digest.hexdigest().casefold() != expected_sha256.casefold():
        raise RuntimeError(
            f"Source NIfTI changed while copying {source}: "
            f"expected size/hash {expected_size}/{expected_sha256}, "
            f"copied {copied_size}/{digest.hexdigest()}"
        )


def _read_frozen_source_json(row: Mapping[str, str], raw_root: Path) -> dict:
    source = _source_path(raw_root, row)
    sidecar = _manifest_path(raw_root, row, "source_json_relpath")
    if sidecar.resolve() != sidecar_for_nifti(source).resolve():
        raise RuntimeError(f"Frozen JSON is not the selected NIfTI sidecar for {row['asset_id']}")
    return read_json(
        sidecar,
        expected_size=row["source_json_size_bytes"],
        expected_sha256=row["source_json_sha256"],
        label=f"source JSON for {row['asset_id']}",
    )


def _materialize_nifti_and_sidecar(
    row: Mapping[str, str],
    raw_root: Path,
    destination: Path,
    metadata_updates: Mapping | None,
    *,
    link_niftis: bool,
) -> Path:
    source = _source_path(raw_root, row)
    metadata = _read_frozen_source_json(row, raw_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Refusing to replace staged file: {destination}")
    _copy_frozen_nifti(
        source,
        destination,
        row["source_size_bytes"],
        row["source_sha256"],
        link_niftis=link_niftis,
    )
    if metadata_updates:
        metadata.update(metadata_updates)
    destination_sidecar = sidecar_for_nifti(destination)
    _write_json(destination_sidecar, metadata)
    return destination_sidecar


def _record_generated(
    rows: list[dict[str, object]],
    staging: Path,
    path: Path,
    *,
    file_type: str,
    asset_id: str = "",
) -> dict[str, object]:
    identity = {
        "asset_id": asset_id,
        "file_type": file_type,
        "bids_relpath": path.relative_to(staging).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    rows.append(identity)
    return identity


def _session_key(subject: int, session: int) -> str:
    return f"sub-{subject:02d}_ses-{session:02d}"


def build_bids(
    config: Mapping,
    rows: list[dict[str, str]],
    raw_root: Path,
    log_root: Path,
    stimuli_csv: Path,
    output: Path,
    subjects: Iterable[int] | None,
    *,
    link_niftis: bool,
    config_path: Path,
    selection_path: Path,
) -> None:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"Refusing to overwrite BIDS output: {output}")
    staging = output.with_name(output.name + ".staging")
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"Refusing to reuse abandoned staging tree: {staging}")
    selected_subjects = sorted(
        set(int(row["subject"]) for row in rows if row["asset_type"] == "bold")
    )
    run_plan = {
        run.asset_id: run for run in iter_expected_runs(config, subjects)
    }
    asset_plan = {
        asset.asset_id: asset for asset in iter_expected_assets(config, subjects)
    }
    row_by_id = {row["asset_id"]: row for row in rows}
    frozen_stimuli = stimuli_identity(config)
    stimuli = load_stimuli(
        stimuli_csv,
        expected_size=frozen_stimuli["size_bytes"],
        expected_sha256=frozen_stimuli["sha256"],
    )
    try:
        staging.mkdir(parents=True)
        generated_files: list[dict[str, object]] = []
        _write_json(
            staging / "dataset_description.json",
            {
                "Name": "AI-IAPS full-cohort raw rerun: remaining 25 subjects",
                "BIDSVersion": "1.10.0",
                "DatasetType": "raw",
                "Authors": ["Yujun Chen"],
                "GeneratedBy": [
                    {
                        "Name": "build_full_cohort_bids.py",
                        "Version": "1.0.0",
                        "Description": "Exact approved-manifest build; no source globbing",
                    }
                ],
            },
        )
        _record_generated(
            generated_files,
            staging,
            staging / "dataset_description.json",
            file_type="dataset_description_json",
        )
        _write_json(
            staging / "task-iaps_events.json",
            {
                "onset": {"Description": "Stimulus onset relative to scanner trigger", "Units": "s"},
                "duration": {"Description": "Image presentation duration", "Units": "s"},
                "trial_type": {"Description": "Six-way source-by-valence condition"},
                "image_id": {"Description": "Stimulus filename without .jpg"},
                "source": {"Description": "Natural or AI-edited image source"},
                "valence": {"Description": "Pleasant, neutral, or unpleasant content"},
                "OriginalOrder": {"Description": "Zero-based within-run trial order"},
            },
        )
        _record_generated(
            generated_files,
            staging,
            staging / "task-iaps_events.json",
            file_type="events_schema_json",
        )
        write_tsv(
            staging / "participants.tsv",
            [{"participant_id": f"sub-{subject:02d}"} for subject in selected_subjects],
            ["participant_id"],
        )

        intended_for: dict[tuple[int, int], list[str]] = {}
        built_manifest: list[dict[str, object]] = []
        for asset_id, run in sorted(
            run_plan.items(), key=lambda item: (item[1].subject, item[1].experimental_run)
        ):
            row = row_by_id[asset_id]
            dest_rel = bids_bold_relpath(run)
            fmap_id = f"{_session_key(run.subject, run.session)}_fmap0"
            updates: dict[str, object] = {"TaskName": "iaps"}
            if run.sdc_mode in {"pepolar", "gre"}:
                updates["B0FieldSource"] = fmap_id
            generated_sidecar = _materialize_nifti_and_sidecar(
                row,
                raw_root,
                staging / dest_rel,
                updates,
                link_niftis=link_niftis,
            )
            json_identity = _record_generated(
                generated_files,
                staging,
                generated_sidecar,
                file_type="image_sidecar_json",
                asset_id=asset_id,
            )
            intended_for.setdefault((run.subject, run.session), []).append(
                f"bids::{dest_rel.as_posix()}"
            )
            log_path = _manifest_path(log_root, row, "onset_mat_relpath")
            events = extract_events(
                log_path,
                stimuli[stimuli["run"] == run.experimental_run],
                expected_size=row["onset_mat_size_bytes"],
                expected_sha256=row["onset_mat_sha256"],
            )
            events_path = (staging / dest_rel).with_name(
                (staging / dest_rel).name.replace("_bold.nii.gz", "_events.tsv")
            )
            events.to_csv(events_path, sep="\t", index=False, float_format="%.6f")
            events_identity = _record_generated(
                generated_files,
                staging,
                events_path,
                file_type="events_tsv",
                asset_id=asset_id,
            )
            built_manifest.append(
                {
                    "asset_id": asset_id,
                    "source_relpath": row["source_relpath"],
                    "bids_relpath": dest_rel.as_posix(),
                    "subject": run.subject,
                    "session": run.session,
                    "experimental_run": run.experimental_run,
                    "source_run_label": run.source_run_label,
                    "sdc_mode": run.sdc_mode,
                    "source_sha256": row["source_sha256"],
                    "source_json_relpath": row["source_json_relpath"],
                    "source_json_sha256": row["source_json_sha256"],
                    "onset_mat_relpath": row["onset_mat_relpath"],
                    "onset_mat_sha256": row["onset_mat_sha256"],
                    "bids_json_relpath": json_identity["bids_relpath"],
                    "bids_json_sha256": json_identity["sha256"],
                    "bids_events_relpath": events_identity["bids_relpath"],
                    "bids_events_sha256": events_identity["sha256"],
                }
            )

        for asset_id, asset in sorted(asset_plan.items()):
            if asset.asset_type != "t1w":
                continue
            row = row_by_id[asset_id]
            dest_rel = (
                Path(f"sub-{asset.subject:02d}")
                / f"ses-{asset.session:02d}"
                / "anat"
                / f"sub-{asset.subject:02d}_ses-{asset.session:02d}_T1w.nii.gz"
            )
            generated_sidecar = _materialize_nifti_and_sidecar(
                row,
                raw_root,
                staging / dest_rel,
                None,
                link_niftis=link_niftis,
            )
            json_identity = _record_generated(
                generated_files,
                staging,
                generated_sidecar,
                file_type="image_sidecar_json",
                asset_id=asset_id,
            )
            built_manifest.append(
                {
                    "asset_id": asset_id,
                    "source_relpath": row["source_relpath"],
                    "bids_relpath": dest_rel.as_posix(),
                    "subject": asset.subject,
                    "session": asset.session,
                    "experimental_run": "",
                    "source_run_label": "",
                    "sdc_mode": asset.sdc_mode,
                    "source_sha256": row["source_sha256"],
                    "source_json_relpath": row["source_json_relpath"],
                    "source_json_sha256": row["source_json_sha256"],
                    "onset_mat_relpath": "",
                    "onset_mat_sha256": "",
                    "bids_json_relpath": json_identity["bids_relpath"],
                    "bids_json_sha256": json_identity["sha256"],
                    "bids_events_relpath": "",
                    "bids_events_sha256": "",
                }
            )

        fmap_assets: dict[tuple[int, int], dict[str, object]] = {}
        for asset in asset_plan.values():
            if asset.asset_type == "fmap":
                fmap_assets.setdefault((asset.subject, asset.session), {})[asset.fmap_role] = asset
        for (subject, session), role_assets in sorted(fmap_assets.items()):
            exemplar = next(iter(role_assets.values()))
            fmap_id = f"{_session_key(subject, session)}_fmap0"
            intended = intended_for[(subject, session)]
            if exemplar.sdc_mode == "pepolar":
                destinations = {
                    "epi_ap": f"sub-{subject:02d}_ses-{session:02d}_dir-AP_epi.nii.gz",
                    "epi_pa": f"sub-{subject:02d}_ses-{session:02d}_dir-PA_epi.nii.gz",
                }
                for role, filename in destinations.items():
                    asset = role_assets[role]
                    row = row_by_id[asset.asset_id]
                    dest_rel = Path(f"sub-{subject:02d}") / f"ses-{session:02d}" / "fmap" / filename
                    generated_sidecar = _materialize_nifti_and_sidecar(
                        row,
                        raw_root,
                        staging / dest_rel,
                        {"B0FieldIdentifier": fmap_id, "IntendedFor": intended},
                        link_niftis=link_niftis,
                    )
                    json_identity = _record_generated(
                        generated_files,
                        staging,
                        generated_sidecar,
                        file_type="image_sidecar_json",
                        asset_id=asset.asset_id,
                    )
                    built_manifest.append(
                        {
                            "asset_id": asset.asset_id,
                            "source_relpath": row["source_relpath"],
                            "bids_relpath": dest_rel.as_posix(),
                            "subject": subject,
                            "session": session,
                            "experimental_run": "",
                            "source_run_label": "",
                            "sdc_mode": "pepolar",
                            "source_sha256": row["source_sha256"],
                            "source_json_relpath": row["source_json_relpath"],
                            "source_json_sha256": row["source_json_sha256"],
                            "onset_mat_relpath": "",
                            "onset_mat_sha256": "",
                            "bids_json_relpath": json_identity["bids_relpath"],
                            "bids_json_sha256": json_identity["sha256"],
                            "bids_events_relpath": "",
                            "bids_events_sha256": "",
                        }
                    )
            elif exemplar.sdc_mode == "gre":
                magnitude1_asset = role_assets["magnitude1"]
                magnitude2_asset = role_assets["magnitude2"]
                phase_asset = role_assets["phasediff"]
                echo1 = float(
                    _read_frozen_source_json(row_by_id[magnitude1_asset.asset_id], raw_root)["EchoTime"]
                )
                echo2 = float(
                    _read_frozen_source_json(row_by_id[magnitude2_asset.asset_id], raw_root)["EchoTime"]
                )
                role_destinations = {
                    "magnitude1": f"sub-{subject:02d}_ses-{session:02d}_magnitude1.nii.gz",
                    "magnitude2": f"sub-{subject:02d}_ses-{session:02d}_magnitude2.nii.gz",
                    "phasediff": f"sub-{subject:02d}_ses-{session:02d}_phasediff.nii.gz",
                }
                for role, filename in role_destinations.items():
                    asset = role_assets[role]
                    row = row_by_id[asset.asset_id]
                    dest_rel = Path(f"sub-{subject:02d}") / f"ses-{session:02d}" / "fmap" / filename
                    updates: dict[str, object] = {
                        "B0FieldIdentifier": fmap_id,
                        "IntendedFor": intended,
                    }
                    if role == "phasediff":
                        updates.update({"EchoTime1": echo1, "EchoTime2": echo2})
                    generated_sidecar = _materialize_nifti_and_sidecar(
                        row,
                        raw_root,
                        staging / dest_rel,
                        updates,
                        link_niftis=link_niftis,
                    )
                    json_identity = _record_generated(
                        generated_files,
                        staging,
                        generated_sidecar,
                        file_type="image_sidecar_json",
                        asset_id=asset.asset_id,
                    )
                    built_manifest.append(
                        {
                            "asset_id": asset.asset_id,
                            "source_relpath": row["source_relpath"],
                            "bids_relpath": dest_rel.as_posix(),
                            "subject": subject,
                            "session": session,
                            "experimental_run": "",
                            "source_run_label": "",
                            "sdc_mode": "gre",
                            "source_sha256": row["source_sha256"],
                            "source_json_relpath": row["source_json_relpath"],
                            "source_json_sha256": row["source_json_sha256"],
                            "onset_mat_relpath": "",
                            "onset_mat_sha256": "",
                            "bids_json_relpath": json_identity["bids_relpath"],
                            "bids_json_sha256": json_identity["sha256"],
                            "bids_events_relpath": "",
                            "bids_events_sha256": "",
                        }
                    )

        code_dir = staging / "code"
        code_dir.mkdir()
        write_tsv(code_dir / "approved_source_selection.tsv", rows, SELECTION_COLUMNS)
        write_tsv(code_dir / "built_source_manifest.tsv", built_manifest)
        generated_manifest_path = code_dir / "generated_bids_file_manifest.tsv"
        write_tsv(
            generated_manifest_path,
            generated_files,
            GENERATED_FILE_COLUMNS,
        )
        _write_json(
            code_dir / "build_provenance.json",
            {
                "config_sha256": sha256_file(config_path),
                "selection_sha256": sha256_file(selection_path),
                "builder_sha256": sha256_file(Path(__file__)),
                "stimuli_csv_size_bytes": frozen_stimuli["size_bytes"],
                "stimuli_csv_sha256": frozen_stimuli["sha256"],
                "subject_count": len(selected_subjects),
                "subjects": selected_subjects,
                "bold_run_count": len(run_plan),
                "asset_count": len(built_manifest),
                "generated_bids_json_event_count": len(generated_files),
                "generated_bids_file_manifest_sha256": sha256_file(generated_manifest_path),
                "built_source_manifest_sha256": sha256_file(code_dir / "built_source_manifest.tsv"),
                "link_niftis": link_niftis,
                "staging_pid": os.getpid(),
                "note": "Every source was revalidated before staging; official BIDS validation remains a mandatory downstream gate.",
            },
        )
        (staging / "README").write_text(
            "Session-correct AI-IAPS BIDS staging built from an approved exact-path manifest.\n"
            "Do not run fMRIPrep until the official BIDS validator passes and every measured\n"
            "fieldmap association is confirmed in the fMRIPrep report.\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        print(
            f"Build failed. Diagnostic staging tree, if created, was preserved at {staging}",
            file=sys.stderr,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--stimuli-csv", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subject", type=int, action="append")
    parser.add_argument("--link-niftis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_root = args.raw_root or Path(config["paths"]["raw_root"])
    log_root = args.log_root or Path(config["paths"]["log_root"])
    stimuli_csv = args.stimuli_csv or Path(config["paths"]["stimuli_csv"])
    subjects = _subjects_arg(args.subject)
    rows, issues = audit_selection(
        config,
        args.selection,
        raw_root,
        log_root,
        stimuli_csv,
        subjects,
        require_approved=True,
        check_onsets=True,
        verify_hashes=True,
    )
    counts = Counter(issue.severity for issue in issues)
    for issue in issues:
        stream = sys.stderr if issue.severity == "error" else sys.stdout
        print(f"{issue.severity.upper()} {issue.code} {issue.asset_id}: {issue.message}", file=stream)
    if counts.get("error", 0):
        raise SystemExit(
            f"FAIL-CLOSED: {counts['error']} source/onset errors; no BIDS output was created"
        )
    print(
        f"AUDIT_PASS rows={len(rows)} warnings={counts.get('warning', 0)} "
        f"subjects={sorted(set(int(row['subject']) for row in rows))}"
    )
    if args.dry_run:
        print("DRY_RUN_PASS: production build was not started")
        return
    build_bids(
        config,
        rows,
        raw_root,
        log_root,
        stimuli_csv,
        args.output,
        subjects,
        link_niftis=args.link_niftis,
        config_path=args.config,
        selection_path=args.selection,
    )
    print(f"BUILD_PASS: {args.output}")


if __name__ == "__main__":
    main()
