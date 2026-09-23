#!/usr/bin/env python3
"""Build a corrected, isolated BIDS dataset for the Sub4-6 decoding pilot.

The source NIfTIs were produced from the original DICOMs.  This script only
copies them into a corrected BIDS layout, preserves acquisition metadata, adds
field-map associations, and reconstructs BIDS events from the raw task logs.
It never modifies the source data or the existing fMRIPrep derivatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


RAW_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/LAB_IAPS_AI/rawfMRI/IAPS-DEV")
LOG_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/LAB_IAPS_AI/DataRecording")
STIMULI_CSV = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/stimuli_600trials.csv")
DEFAULT_OUT = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/pilot_raw_pipeline_sub4_6/bids")


@dataclass(frozen=True)
class RunSource:
    subject: int
    session: int
    run: int
    source_dir: str
    pattern: str


RUNS = [
    # Sub4 was acquired on one day. DEV_004_2 is a duplicate subset and is ignored.
    *[
        RunSource(4, 1, run, "DEV_004", f"DEV_004_cmrr_mbep2d_bold_1.80_RUN{run}_20240212171755_*.nii.gz")
        for run in range(1, 11)
    ],
    # Sub5: February runs 1/3/4/5; replacement runs 2/6-10 were acquired in March.
    RunSource(5, 1, 1, "DEV_005", "DEV_005_cmrr_mbep2d_bold_1.80_RUN1_20240220182253_5.nii.gz"),
    RunSource(5, 1, 3, "DEV_005", "DEV_005_cmrr_mbep2d_bold_1.80_RUN3_20240220182253_9.nii.gz"),
    RunSource(5, 1, 4, "DEV_005", "DEV_005_cmrr_mbep2d_bold_1.80_RUN4_20240220182253_11.nii.gz"),
    RunSource(5, 1, 5, "DEV_005", "DEV_005_cmrr_mbep2d_bold_1.80_RUN5_20240220182253_13.nii.gz"),
    RunSource(5, 2, 2, "DEV_005", "DEV_005_cmrr_mbep2d_bold_1.80_RUN2_20240317133129_6.nii.gz"),
    RunSource(5, 2, 6, "DEV_005", "DEV_005_cmrr_mbep2d_bold_1.80_RUN6_20240317133129_8.nii.gz"),
    RunSource(5, 2, 7, "DEV_005_2", "DEV_005_2_cmrr_mbep2d_bold_1.80_RUN7_20240317133129_10.nii.gz"),
    RunSource(5, 2, 8, "DEV_005_2", "DEV_005_2_cmrr_mbep2d_bold_1.80_RUN8_20240317133129_12.nii.gz"),
    RunSource(5, 2, 9, "DEV_005_2", "DEV_005_2_cmrr_mbep2d_bold_1.80_RUN9_20240317133129_14.nii.gz"),
    RunSource(5, 2, 10, "DEV_005_2", "DEV_005_2_cmrr_mbep2d_bold_1.80_RUN10_20240317133129_16.nii.gz"),
    # Sub6 has two real sessions.
    *[
        RunSource(6, 1, run, "DEV_006", f"DEV_006_cmrr_mbep2d_bold_1.80_RUN{run}_20240229155804_{3 + 2 * run}.nii.gz")
        for run in range(1, 6)
    ],
    *[
        RunSource(6, 2, run, "DEV_006_2", f"DEV_006_2_cmrr_mbep2d_bold_1.80_RUN{run}_20240308080825_{2 * run - 7}.nii.gz")
        for run in range(6, 11)
    ],
]


ANAT = [
    (4, None, "DEV_004", "DEV_004_t1_mprage_sag_p2_iso_20240212171755_40.nii.gz"),
    (5, None, "DEV_005_2", "DEV_005_2_t1_mprage_sag_p2_iso_20240317133129_5.nii.gz"),
    (6, 1, "DEV_006", "DEV_006_t1_mprage_sag_p2_iso_20240229155804_22.nii.gz"),
    (6, 2, "DEV_006_2", "DEV_006_2_t1_mprage_sag_p2_iso_20240308080825_22.nii.gz"),
]


FMAPS = [
    (4, 1, "DEV_004", "DEV_004_fMRI-DistMap_AP_20240212171755_38.nii.gz", "AP"),
    (4, 1, "DEV_004", "DEV_004_fMRI-DistMap_PA_20240212171755_36.nii.gz", "PA"),
    (5, 2, "DEV_005_2", "DEV_005_2_fMRI-DistMap_AP_20240317133129_23.nii.gz", "AP"),
    (5, 2, "DEV_005_2", "DEV_005_2_fMRI-DistMap_PA_20240317133129_21.nii.gz", "PA"),
    (6, 1, "DEV_006", "DEV_006_fMRI-DistMap_AP_20240229155804_20.nii.gz", "AP"),
    (6, 1, "DEV_006", "DEV_006_fMRI-DistMap_PA_20240229155804_18.nii.gz", "PA"),
    (6, 2, "DEV_006_2", "DEV_006_2_fMRI-DistMap_AP_20240308080825_20.nii.gz", "AP"),
    (6, 2, "DEV_006_2", "DEV_006_2_fMRI-DistMap_PA_20240308080825_18.nii.gz", "PA"),
]


def exact_source(source_dir: str, pattern: str) -> Path:
    matches = sorted((RAW_ROOT / source_dir).glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one source for {source_dir}/{pattern}, found {matches}")
    return matches[0]


def sidecar(path: Path) -> Path:
    name = path.name
    if not name.endswith(".nii.gz"):
        raise ValueError(path)
    return path.with_name(name[:-7] + ".json")


def copy_nifti_and_json(
    source: Path,
    destination: Path,
    updates: dict | None = None,
    link_nifti: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link_nifti:
        if not destination.exists():
            destination.symlink_to(source)
    else:
        # SMB-backed workspace paths may reject timestamp preservation from WSL.
        # Copy file contents only; provenance is recorded in the manifest. A
        # size-matched file makes interrupted staging runs safely resumable.
        if not destination.exists() or destination.stat().st_size != source.stat().st_size:
            shutil.copyfile(source, destination)
    source_json = sidecar(source)
    if not source_json.exists():
        raise FileNotFoundError(source_json)
    metadata = json.loads(source_json.read_text(encoding="utf-8"))
    if updates:
        metadata.update(updates)
    destination_json = sidecar(destination)
    destination_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def extract_events(subject: int, run: int, expected: pd.DataFrame) -> pd.DataFrame:
    log_path = LOG_ROOT / f"Sub{subject}" / "LogFiles" / f"Run{run:02d}.mat"
    data_log = loadmat(log_path, simplify_cells=True)["dataLog"]
    rows = [row for row in data_log[1:] if isinstance(row[1], str)]
    fix_rows = [row for row in rows if row[1] == "FixOn"]
    stim_rows = [row for row in rows if row[1] == "Stim on"]
    if len(fix_rows) != 1 or len(stim_rows) != 60:
        raise RuntimeError(f"Unexpected log structure in {log_path}")
    fix_onset = float(fix_rows[0][3])
    raw_names = [str(row[2]) for row in stim_rows]
    expected_names = expected.sort_values("OriginalOrder")["img"].tolist()
    if raw_names != expected_names:
        raise RuntimeError(f"Stimulus order mismatch for Sub{subject} Run{run:02d}")
    onsets = np.asarray([float(row[3]) - fix_onset for row in stim_rows])
    if not np.all(np.isfinite(onsets)) or not np.all(np.diff(onsets) > 0):
        raise RuntimeError(f"Invalid onset timing in {log_path}")
    events = expected.sort_values("OriginalOrder").copy()
    events.insert(0, "onset", onsets)
    events.insert(1, "duration", 3.0)
    events["trial_type"] = events["group"]
    events["image_id"] = events["img"].str.replace(r"\.jpg$", "", regex=True)
    events["source"] = np.where(events["AI"].str.upper().eq("YES"), "ai", "natural")
    events["valence"] = events["emotion_type"]
    return events[["onset", "duration", "trial_type", "image_id", "source", "valence", "OriginalOrder"]]


def build(output: Path, link_niftis: bool = False, refresh: bool = False) -> None:
    if (output / "code" / "pilot_source_manifest.tsv").exists() and not refresh:
        raise RuntimeError(f"Refusing to overwrite completed staging directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "AI-IAPS corrected Sub4-6 raw-data decoding pilot",
                "BIDSVersion": "1.10.0",
                "DatasetType": "raw",
                "Authors": ["Yujun Chen"],
                "GeneratedBy": [{"Name": "build_pilot_bids.py", "Version": "1.0"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / ".bidsignore").write_text(
        "pilot_source_manifest.tsv\n",
        encoding="utf-8",
    )
    with (output / "participants.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["participant_id"])
        for subject in (4, 5, 6):
            writer.writerow([f"sub-{subject:02d}"])
    (output / "task-iaps_events.json").write_text(
        json.dumps(
            {
                "onset": {"Description": "Stimulus onset relative to the scanner trigger", "Units": "s"},
                "duration": {"Description": "Image presentation duration", "Units": "s"},
                "trial_type": {"Description": "Six-way source-by-valence condition"},
                "image_id": {"Description": "Stimulus filename without the .jpg extension"},
                "source": {"Description": "Natural or AI-edited image source"},
                "valence": {"Description": "Pleasant, neutral, or unpleasant content"},
                "OriginalOrder": {"Description": "Zero-based trial order within run"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    stimuli = pd.read_csv(STIMULI_CSV)
    intended: dict[tuple[int, int], list[str]] = {}
    manifest_rows: list[dict] = []
    for item in RUNS:
        source = exact_source(item.source_dir, item.pattern)
        dest_rel = Path(f"sub-{item.subject:02d}") / f"ses-{item.session:02d}" / "func" / (
            f"sub-{item.subject:02d}_ses-{item.session:02d}_task-iaps_run-{item.run:02d}_bold.nii.gz"
        )
        fmap_id = f"sub-{item.subject:02d}_ses-{item.session:02d}_fmap0"
        has_measured_fmap = not (item.subject == 5 and item.session == 1)
        updates = {"TaskName": "iaps"}
        if has_measured_fmap:
            updates["B0FieldSource"] = fmap_id
        copy_nifti_and_json(source, output / dest_rel, updates, link_niftis)
        intended.setdefault((item.subject, item.session), []).append(f"bids::{dest_rel.as_posix()}")

        expected = stimuli[stimuli["run"] == item.run]
        events = extract_events(item.subject, item.run, expected)
        events_path = (output / dest_rel).with_name((output / dest_rel).name.replace("_bold.nii.gz", "_events.tsv"))
        events.to_csv(events_path, sep="\t", index=False, float_format="%.6f")
        manifest_rows.append(
            {
                "subject": item.subject,
                "session": item.session,
                "run": item.run,
                "source_nifti": str(source),
                "bids_nifti": str(dest_rel),
                "measured_sdc": has_measured_fmap,
            }
        )

    for subject, session, source_dir, source_name in ANAT:
        source = exact_source(source_dir, source_name)
        if session is None:
            dest_rel = Path(f"sub-{subject:02d}") / "anat" / f"sub-{subject:02d}_T1w.nii.gz"
        else:
            dest_rel = Path(f"sub-{subject:02d}") / f"ses-{session:02d}" / "anat" / (
                f"sub-{subject:02d}_ses-{session:02d}_T1w.nii.gz"
            )
        copy_nifti_and_json(source, output / dest_rel, link_nifti=link_niftis)

    for subject, session, source_dir, source_name, direction in FMAPS:
        source = exact_source(source_dir, source_name)
        dest_rel = Path(f"sub-{subject:02d}") / f"ses-{session:02d}" / "fmap" / (
            f"sub-{subject:02d}_ses-{session:02d}_dir-{direction}_epi.nii.gz"
        )
        copy_nifti_and_json(
            source,
            output / dest_rel,
            {
                "B0FieldIdentifier": f"sub-{subject:02d}_ses-{session:02d}_fmap0",
                "IntendedFor": intended[(subject, session)],
            },
            link_niftis,
        )

    manifest = pd.DataFrame(manifest_rows).sort_values(["subject", "run"])
    (output / "code").mkdir(exist_ok=True)
    manifest.to_csv(output / "code" / "pilot_source_manifest.tsv", sep="\t", index=False)
    (output / "README").write_text(
        "Corrected staging BIDS for the AI-IAPS Sub4-6 decoding pilot.\n"
        "Sub4 is one actual acquisition session. Sub5 sessions follow acquisition date;\n"
        "its February runs have no measured field map and must use fieldmapless SDC.\n"
        "Sub6 has two actual acquisition sessions. Source data are never modified.\n",
        encoding="utf-8",
    )
    print(f"Built {output}")
    print(manifest.groupby(["subject", "session", "measured_sdc"]).size())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--link-niftis",
        action="store_true",
        help="Symlink source NIfTIs (useful for a local WSL staging tree).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh sidecars/events in a completed tree; size-matched NIfTIs are not recopied.",
    )
    args = parser.parse_args()
    build(args.output, link_niftis=args.link_niftis, refresh=args.refresh)


if __name__ == "__main__":
    main()
