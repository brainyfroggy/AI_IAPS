from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from cohortlib import (  # noqa: E402
    SELECTION_COLUMNS,
    audit_selection,
    bids_bold_relpath,
    extract_events,
    iter_expected_assets,
    iter_expected_runs,
    load_config,
    write_tsv,
)
from build_full_cohort_bids import build_bids  # noqa: E402


PRODUCTION_CONFIG = Path(__file__).resolve().parents[1] / "config" / "cohort.json"


def minimal_config(subject: int, sessions: list[dict], log_template: str = "Run{run:02d}.mat") -> dict:
    return {
        "cohort": {"remaining_subjects": [subject]},
        "subjects": {
            str(subject): {
                "log_filename_template": log_template,
                "sessions": sessions,
            }
        },
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_stimuli(config: dict, stimuli: Path) -> dict:
    config["paths"] = {"stimuli_csv": str(stimuli)}
    config["input_identities"] = {
        "stimuli_600trials_csv": {
            "path": str(stimuli),
            "size_bytes": stimuli.stat().st_size,
            "sha256": sha256(stimuli),
        }
    }
    return config


def create_source(raw_root: Path, relpath: str, metadata: dict, payload: bytes = b"nifti") -> int:
    source = raw_root.joinpath(*Path(relpath).parts)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    sidecar = source.with_name(source.name[:-7] + ".json")
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    return len(payload)


def selection_row(
    asset,
    relpath: str,
    size: int,
    raw_root: Path,
    log_root: Path,
    status: str = "approved",
) -> dict[str, object]:
    source = raw_root.joinpath(*Path(relpath).parts)
    sidecar = source.with_name(source.name[:-7] + ".json")
    sidecar_relpath = sidecar.relative_to(raw_root).as_posix()
    if asset.asset_type == "bold":
        onset = log_root.joinpath(*Path(asset.log_relpath).parts)
        onset.parent.mkdir(parents=True, exist_ok=True)
        if not onset.exists():
            onset.write_bytes(b"test onset identity")
        onset_size = onset.stat().st_size
        onset_hash = sha256(onset)
    else:
        onset_size = ""
        onset_hash = ""
    return {
        "asset_id": asset.asset_id,
        "asset_type": asset.asset_type,
        "subject": asset.subject,
        "session": asset.session,
        "experimental_run": asset.experimental_run if asset.experimental_run is not None else "",
        "source_run_label": asset.source_run_label if asset.source_run_label is not None else "",
        "source_relpath": relpath,
        "source_json_relpath": sidecar_relpath,
        "log_relpath": asset.log_relpath,
        "onset_mat_relpath": asset.log_relpath,
        "sdc_mode": asset.sdc_mode,
        "fmap_role": asset.fmap_role,
        "approval_status": status,
        "source_size_bytes": size,
        "source_sha256": sha256(source),
        "source_json_size_bytes": sidecar.stat().st_size,
        "source_json_sha256": sha256(sidecar),
        "onset_mat_size_bytes": onset_size,
        "onset_mat_sha256": onset_hash,
        "notes": "unit test",
    }


class ProductionPlanTests(unittest.TestCase):
    def test_exact_cohort_counts_and_all_scanner_label_offsets(self) -> None:
        config = load_config(PRODUCTION_CONFIG)
        runs = list(iter_expected_runs(config))
        assets = list(iter_expected_assets(config))
        self.assertEqual(len(runs), 250)
        self.assertEqual(len(assets), 394)
        observed = {
            (run.subject, run.experimental_run): run.source_run_label
            for run in runs
        }
        self.assertEqual([observed[(11, run)] for run in range(7, 11)], [8, 9, 10, 11])
        self.assertEqual([observed[(12, run)] for run in range(1, 7)], [1, 3, 4, 5, 6, 7])
        self.assertEqual([observed[(16, run)] for run in range(7, 11)], [7, 8, 10, 11])
        self.assertEqual([observed[(17, run)] for run in range(1, 7)], [1, 3, 4, 5, 6, 7])
        sub1 = next(run for run in runs if run.subject == 1 and run.experimental_run == 1)
        sub2 = next(run for run in runs if run.subject == 2 and run.experimental_run == 1)
        self.assertEqual(sub1.log_relpath, "Sub1/LogFiles/Run1.mat")
        self.assertEqual(sub2.log_relpath, "Sub2/LogFiles/Run01.mat")
        self.assertEqual(
            config["task"]["acquisition_geometry"]["native_nifti_voxel_spacing_mm"],
            [1.7966, 1.7966, 2.25],
        )
        flags = {
            row["asset_id"]: row["observed_volumes"]
            for row in config["task"]["raw_bold_volume_qc"]["flagged_runs"]
        }
        self.assertEqual(
            flags,
            {
                "sub-29_ses-01_run-06_bold": 216,
                "sub-30_ses-01_run-06_bold": 210,
            },
        )


class StructuralAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.logs = self.root / "logs"
        self.stimuli = self.root / "stimuli.csv"
        self.raw.mkdir()
        self.logs.mkdir()
        pd.DataFrame(
            columns=["run", "OriginalOrder", "img", "group", "AI", "emotion_type"]
        ).to_csv(self.stimuli, index=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _offset_fixture(self, status: str = "approved", moco: bool = False):
        config = minimal_config(
            16,
            [
                {
                    "session": 2,
                    "source_dir": "DEV_016_2",
                    "experimental_runs": [9],
                    "source_run_overrides": {"9": 10},
                    "sdc_mode": "syn",
                }
            ],
        )
        freeze_stimuli(config, self.stimuli)
        assets = list(iter_expected_assets(config))
        rows = []
        for asset in assets:
            if asset.asset_type == "bold":
                # Filename encodes the logical run while JSON preserves scanner label RUN10.
                rel = "DEV_016_2/DICOM/25061322/source_filename_RUN9_series12.nii.gz"
                metadata = {
                    "SeriesDescription": "MoCoSeries" if moco else "cmrr_mbep2d_bold 1.80 RUN10",
                    "ProtocolName": "cmrr_mbep2d_bold 1.80 RUN10",
                    "SeriesNumber": 12,
                    "ImageType": ["ORIGINAL", "PRIMARY", "FMRI", "NONE"],
                }
            else:
                rel = "DEV_016_2/DICOM/25061322/t1_mprage_sag_p2_iso.nii.gz"
                metadata = {
                    "SeriesDescription": "t1_mprage_sag_p2_iso",
                    "ProtocolName": "t1_mprage_sag_p2_iso",
                    "ImageType": ["ORIGINAL", "PRIMARY", "M"],
                }
            size = create_source(self.raw, rel, metadata)
            rows.append(selection_row(asset, rel, size, self.raw, self.logs, status))
        selection = self.root / "selection.tsv"
        write_tsv(selection, rows, SELECTION_COLUMNS)
        return config, assets, selection

    def test_logical_run_is_independent_of_json_scanner_label(self) -> None:
        config, assets, selection = self._offset_fixture()
        rows, issues = audit_selection(
            config,
            selection,
            self.raw,
            self.logs,
            self.stimuli,
            check_onsets=False,
        )
        self.assertEqual([issue for issue in issues if issue.severity == "error"], [])
        run = next(iter_expected_runs(config))
        self.assertEqual(run.experimental_run, 9)
        self.assertEqual(run.source_run_label, 10)
        self.assertTrue(str(bids_bold_relpath(run)).endswith("run-09_bold.nii.gz"))
        self.assertEqual(len(rows), len(assets))

    def test_moco_series_is_rejected_even_if_protocol_and_filename_match(self) -> None:
        config, _, selection = self._offset_fixture(moco=True)
        _, issues = audit_selection(
            config,
            selection,
            self.raw,
            self.logs,
            self.stimuli,
            check_onsets=False,
        )
        messages = "\n".join(issue.message for issue in issues)
        self.assertIn("moco_series", messages)

    def test_unapproved_manifest_fails_closed(self) -> None:
        config, _, selection = self._offset_fixture(status="proposed_unique")
        _, issues = audit_selection(
            config,
            selection,
            self.raw,
            self.logs,
            self.stimuli,
            check_onsets=False,
        )
        self.assertIn("not_approved", {issue.code for issue in issues})

    def test_frozen_json_onset_and_stimulus_hashes_detect_changes(self) -> None:
        config, _, selection = self._offset_fixture()
        sidecar = (
            self.raw
            / "DEV_016_2"
            / "DICOM"
            / "25061322"
            / "source_filename_RUN9_series12.json"
        )
        sidecar.write_text(sidecar.read_text(encoding="utf-8") + " ", encoding="utf-8")
        onset = self.logs / "Sub16" / "LogFiles" / "Run09.mat"
        onset.write_bytes(onset.read_bytes() + b"changed")
        self.stimuli.write_text(self.stimuli.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        _, issues = audit_selection(
            config,
            selection,
            self.raw,
            self.logs,
            self.stimuli,
            check_onsets=False,
            verify_hashes=True,
        )
        codes = {issue.code for issue in issues}
        self.assertIn("source_json_hash_changed", codes)
        self.assertIn("onset_mat_hash_changed", codes)
        self.assertIn("stimuli_csv_hash_changed", codes)

    def test_cross_session_fmap_source_is_rejected(self) -> None:
        config = minimal_config(
            7,
            [
                {
                    "session": 1,
                    "source_dir": "DEV_007",
                    "experimental_runs": [1],
                    "sdc_mode": "pepolar",
                },
                {
                    "session": 2,
                    "source_dir": "DEV_007_2",
                    "experimental_runs": [7],
                    "sdc_mode": "syn",
                },
            ],
        )
        freeze_stimuli(config, self.stimuli)
        rows = []
        for asset in iter_expected_assets(config):
            directory = asset.source_dir
            if asset.asset_type == "bold":
                rel = f"{directory}/bold_RUN{asset.source_run_label}.nii.gz"
                metadata = {
                    "SeriesDescription": f"cmrr_mbep2d_bold 1.80 RUN{asset.source_run_label}",
                    "ProtocolName": f"cmrr_mbep2d_bold 1.80 RUN{asset.source_run_label}",
                    "ImageType": ["ORIGINAL", "PRIMARY", "FMRI"],
                }
            elif asset.asset_type == "t1w":
                rel = f"{directory}/t1_mprage.nii.gz"
                metadata = {"SeriesDescription": "t1_mprage", "ProtocolName": "t1_mprage"}
            else:
                # Deliberately select a same-subject but wrong-session measured map.
                wrong_directory = "DEV_007_2"
                direction = "AP" if asset.fmap_role == "epi_ap" else "PA"
                rel = f"{wrong_directory}/fMRI-DistMap_{direction}.nii.gz"
                metadata = {
                    "SeriesDescription": f"fMRI-DistMap_{direction}",
                    "ProtocolName": f"fMRI-DistMap_{direction}",
                }
            size = create_source(self.raw, rel, metadata, payload=asset.asset_id.encode())
            rows.append(selection_row(asset, rel, size, self.raw, self.logs))
        selection = self.root / "cross_session.tsv"
        write_tsv(selection, rows, SELECTION_COLUMNS)
        _, issues = audit_selection(
            config,
            selection,
            self.raw,
            self.logs,
            self.stimuli,
            check_onsets=False,
        )
        self.assertIn("wrong_acquisition_directory", {issue.code for issue in issues})


class BuilderProvenanceTests(unittest.TestCase):
    def test_generated_json_and_events_hashes_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            logs = root / "logs"
            raw.mkdir()
            logs.mkdir()
            stimuli = root / "stimuli.csv"
            expected = pd.DataFrame(
                {
                    "run": [1] * 60,
                    "OriginalOrder": list(range(60)),
                    "img": [f"image_{index:02d}.jpg" for index in range(60)],
                    "group": ["pleasant_natural"] * 60,
                    "AI": ["NO"] * 60,
                    "emotion_type": ["pleasant"] * 60,
                }
            )
            expected.to_csv(stimuli, index=False)
            config = minimal_config(
                40,
                [
                    {
                        "session": 1,
                        "source_dir": "DEV_040",
                        "experimental_runs": [1],
                        "sdc_mode": "syn",
                    }
                ],
            )
            freeze_stimuli(config, stimuli)
            log_path = logs / "Sub40" / "LogFiles" / "Run01.mat"
            log_path.parent.mkdir(parents=True)
            data_log = np.empty((62, 4), dtype=object)
            data_log[0] = ["trial", "event", "stimulus", "time"]
            data_log[1] = [0, "FixOn", "", 100.0]
            for index, image_name in enumerate(expected["img"]):
                data_log[index + 2] = [index + 1, "Stim on", image_name, 101.0 + 4.0 * index]
            savemat(log_path, {"dataLog": data_log})

            rows = []
            for asset in iter_expected_assets(config):
                if asset.asset_type == "bold":
                    relpath = "DEV_040/bold_RUN1.nii.gz"
                    metadata = {
                        "SeriesDescription": "cmrr_mbep2d_bold 1.80 RUN1",
                        "ProtocolName": "cmrr_mbep2d_bold 1.80 RUN1",
                    }
                else:
                    relpath = "DEV_040/t1_mprage.nii.gz"
                    metadata = {
                        "SeriesDescription": "t1_mprage",
                        "ProtocolName": "t1_mprage",
                    }
                size = create_source(raw, relpath, metadata, payload=asset.asset_id.encode())
                rows.append(selection_row(asset, relpath, size, raw, logs))
            selection = root / "selection.tsv"
            write_tsv(selection, rows, SELECTION_COLUMNS)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "bids"
            build_bids(
                config,
                rows,
                raw,
                logs,
                stimuli,
                output,
                None,
                link_niftis=False,
                config_path=config_path,
                selection_path=selection,
            )
            generated = pd.read_csv(output / "code" / "generated_bids_file_manifest.tsv", sep="\t")
            self.assertEqual(len(generated), 5)
            for record in generated.to_dict("records"):
                path = output.joinpath(*Path(record["bids_relpath"]).parts)
                self.assertEqual(int(record["size_bytes"]), path.stat().st_size)
                self.assertEqual(record["sha256"], sha256(path))
            built = pd.read_csv(output / "code" / "built_source_manifest.tsv", sep="\t")
            bold = built[built["asset_id"].str.endswith("_bold")].iloc[0]
            self.assertEqual(
                bold["bids_events_sha256"],
                sha256(output.joinpath(*Path(bold["bids_events_relpath"]).parts)),
            )


class OnsetTests(unittest.TestCase):
    def test_sub1_unpadded_log_and_exact_60_trial_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "Sub1" / "LogFiles" / "Run1.mat"
            log_path.parent.mkdir(parents=True)
            expected = pd.DataFrame(
                {
                    "run": [1] * 60,
                    "OriginalOrder": list(range(60)),
                    "img": [f"image_{index:02d}.jpg" for index in range(60)],
                    "group": ["pleasant_natural"] * 60,
                    "AI": ["NO"] * 60,
                    "emotion_type": ["pleasant"] * 60,
                }
            )
            data_log = np.empty((62, 4), dtype=object)
            data_log[0] = ["trial", "event", "stimulus", "time"]
            data_log[1] = [0, "FixOn", "", 100.0]
            for index, image_name in enumerate(expected["img"], start=0):
                data_log[index + 2] = [index + 1, "Stim on", image_name, 101.0 + 4.0 * index]
            savemat(log_path, {"dataLog": data_log})
            events = extract_events(log_path, expected)
            self.assertEqual(len(events), 60)
            self.assertEqual(events["image_id"].iloc[0], "image_00")
            self.assertTrue(np.all(np.diff(events["onset"]) > 0))

            bad = expected.copy()
            bad.loc[5, "img"] = "wrong.jpg"
            with self.assertRaisesRegex(RuntimeError, "Stimulus order/name mismatch"):
                extract_events(log_path, bad)


if __name__ == "__main__":
    unittest.main()
