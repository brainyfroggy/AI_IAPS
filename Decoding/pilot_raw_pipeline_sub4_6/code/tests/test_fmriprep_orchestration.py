"""Small, filesystem-only tests for Sub-05 staging and derivative merging."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import build_sub05_feb_syn_subset as subset_builder  # noqa: E402
import merge_sub05_fmriprep as merger  # noqa: E402


SYN_FILTER = CODE_ROOT / "sub05_feb_syn_bids_filter.json"
RUNNER = CODE_ROOT / "run_fmriprep_container.ps1"


def touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_syn_filter(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(SYN_FILTER.read_bytes())
    return path


def make_bids(root: Path, with_field_source: bool = False) -> None:
    touch(root / "dataset_description.json", '{"Name":"test","BIDSVersion":"1.10.0"}')
    touch(root / "participants.tsv", "participant_id\nsub-04\nsub-05\nsub-06\n")
    touch(root / "task-iaps_events.json", "{}")
    touch(root / "README", "test")
    touch(root / "sub-05" / "anat" / "sub-05_T1w.nii.gz")
    touch(root / "sub-05" / "anat" / "sub-05_T1w.json", "{}")
    for run in subset_builder.EXPECTED_RUNS:
        prefix = root / "sub-05" / "ses-01" / "func" / (
            f"sub-05_ses-01_task-iaps_run-{run:02d}"
        )
        touch(prefix.with_name(prefix.name + "_bold.nii.gz"))
        metadata = {"TaskName": "iaps"}
        if with_field_source and run == 1:
            metadata["B0FieldSource"] = "wrong-session-fieldmap"
        touch(prefix.with_name(prefix.name + "_bold.json"), json.dumps(metadata))
        touch(prefix.with_name(prefix.name + "_events.tsv"), "onset\tduration\n1\t3\n")


def make_derivative(root: Path, sessions: dict[str, set[int]], markers: dict[str, str]) -> None:
    touch(root / "dataset_description.json", '{"Name":"fMRIPrep test","DatasetType":"derivative"}')
    touch(root / "sub-05.html", "<html>complete report</html>")
    for session, runs in sessions.items():
        for run in runs:
            prefix = root / "sub-05" / session / "func" / (
                f"sub-05_{session}_task-iaps_run-{run:02d}"
            )
            for suffix in (
                "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
                "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.json",
                "_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz",
                "_desc-confounds_timeseries.tsv",
            ):
                touch(prefix.with_name(prefix.name + suffix), f"{session}-{run}")
            marker = markers[session]
            touch(
                root
                / "sub-05"
                / "figures"
                / f"sub-05_{session}_task-iaps_run-{run:02d}_desc-summary_bold.html",
                f"Susceptibility distortion correction: {marker}",
            )
            touch(
                root
                / "sub-05"
                / "figures"
                / f"sub-05_{session}_task-iaps_run-{run:02d}_desc-sdc_bold.svg",
                marker,
            )


class FebruarySubsetTests(unittest.TestCase):
    def test_syn_filter_requests_optional_session_scope(self) -> None:
        self.assertEqual(
            json.loads(SYN_FILTER.read_text(encoding="utf-8")),
            {"fmap": {"session": "<Query.OPTIONAL: 3>"}},
        )

    def test_runner_supports_filter_and_fail_closed_syn_mode(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("[ValidateSet('warn', 'error')]", runner)
        self.assertIn("[string]$SynSdcMode = 'warn'", runner)
        self.assertIn("'--use-syn-sdc', $SynSdcMode", runner)
        self.assertIn("'--bids-filter-file', '/bids-filter.json'", runner)
        self.assertIn("[switch]$RequireSynAnatEstimator", runner)
        self.assertIn("[switch]$PreflightOnly", runner)
        self.assertIn("SyN ANAT estimator preflight passed", runner)

    def test_builds_only_exact_february_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            source, destination = root / "source", root / "subset"
            make_bids(source)
            copied = subset_builder.build_subset(source, destination, hardlink=False)
            self.assertEqual(sum(path.name.endswith("_bold.nii.gz") for path in copied), 4)
            self.assertFalse(any("ses-02" in str(path) for path in destination.rglob("*")))
            self.assertFalse(any("fmap" in path.name for path in destination.rglob("*")))
            self.assertEqual(
                (destination / "participants.tsv").read_text(encoding="utf-8"),
                "participant_id\nsub-05\n",
            )
            self.assertEqual(
                (source / "participants.tsv").read_text(encoding="utf-8"),
                "participant_id\nsub-04\nsub-05\nsub-06\n",
            )
            filter_path = destination.with_name(
                f"{destination.name}_fmriprep_bids_filter.json"
            )
            self.assertEqual(
                json.loads(filter_path.read_text(encoding="utf-8")),
                {"fmap": {"session": "<Query.OPTIONAL: 3>"}},
            )

    def test_rejects_fieldmap_reference_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            source, destination = root / "source", root / "subset"
            make_bids(source, with_field_source=True)
            with self.assertRaisesRegex(RuntimeError, "fieldmap"):
                subset_builder.build_subset(source, destination, hardlink=False)
            self.assertFalse(destination.exists())

    def test_rejects_session_labelled_anatomical_alternative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            source, destination = root / "source", root / "subset"
            make_bids(source)
            touch(source / "sub-05" / "ses-01" / "anat" / "sub-05_ses-01_T1w.nii.gz")
            with self.assertRaisesRegex(RuntimeError, "session-labelled"):
                subset_builder.build_subset(source, destination, hardlink=False)
            self.assertFalse(destination.exists())

    def test_rejects_stale_filter_before_writing_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            source, destination = root / "source", root / "subset"
            make_bids(source)
            filter_path = destination.with_name(
                f"{destination.name}_fmriprep_bids_filter.json"
            )
            touch(filter_path, "user-owned")
            with self.assertRaisesRegex(FileExistsError, "BIDS filter"):
                subset_builder.build_subset(source, destination, hardlink=False)
            self.assertFalse(destination.exists())
            self.assertEqual(filter_path.read_text(encoding="utf-8"), "user-owned")


class MergeTests(unittest.TestCase):
    def test_rejects_nonempty_destination_without_overwriting_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            all_sessions, feb_syn, destination = root / "all", root / "feb", root / "merged"
            make_derivative(all_sessions, merger.EXPECTED_RUNS, {"ses-01": "None", "ses-02": "PEPOLAR"})
            make_derivative(
                feb_syn,
                {"ses-01": merger.EXPECTED_RUNS["ses-01"], "ses-02": set()},
                {"ses-01": "fieldmap-less SyN", "ses-02": "unused"},
            )
            feb_filter = make_syn_filter(root / "feb_bids_filter.json")
            sentinel = destination / "user-owned.txt"
            touch(sentinel, "preserve me")

            with self.assertRaisesRegex(FileExistsError, "absent or empty"):
                merger.assemble(
                    all_sessions,
                    feb_syn,
                    destination,
                    hardlink=False,
                    feb_bids_filter=feb_filter,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me")
            self.assertEqual([path for path in destination.rglob("*") if path.is_file()], [sentinel])

    def test_merge_is_session_specific_and_does_not_copy_other_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            all_sessions, feb_syn, destination = root / "all", root / "feb", root / "merged"
            make_derivative(all_sessions, merger.EXPECTED_RUNS, {"ses-01": "None", "ses-02": "PEPOLAR"})
            make_derivative(
                feb_syn,
                {"ses-01": merger.EXPECTED_RUNS["ses-01"], "ses-02": set()},
                {"ses-01": "fieldmap-less SyN", "ses-02": "unused"},
            )
            feb_filter = make_syn_filter(root / "feb_bids_filter.json")
            touch(all_sessions / "sub-06" / "ses-01" / "func" / "must-not-copy.txt")

            merger.assemble(
                all_sessions,
                feb_syn,
                destination,
                hardlink=True,
                feb_bids_filter=feb_filter,
            )

            self.assertFalse((destination / "sub-06").exists())
            self.assertFalse((destination / "sub-05.html").exists())
            self.assertTrue(
                (destination / "logs" / "merge_branches" / "all_sessions" / "sub-05.html").is_file()
            )
            self.assertTrue(
                (destination / "logs" / "merge_branches" / "february_syn" / "sub-05.html").is_file()
            )
            feb_reportlet = (
                destination
                / "sub-05"
                / "figures"
                / "sub-05_ses-01_task-iaps_run-01_desc-sdc_bold.svg"
            )
            self.assertEqual(feb_reportlet.read_text(encoding="utf-8"), "fieldmap-less SyN")
            self.assertEqual(merger.session_runs(destination, "_desc-preproc_bold.nii.gz"), merger.EXPECTED_RUNS)
            for suffix in merger.REQUIRED_RUN_SUFFIXES:
                self.assertEqual(merger.session_runs(destination, suffix), merger.EXPECTED_RUNS)

            provenance_path = destination / "logs" / "merge_sub05_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(
                provenance["report_bundles"],
                {
                    "all_sessions": "logs/merge_branches/all_sessions/sub-05.html",
                    "february_syn": "logs/merge_branches/february_syn/sub-05.html",
                },
            )
            for relative_path in provenance["report_bundles"].values():
                self.assertTrue((destination / relative_path).is_file())
            archived_filter = (
                destination / "logs" / "merge_branches" / "february_syn" / "bids_filter.json"
            )
            expected_sha256 = hashlib.sha256(SYN_FILTER.read_bytes()).hexdigest()
            self.assertEqual(archived_filter.read_bytes(), SYN_FILTER.read_bytes())
            self.assertFalse(archived_filter.samefile(feb_filter))
            self.assertEqual(
                provenance["february_bids_filter"],
                {
                    "source_path": str(feb_filter.resolve()),
                    "archived_relative_path": (
                        "logs/merge_branches/february_syn/bids_filter.json"
                    ),
                    "sha256": expected_sha256,
                },
            )

    def test_rejects_wrong_february_filter_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            all_sessions, feb_syn, destination = root / "all", root / "feb", root / "merged"
            make_derivative(
                all_sessions,
                merger.EXPECTED_RUNS,
                {"ses-01": "None", "ses-02": "PEPOLAR"},
            )
            make_derivative(
                feb_syn,
                {"ses-01": merger.EXPECTED_RUNS["ses-01"], "ses-02": set()},
                {"ses-01": "fieldmap-less SyN", "ses-02": "unused"},
            )
            wrong_filter = root / "wrong_filter.json"
            touch(wrong_filter, '{"fmap":{"session":"01"}}\n')

            with self.assertRaisesRegex(RuntimeError, "does not match the pinned"):
                merger.assemble(
                    all_sessions,
                    feb_syn,
                    destination,
                    hardlink=False,
                    feb_bids_filter=wrong_filter,
                )
            self.assertFalse(destination.exists())

    def test_rejects_a_non_syn_february_branch_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            all_sessions, feb_syn, destination = root / "all", root / "feb", root / "merged"
            make_derivative(all_sessions, merger.EXPECTED_RUNS, {"ses-01": "None", "ses-02": "PEPOLAR"})
            make_derivative(
                feb_syn,
                {"ses-01": merger.EXPECTED_RUNS["ses-01"], "ses-02": set()},
                {"ses-01": "None", "ses-02": "unused"},
            )
            feb_filter = make_syn_filter(root / "feb_bids_filter.json")
            with self.assertRaisesRegex(RuntimeError, "'syn'"):
                merger.assemble(
                    all_sessions,
                    feb_syn,
                    destination,
                    hardlink=False,
                    feb_bids_filter=feb_filter,
                )
            self.assertFalse(destination.exists())

    def test_rejects_incidental_syn_text_when_sdc_method_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            all_sessions, feb_syn, destination = root / "all", root / "feb", root / "merged"
            make_derivative(all_sessions, merger.EXPECTED_RUNS, {"ses-01": "None", "ses-02": "PEPOLAR"})
            make_derivative(
                feb_syn,
                {"ses-01": merger.EXPECTED_RUNS["ses-01"], "ses-02": set()},
                {"ses-01": "None", "ses-02": "unused"},
            )
            for report in (feb_syn / "sub-05" / "figures").glob("*_desc-summary_bold.html"):
                report.write_text(
                    report.read_text(encoding="utf-8") + "\n<!-- unrelated SyN text -->\n",
                    encoding="utf-8",
                )
            feb_filter = make_syn_filter(root / "feb_bids_filter.json")

            with self.assertRaisesRegex(RuntimeError, "found 'none'"):
                merger.assemble(
                    all_sessions,
                    feb_syn,
                    destination,
                    hardlink=False,
                    feb_bids_filter=feb_filter,
                )
            self.assertFalse(destination.exists())

    def test_rejects_wrong_standard_space_derivative_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            all_sessions, feb_syn, destination = root / "all", root / "feb", root / "merged"
            make_derivative(all_sessions, merger.EXPECTED_RUNS, {"ses-01": "None", "ses-02": "PEPOLAR"})
            make_derivative(
                feb_syn,
                {"ses-01": merger.EXPECTED_RUNS["ses-01"], "ses-02": set()},
                {"ses-01": "fieldmap-less SyN", "ses-02": "unused"},
            )
            correct = next(
                (feb_syn / "sub-05" / "ses-01" / "func").glob(
                    "*_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
                )
            )
            wrong = correct.with_name(
                correct.name.replace("space-MNI152NLin6Asym_res-2", "space-MNI152NLin2009cAsym_res-2")
            )
            correct.rename(wrong)
            feb_filter = make_syn_filter(root / "feb_bids_filter.json")

            with self.assertRaisesRegex(RuntimeError, "wrong session/run set"):
                merger.assemble(
                    all_sessions,
                    feb_syn,
                    destination,
                    hardlink=False,
                    feb_bids_filter=feb_filter,
                )
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
