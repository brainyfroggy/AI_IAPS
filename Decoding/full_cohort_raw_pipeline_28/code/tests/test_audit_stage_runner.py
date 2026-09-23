from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
SCRIPT = CODE / "audit_stage_runner.py"
SPEC = importlib.util.spec_from_file_location("audit_stage_runner", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from production_control import file_record, read_ledger  # noqa: E402


DiskUsage = namedtuple("DiskUsage", "total used free")
GOOD_DISK = DiskUsage(200 * 1024**3, 100 * 1024**3, 100 * 1024**3)
LOW_DISK = DiskUsage(200 * 1024**3, 199 * 1024**3, 1 * 1024**3)


class AuditStageRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.canonical_cohort = self.root / "canonical_cohort.json"
        self.canonical_cohort.write_text('{"subjects": 28}\n', encoding="utf-8")
        self.freeze = self.root / "freeze.json"
        included = [1, 2, 4, 5, 6, 7, 9, *range(11, 32)]
        remaining = [1, 2, 7, 9, *range(11, 32)]
        self.freeze.write_text(
            json.dumps(
                {
                    "analysis_id": "test-analysis",
                    "freeze_status": "frozen_for_production",
                    "included_subjects": included,
                    "pilot_complete_subjects": [4, 5, 6],
                    "remaining_batch_subjects": remaining,
                    "canonical_cohort_config": str(self.canonical_cohort),
                    "canonical_cohort_config_sha256": file_record(
                        self.canonical_cohort
                    )["sha256"],
                }
            ),
            encoding="utf-8",
        )
        self.cohort = self.root / "cohort.json"
        self.cohort.write_text('{"test": true}\n', encoding="utf-8")
        self.selection = self.root / "selection.tsv"
        self.selection.write_text("test\n", encoding="utf-8")
        self.command_script = self.root / "command.py"
        self.marker = self.root / "ran.txt"
        self.command_script.write_text(
            "import os, pathlib, sys\n"
            "pathlib.Path(os.environ['CONTROL_TEST_MARKER']).write_text('ran')\n"
            "print('controlled', *sys.argv[1:])\n"
            "raise SystemExit(int(os.environ.get('CONTROL_TEST_EXIT', '0')))\n",
            encoding="utf-8",
        )
        self.contract = self.root / "contract.json"
        self._write_contract()
        self.approval = self.root / "approval.json"
        self._write_approval()
        self.logs = self.root / "logs"
        self.environment = mock.patch.dict(
            os.environ,
            {"CONTROL_TEST_MARKER": str(self.marker), "CONTROL_TEST_EXIT": "0"},
        )
        self.environment.start()
        self.disk = mock.patch.object(runner.shutil, "disk_usage", return_value=GOOD_DISK)
        self.disk.start()

    def tearDown(self) -> None:
        self.disk.stop()
        self.environment.stop()
        self.temp.cleanup()

    def _write_contract(self, *, blocked: str | None = None) -> None:
        source = {
            "state": "blocked" if blocked else "executable",
            "required_approved_files": [
                "freeze",
                "cohort_config",
                "source_selection",
                "stage_contract",
                "python_executable",
                "command_script",
            ],
            "command_argv": [
                "{python_executable}",
                "{command_script}",
                "--subject",
                "{subject}",
                "--stage",
                "source_audit",
            ],
        }
        if blocked:
            source["blocked_reason"] = blocked
        payload = {
            "schema_version": 1,
            "analysis_id": "test-analysis",
            "storage_gate": {
                "path": str(self.root),
                "minimum_free_bytes": 60 * 1024**3,
            },
            "stage_order": ["source_audit", "bids_build"],
            "pin_files": {
                "python_executable": sys.executable,
                "command_script": str(self.command_script),
            },
            "path_variables": {},
            "stages": {
                "source_audit": source,
                "bids_build": {
                    "state": "executable",
                    "required_approved_files": [
                        "freeze",
                        "cohort_config",
                        "source_selection",
                        "stage_contract",
                        "python_executable",
                        "command_script",
                    ],
                    "command_argv": [
                        "{python_executable}",
                        "{command_script}",
                        "--subject",
                        "{subject}",
                        "--stage",
                        "bids_build",
                    ],
                },
            },
        }
        self.contract.write_text(json.dumps(payload), encoding="utf-8")

    def _write_approval(self) -> None:
        approved_files = {
            "freeze": file_record(self.freeze),
            "stage_contract": file_record(self.contract),
            "canonical_cohort_config": file_record(self.canonical_cohort),
            "cohort_config": file_record(self.cohort),
            "source_selection": file_record(self.selection),
            "python_executable": file_record(Path(sys.executable)),
            "command_script": file_record(self.command_script),
        }
        self.approval.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "approval_status": "approved_for_production",
                    "analysis_id": "test-analysis",
                    "production_subjects": [1, 2, 7, 9, *range(11, 32)],
                    "approved_files": approved_files,
                }
            ),
            encoding="utf-8",
        )

    def argv(self, stage: str = "source_audit", *extra: str) -> list[str]:
        return [
            "--subject",
            "1",
            "--stage",
            stage,
            "--freeze",
            str(self.freeze),
            "--contract",
            str(self.contract),
            "--approval",
            str(self.approval),
            "--log-root",
            str(self.logs),
            *extra,
        ]

    def result_paths(self, stage: str = "source_audit") -> list[Path]:
        return list(
            (self.logs / "subjects" / "Sub1" / "stages" / stage / "attempts").glob(
                "*/result.json"
            )
        )

    def test_success_is_hash_chained_and_lock_is_released(self) -> None:
        self.assertEqual(runner.main(self.argv()), 0)
        self.assertTrue(self.marker.exists())
        result_path = self.result_paths()[0]
        result = json.loads(result_path.read_text())
        attempt = json.loads((result_path.parent / "attempt.json").read_text())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(len(attempt["required_approved_files"]), 7)
        self.assertGreaterEqual(attempt["storage_gate"]["free_bytes"], 60 * 1024**3)
        self.assertFalse((self.logs / "subjects" / "Sub1" / ".lifecycle.lock").exists())
        state = read_ledger(self.logs / "ledger.jsonl")
        self.assertEqual([event["event"] for event in state.events], ["started", "completed"])
        runner.verify_event_mirror(self.logs, state)

    def test_arbitrary_command_arguments_are_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            runner.parse_args([*self.argv(), "--", sys.executable, "-c", "print(1)"])

    def test_subject_outside_approved_batch_is_rejected_without_logs(self) -> None:
        args = self.argv()
        args[1] = "4"
        self.assertEqual(runner.main(args), 2)
        self.assertFalse(self.logs.exists())

    def test_prelaunch_freeze_is_rejected_without_logs(self) -> None:
        payload = json.loads(self.freeze.read_text())
        payload["freeze_status"] = "prelaunch_pending_source_manifest_and_bids_validation"
        self.freeze.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(runner.main(self.argv()), 2)
        self.assertFalse(self.logs.exists())

    def test_changed_config_is_rejected_by_approval(self) -> None:
        self.cohort.write_text('{"changed": true}\n', encoding="utf-8")
        self.assertEqual(runner.main(self.argv()), 2)
        self.assertFalse(self.marker.exists())

    def test_low_storage_is_recorded_and_command_is_not_run(self) -> None:
        self.disk.stop()
        self.disk = mock.patch.object(runner.shutil, "disk_usage", return_value=LOW_DISK)
        self.disk.start()
        self.assertEqual(runner.main(self.argv()), 2)
        self.assertFalse(self.marker.exists())
        result = json.loads(self.result_paths()[0].read_text())
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["storage_gate_passed"])
        self.assertEqual(read_ledger(self.logs / "ledger.jsonl").events[-1]["status"], "blocked")

    def test_low_storage_attempt_can_resume_only_with_same_signature(self) -> None:
        self.disk.stop()
        self.disk = mock.patch.object(runner.shutil, "disk_usage", return_value=LOW_DISK)
        self.disk.start()
        self.assertEqual(runner.main(self.argv()), 2)
        self.disk.stop()
        self.disk = mock.patch.object(runner.shutil, "disk_usage", return_value=GOOD_DISK)
        self.disk.start()
        self.assertEqual(runner.main(self.argv()), 2)
        self.assertEqual(runner.main(self.argv("source_audit", "--resume")), 0)
        self.assertEqual(len(self.result_paths()), 2)

    def test_failed_command_requires_explicit_same_signature_resume(self) -> None:
        os.environ["CONTROL_TEST_EXIT"] = "7"
        self.assertEqual(runner.main(self.argv()), 7)
        os.environ["CONTROL_TEST_EXIT"] = "0"
        self.assertEqual(runner.main(self.argv()), 2)
        self.assertEqual(runner.main(self.argv("source_audit", "--resume")), 0)

    def test_prior_stage_pass_is_mandatory(self) -> None:
        self.assertEqual(runner.main(self.argv("bids_build")), 2)
        self.assertFalse(self.marker.exists())
        self.assertEqual(runner.main(self.argv("source_audit")), 0)
        self.marker.unlink()
        self.assertEqual(runner.main(self.argv("bids_build")), 0)
        self.assertTrue(self.marker.exists())

    def test_duplicate_passed_stage_is_forbidden(self) -> None:
        self.assertEqual(runner.main(self.argv()), 0)
        self.marker.unlink()
        self.assertEqual(runner.main(self.argv()), 2)
        self.assertFalse(self.marker.exists())

    def test_dry_run_has_no_log_or_command_side_effects(self) -> None:
        self.assertEqual(runner.main(self.argv("source_audit", "--dry-run")), 0)
        self.assertFalse(self.logs.exists())
        self.assertFalse(self.marker.exists())

    def test_existing_subject_lock_fails_closed(self) -> None:
        lock = self.logs / "subjects" / "Sub1" / ".lifecycle.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text('{"token":"other"}\n', encoding="utf-8")
        self.assertEqual(runner.main(self.argv()), 2)
        self.assertFalse(self.marker.exists())

    def test_tampered_ledger_is_detected(self) -> None:
        self.assertEqual(runner.main(self.argv()), 0)
        lines = (self.logs / "ledger.jsonl").read_text().splitlines()
        event = json.loads(lines[0])
        event["subject"] = 2
        lines[0] = json.dumps(event, separators=(",", ":"), sort_keys=True)
        (self.logs / "ledger.jsonl").write_text("\n".join(lines) + "\n")
        self.assertEqual(runner.main(self.argv("bids_build")), 2)

    def test_tampered_completed_stdout_is_detected(self) -> None:
        self.assertEqual(runner.main(self.argv()), 0)
        result_path = self.result_paths()[0]
        (result_path.parent / "stdout.log").write_text("changed\n", encoding="utf-8")
        self.assertEqual(runner.main(self.argv("bids_build")), 2)
        self.assertEqual(len(self.result_paths("bids_build")), 0)

    def test_blocked_contract_cannot_launch(self) -> None:
        self._write_contract(blocked="not approved")
        self._write_approval()
        self.assertEqual(runner.main(self.argv()), 2)
        self.assertFalse(self.marker.exists())


if __name__ == "__main__":
    unittest.main()
