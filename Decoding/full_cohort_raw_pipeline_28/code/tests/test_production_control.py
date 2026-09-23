from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import production_control as control  # noqa: E402
from production_stage_helpers import count_validator_errors, windows_to_wsl  # noqa: E402


class ProductionControlTests(unittest.TestCase):
    def test_hash_chained_event_round_trip_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.jsonl"
            empty = control.read_ledger(ledger)
            first = control.encode_ledger_event(empty, {"event": "started"})
            ledger.write_text(json.dumps(first, sort_keys=True, separators=(",", ":")) + "\n")
            state = control.read_ledger(ledger)
            self.assertEqual(state.head_sha256, first["event_sha256"])
            second = control.encode_ledger_event(state, {"event": "completed"})
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(second, sort_keys=True, separators=(",", ":")) + "\n")
            self.assertEqual(len(control.read_ledger(ledger).events), 2)
            payload = json.loads(ledger.read_text().splitlines()[0])
            payload["event"] = "changed"
            lines = ledger.read_text().splitlines()
            lines[0] = json.dumps(payload)
            ledger.write_text("\n".join(lines) + "\n")
            with self.assertRaises(control.ControlError):
                control.read_ledger(ledger)

    def test_manifest_approval_requires_exact_25_and_ten_bolds(self) -> None:
        subjects = [1, 2, 7, 9, *range(11, 32)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.tsv"
            fields = ["asset_id", "asset_type", "subject", "approval_status", "source_sha256"]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                for subject in subjects:
                    for run in range(1, 11):
                        writer.writerow(
                            {
                                "asset_id": f"sub{subject}-run{run}",
                                "asset_type": "bold",
                                "subject": subject,
                                "approval_status": "approved",
                                "source_sha256": "a" * 64,
                            }
                        )
            summary = control.validate_source_selection_for_approval(path, subjects)
            self.assertEqual(summary["row_count"], 250)
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaises(control.ControlError):
                control.validate_source_selection_for_approval(path, subjects)

    def test_subject_set_order_must_be_exact(self) -> None:
        with self.assertRaises(control.ControlError):
            control.ensure_exact_subjects([2, 1], [1, 2])
        control.ensure_exact_subjects([1, 2], [1, 2])

    def test_validator_error_count(self) -> None:
        payload = {
            "issues": {
                "issues": [
                    {"severity": "warning"},
                    {"severity": "error"},
                    {"severity": "ERROR"},
                ]
            }
        }
        self.assertEqual(count_validator_errors(payload), 2)
        with self.assertRaises(ValueError):
            count_validator_errors({})

    @unittest.skipUnless(sys.platform == "win32", "Windows path mapping test")
    def test_windows_to_wsl_path(self) -> None:
        value = windows_to_wsl(Path(r"N:\folder\file.json"))
        self.assertEqual(value, "/mnt/n/folder/file.json")
        resolved = Path("N:\\").resolve(strict=True) / "folder" / "file.json"
        self.assertEqual(windows_to_wsl(resolved), "/mnt/n/folder/file.json")


if __name__ == "__main__":
    unittest.main()
