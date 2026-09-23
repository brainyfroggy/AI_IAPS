from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import fmriprep_sdc_workflow as workflow
import production_postproc_gates as gates
import production_control


def one_session_plan(subject: int = 1) -> workflow.SubjectPlan:
    return workflow.SubjectPlan(
        subject=subject,
        subject_label=f"{subject:02d}",
        strategy="direct_all_measured_sessions",
        sessions=(workflow.SessionSpec("01", "pepolar", tuple(range(1, 11))),),
        branches=(
            workflow.BranchSpec(
                "all_measured",
                "pepolar",
                ("01",),
                tuple(range(1, 11)),
                "canonical",
                False,
            ),
        ),
    )


class VisualGateTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Production path mapping is Windows-specific")
    def test_resolved_n_share_maps_to_exact_wsl_mount(self) -> None:
        mapped = gates._wsl_path(
            Path(r"C:\Windows\System32\wsl.exe"), "Ubuntu-22.04", ROOT
        )
        self.assertEqual(
            mapped,
            "/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/"
            "full_cohort_raw_pipeline_28",
        )

    def make_visual_tree(self, base: Path) -> tuple[Path, Path, workflow.SubjectPlan]:
        plan = one_session_plan()
        derivative = base / "derivative"
        figures = derivative / "sub-01" / "figures"
        figures.mkdir(parents=True)
        output = base / "visual"
        session_root = output / "visual" / "ses-01"
        session_root.mkdir(parents=True)
        sources = []
        rows = []
        for run in range(1, 11):
            for description, layers in (
                ("sdc", ("background", "foreground")),
                ("coreg", ("background", "foreground")),
                ("rois", ("static",)),
                ("carpetplot", ("static",)),
            ):
                source = figures / (
                    f"sub-01_ses-01_task-iaps_run-{run:02d}_"
                    f"desc-{description}_bold.svg"
                )
                source.write_text(f"<svg>{run}-{description}</svg>", encoding="utf-8")
                source_hash = gates.sha256_file(source)
                sources.append(
                    {
                        "run": run,
                        "description": description,
                        "path": str(source.resolve()),
                        "sha256": source_hash,
                    }
                )
                for layer in layers:
                    suffix = f"_{layer}" if layer != "static" else ""
                    png = session_root / f"run-{run:02d}_{description}{suffix}.png"
                    png.write_bytes(b"PNG" + bytes([run]))
                    rows.append(
                        {
                            "run": f"{run:02d}",
                            "session": "01",
                            "task": "iaps",
                            "description": description,
                            "layer": layer,
                            "source_svg": str(source.resolve()),
                            "source_sha256": source_hash,
                            "output_png": png.name,
                            "width_pixels": "10",
                            "height_pixels": "10",
                        }
                    )
        contacts = {}
        for category in gates.VISUAL_CATEGORIES:
            name = f"sub-01_ses-01_{category}_contact.png"
            (session_root / name).write_bytes(b"CONTACT")
            contacts[category] = name
        with (session_root / "visual_qc_manifest.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        provenance = {
            "source_fmriprep_root": str(derivative.resolve()),
            "subject": 1,
            "session": "01",
            "task": "iaps",
            "expected_runs": list(range(1, 11)),
            "source_reportlets": sources,
            "contact_sheets": contacts,
        }
        (session_root / "provenance.json").write_text(
            json.dumps(provenance) + "\n", encoding="utf-8"
        )
        payload = gates.visual_manifest_payload(
            output,
            derivative,
            plan,
            {"renderer_sha256": "a" * 64, "renderer": "/usr/bin/inkscape"},
        )
        gates.write_json_immutable(output / gates.VISUAL_MANIFEST, payload)
        gates.write_pending_attestation(
            output / gates.ATTESTATION,
            plan,
            gates.sha256_file(output / gates.VISUAL_MANIFEST),
        )
        return output, derivative, plan

    def test_visual_manifest_is_deterministic_and_attestation_fails_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            output, derivative, plan = self.make_visual_tree(Path(temp_name))
            first = gates.validate_visual_manifest(output, derivative, plan)
            second = gates.validate_visual_manifest(output, derivative, plan)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(RuntimeError, "explicit ACCEPT"):
                gates.validate_attestation(
                    output / gates.ATTESTATION, plan, first["sha256"]
                )

    def test_attestation_binds_exact_visual_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            output, derivative, plan = self.make_visual_tree(Path(temp_name))
            manifest = gates.validate_visual_manifest(output, derivative, plan)
            path = output / gates.ATTESTATION
            with path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            for row in rows:
                for field in (
                    "sdc_review",
                    "coreg_review",
                    "carpet_motion_review",
                    "overall_decision",
                ):
                    row[field] = "ACCEPT"
                row["reviewer"] = "Unit Test Reviewer"
                row["reviewed_at_utc"] = "2026-07-29T23:00:00Z"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=gates.ATTESTATION_FIELDS,
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(
                gates.validate_attestation(path, plan, manifest["sha256"])["status"],
                "pass",
            )
            rows[0]["visual_manifest_sha256"] = "0" * 64
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=gates.ATTESTATION_FIELDS,
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(RuntimeError, "manifest hash"):
                gates.validate_attestation(path, plan, manifest["sha256"])


class ArchiveRetirementTests(unittest.TestCase):
    def test_checksum_archive_rejects_files_added_after_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            root = base / "root"
            root.mkdir()
            frozen = root / "frozen.txt"
            frozen.write_text("frozen", encoding="utf-8")
            manifest = base / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "file_count": 1,
                        "roots": {"root": str(root)},
                        "entries": [
                            {
                                "root_label": "root",
                                "relative_path": "frozen.txt",
                                "size_bytes": frozen.stat().st_size,
                                "sha256": gates.sha256_file(frozen),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(
                gates.fpost,
                "verify_checksum_manifest",
                return_value={"status": "pass", "manifest_sha256": "a" * 64},
            ):
                self.assertEqual(
                    gates.verify_checksum_manifest_exact(manifest)["status"], "pass"
                )
                (root / "late.txt").write_text("late", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "file set changed"):
                    gates.verify_checksum_manifest_exact(manifest)

    def test_docker_daemon_error_is_not_mistaken_for_absent_volume(self) -> None:
        docker = Path(sys.executable)
        volume = "ai_iaps_full28_fmriprep_work_sub01_all_measured"
        with patch.object(
            gates.subprocess,
            "run",
            return_value=subprocess_result(1, "", "error during connect: daemon unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "inspection failed"):
                gates.docker_volume_exists(docker, volume)
        with patch.object(
            gates.subprocess,
            "run",
            return_value=subprocess_result(
                1, "", "Error response from daemon: get x: no such volume"
            ),
        ):
            self.assertFalse(gates.docker_volume_exists(docker, volume))

    def test_archive_verification_precedes_exact_volume_removal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            docker = root / "docker.exe"
            docker.write_bytes(b"docker")
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            plan = one_session_plan()
            volume = gates._expected_volumes(plan)[0]
            launch = {"sha256": "b" * 64}
            calls: list[list[str]] = []

            def fake_run(argv, **kwargs):
                calls.append(list(argv))
                return subprocess_result(0, volume + "\n", "")

            with patch.object(
                gates.fpost,
                "verify_checksum_manifest",
                return_value={"status": "pass", "manifest_sha256": "a" * 64},
            ) as verified, patch.object(
                gates, "docker_volume_exists", side_effect=[True, False, False]
            ), patch.object(gates.subprocess, "run", side_effect=fake_run):
                receipts = gates.retire_exact_work_volumes(
                    docker, plan, launch, manifest, root / "receipts"
                )
            self.assertGreaterEqual(verified.call_count, 3)
            self.assertEqual(
                calls,
                [[str(docker), "volume", "rm", volume]],
            )
            self.assertEqual(receipts[0]["volume"], volume)

    def test_missing_volume_without_receipt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            docker = root / "docker.exe"
            docker.write_bytes(b"docker")
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with patch.object(
                gates.fpost,
                "verify_checksum_manifest",
                return_value={"status": "pass", "manifest_sha256": "a" * 64},
            ), patch.object(gates, "docker_volume_exists", return_value=False), patch.object(
                gates.subprocess, "run"
            ) as remover:
                with self.assertRaisesRegex(RuntimeError, "absent without"):
                    gates.retire_exact_work_volumes(
                        docker,
                        one_session_plan(),
                        {"sha256": "b" * 64},
                        manifest,
                        root / "receipts",
                    )
            remover.assert_not_called()


def subprocess_result(returncode: int, stdout: str, stderr: str):
    return type(
        "Completed",
        (),
        {"returncode": returncode, "stdout": stdout, "stderr": stderr},
    )()


class GLMsingleFinalGateTests(unittest.TestCase):
    def make_glmsingle(self, root: Path) -> dict:
        n_voxels = 3
        (root / "glmsingle").mkdir(parents=True)
        freeze_path = root / "freeze.json"
        freeze_path.write_text("{}\n", encoding="utf-8")
        cohort_path = root / "cohort.json"
        cohort_path.write_text("{}\n", encoding="utf-8")
        freeze = {
            "random_seed_for_matched_nonhistorical_code": 20260728,
            "glmsingle": {
                "commit": "commit",
                "candidate_noise_pcs": 10,
                "fractional_ridge_grid": [1.0, 0.5],
                "fracridge_vendor": {"record_sha256": "f" * 64},
                "source_tree_sha256": "s" * 64,
                "ordered_patch_sha256": ["p" * 64, "q" * 64],
            }
        }
        provenance = {
            "subject": 1,
            "n_voxels": n_voxels,
            "glmsingle_commit": "commit",
            "n_pcs": 10,
            "fracs": [1.0, 0.5],
            "seed": 20260728,
            "space": "MNI152NLin6Asym",
            "sessionindicator": [1] * 10,
            "run_n_volumes": [218] * 10,
            "glmsingle_vendor_validation": {
                "validated": True,
                "commit": "commit",
                "source_sha256": "s" * 64,
                "patches": [{"sha256": "p" * 64}, {"sha256": "q" * 64}],
            },
            "full_cohort_entry_point": {
                "production_contract_enforced": True,
                "freeze_path": str(freeze_path),
                "freeze_sha256": gates.sha256_file(freeze_path),
                "cohort_config_path": str(cohort_path),
                "cohort_config_sha256": gates.sha256_file(cohort_path),
                "session_indicators": [1] * 10,
                "fracridge_vendor_validation": {
                    "validated": True,
                    "record_sha256": "f" * 64,
                },
            },
        }
        (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        validation = {
            "betasmd_shape": [n_voxels, 1, 1, 600],
            "betasmd_dtype": "float32",
            "betasmd_all_values_finite": True,
            "betasmd_min": 1.0,
            "betasmd_max": 1.0,
        }
        (root / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
        with h5py.File(root / "glmsingle" / gates.TYPE_D_NAME, "w") as handle:
            handle.create_dataset(
                "betasmd", data=np.ones((n_voxels, 1, 1, 600), dtype=np.float32)
            )
            for name in ("HRFindex", "FRACvalue", "R2"):
                values = 0.5 if name == "FRACvalue" else 1.0
                handle.create_dataset(
                    name, data=np.full((n_voxels, 1, 1), values, dtype=np.float32)
                )
        np.save(root / "flat_mask_indices.npy", np.arange(n_voxels))
        nib.save(
            nib.Nifti1Image(np.ones((n_voxels, 1, 1), dtype=np.uint8), np.eye(4)),
            root / "analysis_mask.nii.gz",
        )
        with (root / "trial_manifest.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("beta_index", "run", "image_id", "onset"),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for index in range(600):
                writer.writerow(
                    {
                        "beta_index": index,
                        "run": index // 60 + 1,
                        "image_id": f"image-{index % 120:03d}",
                        "onset": float(index % 60) * 3.0,
                    }
                )
        return freeze, freeze_path, cohort_path

    def test_independently_scans_all_600_type_d_betas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            freeze, freeze_path, cohort_path = self.make_glmsingle(root)
            result = gates.verify_glmsingle(
                root, 1, freeze, freeze_path, cohort_path, [1] * 10
            )
            self.assertEqual(result["typed_shape"], [3, 1, 1, 600])
            self.assertTrue(result["typed_all_finite"])
            with h5py.File(root / "glmsingle" / gates.TYPE_D_NAME, "r+") as handle:
                handle["betasmd"][2, 0, 0, 599] = np.nan
            with self.assertRaisesRegex(RuntimeError, "Nonfinite Type-D"):
                gates.verify_glmsingle(
                    root, 1, freeze, freeze_path, cohort_path, [1] * 10
                )


def subject_rows(subject: int, method: str) -> pd.DataFrame:
    rows = []
    for pipeline in gates.PIPELINES:
        for contrast in gates.CONTRASTS:
            kind = "within" if contrast.startswith("within_") else "cross"
            for roi in gates.ROI_ORDER:
                rows.append(
                    {
                        "erp_method": method,
                        "subject": f"Sub{subject}",
                        "pipeline": pipeline,
                        "analysis_kind": kind,
                        "contrast": contrast,
                        "contrast_label": contrast,
                        "roi": roi,
                        "n_voxels": 20,
                        "accuracy": 0.6,
                    }
                )
    return pd.DataFrame(rows)


def historical_folds(subject: int) -> pd.DataFrame:
    rows = []
    for pipeline in gates.PIPELINES:
        for contrast in gates.CONTRASTS:
            folds = range(1, 5) if contrast.startswith("within_") else (0,)
            for roi in gates.ROI_ORDER:
                for repeat in range(1, 21):
                    for fold in folds:
                        rows.append(
                            {
                                "erp_method": gates.HISTORICAL_METHOD,
                                "subject": f"Sub{subject}",
                                "pipeline": pipeline,
                                "contrast": contrast,
                                "roi": roi,
                                "repeat": repeat,
                                "fold": fold,
                                "accuracy": 0.5,
                            }
                        )
    return pd.DataFrame(rows)


def runwise_folds(subject: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "erp_method": gates.RUNWISE_METHOD,
                "subject": f"Sub{subject}",
                "pipeline": pipeline,
                "contrast": contrast,
                "roi": roi,
                "held_out_run": run,
                "accuracy": 0.5,
            }
            for pipeline in gates.PIPELINES
            for contrast in gates.CONTRASTS
            for roi in gates.ROI_ORDER
            for run in range(1, 11)
        ]
    )


class ERPFinalGateTests(unittest.TestCase):
    def make_erp(self, base: Path) -> tuple[Path, Path, Path]:
        erp = base / "erp"
        glm = base / "glm"
        erp.mkdir()
        glm.mkdir()
        freeze = base / "freeze.json"
        freeze.write_text('{"analysis_id":"test"}\n', encoding="utf-8")
        inputs = {}
        for name in (
            "typed_hdf5",
            "analysis_mask",
            "flat_mask_indices",
            "trial_manifest",
            "validation",
            "provenance",
        ):
            path = glm / f"{name}.dat"
            path.write_bytes(name.encode("ascii"))
            inputs[name] = gates.quick_file_signature(path)
        unsigned = {
            "analysis_id": "test",
            "subject": 1,
            "freeze": {"path": str(freeze), "sha256": gates.sha256_file(freeze)},
            "glmsingle_subject_root": str(glm.resolve()),
            "glmsingle_inputs": inputs,
        }
        signature = gates.stable_json_sha256(unsigned)
        request = {**unsigned, "request_signature_sha256": signature}
        row_counts = {
            "historical_subject_rows": 3 * 8 * 17,
            "historical_fold_rows": 3 * 17 * (4 * 80 + 4 * 20),
            "runwise_subject_rows": 3 * 8 * 17,
            "runwise_fold_rows": 3 * 8 * 17 * 10,
            "roi_count_rows": 3 * 17,
        }
        provenance = {
            "status": "complete",
            "subject": "Sub1",
            "analysis_id": "test",
            "methods": {
                gates.HISTORICAL_METHOD: {},
                gates.RUNWISE_METHOD: {},
            },
            "request_signature_sha256": signature,
            "request": request,
            "row_counts": row_counts,
        }
        validation = {
            "status": "complete",
            "subject": "Sub1",
            "pipelines": list(gates.PIPELINES),
            "roi_order": list(gates.ROI_ORDER),
            "all_accuracy_values_finite_and_in_0_1": True,
            "row_counts": row_counts,
        }
        (erp / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        (erp / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
        subject_rows(1, gates.HISTORICAL_METHOD).to_csv(
            erp / "historical_avg_random_subject_results.csv", index=False
        )
        historical_folds(1).to_csv(
            erp / "historical_avg_random_fold_results.csv", index=False
        )
        subject_rows(1, gates.RUNWISE_METHOD).to_csv(
            erp / "runwise_loro_subject_results.csv", index=False
        )
        runwise_folds(1).to_csv(erp / "runwise_loro_fold_results.csv", index=False)
        pd.DataFrame(
            [
                {
                    "subject": "Sub1",
                    "pipeline": pipeline,
                    "roi": roi,
                    "analysis_voxels": 20,
                }
                for pipeline in gates.PIPELINES
                for roi in gates.ROI_ORDER
            ]
        ).to_csv(erp / "roi_voxel_counts.csv", index=False)
        return erp, glm, freeze

    def test_exact_three_by_two_by_eight_by_seventeen_erp_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            erp, glm, freeze = self.make_erp(Path(temp_name))
            result = gates.verify_erp(erp, 1, freeze, glm)
            self.assertEqual(result["pipelines"], list(gates.PIPELINES))
            self.assertEqual(result["methods"], [gates.HISTORICAL_METHOD, gates.RUNWISE_METHOD])
            self.assertEqual(result["roi_count"], 17)
            self.assertEqual(len(result["contrasts"]), 8)
            path = erp / "runwise_loro_subject_results.csv"
            damaged = pd.read_csv(path).iloc[:-1]
            damaged.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "key mismatch"):
                gates.verify_erp(erp, 1, freeze, glm)


class LedgerIdentityTests(unittest.TestCase):
    def test_current_gate_and_all_prior_stages_share_control_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            attempt_dir = root / "attempt"
            attempt_dir.mkdir()
            start = {
                "event": "started",
                "attempt_id": "gate-attempt",
                "subject": 1,
                "stage": "subject_gate",
                "analysis_id": "analysis",
                "control_identity_sha256": "identity",
                "attempt_dir": str(attempt_dir),
                "event_sha256": "start-hash",
            }
            (attempt_dir / "attempt.json").write_text(
                json.dumps(start), encoding="utf-8"
            )
            events = (
                {
                    "event": "completed",
                    "subject": 1,
                    "stage": "erp",
                    "status": "passed",
                    "control_identity_sha256": "identity",
                    "event_sha256": "erp-hash",
                },
                start,
            )
            state = SimpleNamespace(events=events, head_sha256="start-hash")
            contract = {"stage_order": ["erp", "subject_gate"]}
            with patch.object(
                gates,
                "validate_approval",
                return_value=(
                    {"production_subjects": [1]},
                    {},
                    {"analysis_id": "analysis"},
                    contract,
                    {},
                ),
            ), patch.object(
                gates, "build_control_identity", return_value=({}, "identity")
            ), patch.object(gates, "read_ledger", return_value=state), patch.object(
                gates, "verify_event_mirror"
            ), patch.object(gates, "verify_attempt_evidence"):
                result = gates.verify_ledger(
                    1,
                    root / "freeze.json",
                    root / "contract.json",
                    root / "approval.json",
                    root,
                )
                self.assertEqual(result["control_identity_sha256"], "identity")
                events[0]["control_identity_sha256"] = "wrong"
                with self.assertRaisesRegex(RuntimeError, "identity-matched"):
                    gates.verify_ledger(
                        1,
                        root / "freeze.json",
                        root / "contract.json",
                        root / "approval.json",
                        root,
                    )


class ActiveContractTests(unittest.TestCase):
    def test_only_proven_postprocessing_commands_are_exposed(self) -> None:
        contract, _ = production_control.load_contract(
            ROOT / "config" / "stage_contracts.json"
        )
        for stage in ("visual_qc", "archive", "subject_gate"):
            self.assertEqual(contract["stages"][stage]["state"], "executable")
            self.assertIn(
                "{postproc_gates_script}", contract["stages"][stage]["command_argv"]
            )
        visual = contract["stages"]["visual_qc"]["command_argv"]
        self.assertIn("/usr/bin/inkscape", visual)
        self.assertIn("ad5b4926bc77472d0c730f968b1aaab2028d784f57d419f6efdffd5d0f9859b9", visual)
        self.assertIn(
            "--execute-retirement", contract["stages"]["archive"]["command_argv"]
        )
        final = contract["stages"]["subject_gate"]["command_argv"]
        for item in ("--approval", "--log-root", "--archive-root", "--glmsingle-root", "--erp-root"):
            self.assertIn(item, final)


if __name__ == "__main__":
    unittest.main()
