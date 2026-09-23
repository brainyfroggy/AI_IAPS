from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

import fmriprep_postprocess as post  # noqa: E402
import fmriprep_sdc_workflow as workflow  # noqa: E402
import fmriprep_sdc_workflow_v6 as workflow_v6  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_file(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def root_files(root: Path, subject: int) -> None:
    write_json(root / "dataset_description.json", {"Name": "test", "BIDSVersion": "1.10.0"})
    (root / "participants.tsv").write_text(
        f"participant_id\nsub-{subject:02d}\n", encoding="utf-8"
    )
    write_json(root / "task-iaps_events.json", {"onset": {"Description": "test"}})


def add_anat(root: Path, subject: int, session: int) -> None:
    prefix = f"sub-{subject:02d}_ses-{session:02d}_T1w"
    base = root / f"sub-{subject:02d}" / f"ses-{session:02d}" / "anat"
    write_file(base / f"{prefix}.nii.gz")
    write_json(base / f"{prefix}.json", {})


def add_bold(root: Path, subject: int, session: int, run: int, mode: str) -> str:
    prefix = f"sub-{subject:02d}_ses-{session:02d}_task-iaps_run-{run:02d}"
    base = root / f"sub-{subject:02d}" / f"ses-{session:02d}" / "func"
    write_file(base / f"{prefix}_bold.nii.gz")
    metadata = {
        "TaskName": "iaps",
        "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.05,
    }
    if mode != "syn":
        metadata["B0FieldSource"] = f"sub-{subject:02d}_ses-{session:02d}_fmap0"
    write_json(base / f"{prefix}_bold.json", metadata)
    (base / f"{prefix}_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0\t3\tpleasant\n", encoding="utf-8"
    )
    return "bids::" + str((base / f"{prefix}_bold.nii.gz").relative_to(root)).replace("\\", "/")


def add_gre(root: Path, subject: int, session: int, intended: list[str]) -> None:
    base = root / f"sub-{subject:02d}" / f"ses-{session:02d}" / "fmap"
    identifier = f"sub-{subject:02d}_ses-{session:02d}_fmap0"
    for role in ("magnitude1", "magnitude2", "phasediff"):
        prefix = f"sub-{subject:02d}_ses-{session:02d}_{role}"
        write_file(base / f"{prefix}.nii.gz")
        metadata = {"B0FieldIdentifier": identifier, "IntendedFor": intended}
        if role == "phasediff":
            metadata.update({"EchoTime1": 0.00492, "EchoTime2": 0.00738})
        write_json(base / f"{prefix}.json", metadata)


def add_pepolar(root: Path, subject: int, session: int, intended: list[str]) -> None:
    base = root / f"sub-{subject:02d}" / f"ses-{session:02d}" / "fmap"
    identifier = f"sub-{subject:02d}_ses-{session:02d}_fmap0"
    for direction, pe in (("AP", "j-"), ("PA", "j")):
        prefix = f"sub-{subject:02d}_ses-{session:02d}_dir-{direction}_epi"
        write_file(base / f"{prefix}.nii.gz")
        write_json(
            base / f"{prefix}.json",
            {
                "B0FieldIdentifier": identifier,
                "IntendedFor": intended,
                "PhaseEncodingDirection": pe,
                "TotalReadoutTime": 0.05,
            },
        )


def mixed_bids(root: Path) -> None:
    root_files(root, 17)
    add_anat(root, 17, 1)
    add_anat(root, 17, 2)
    intended = [add_bold(root, 17, 1, run, "gre") for run in range(1, 7)]
    add_gre(root, 17, 1, intended)
    for run in range(7, 11):
        add_bold(root, 17, 2, run, "syn")


def pepolar_bids(root: Path) -> None:
    root_files(root, 1)
    add_anat(root, 1, 1)
    intended = [add_bold(root, 1, 1, run, "pepolar") for run in range(1, 11)]
    add_pepolar(root, 1, 1, intended)


def derivative_branch(root: Path, plan: workflow.SubjectPlan, branch: workflow.BranchSpec) -> None:
    write_json(
        root / "dataset_description.json",
        {"GeneratedBy": [{"Name": "fMRIPrep", "Version": "25.1.3"}]},
    )
    write_file(root / f"sub-{plan.subject_label}.html", b"report")
    write_file(root / f"sub-{plan.subject_label}" / "anat" / "common.txt", b"same anatomy")
    for session in branch.sessions:
        spec = next(item for item in plan.sessions if item.session == session)
        method = {
            "gre": "FMB (fieldmap-based)",
            "syn": 'FLB ("fieldmap-less", SyN-based)',
            "pepolar": "PEB/PEPOLAR (phase-encoding based)",
        }[spec.mode]
        for run in spec.runs:
            prefix = f"sub-{plan.subject_label}_ses-{session}_task-iaps_run-{run:02d}"
            func = root / f"sub-{plan.subject_label}" / f"ses-{session}" / "func"
            write_file(func / f"{prefix}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")
            write_json(func / f"{prefix}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.json", {})
            write_file(func / f"{prefix}_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz")
            (func / f"{prefix}_desc-confounds_timeseries.tsv").write_text("x\n0\n", encoding="utf-8")
            figures = root / f"sub-{plan.subject_label}" / "figures"
            write_file(
                figures / f"{prefix}_desc-summary_bold.html",
                f"<li>Susceptibility distortion correction: {method}</li>".encode(),
            )


class SDCWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cohort, cls.sdc = workflow.load_configs(
            workflow.DEFAULT_COHORT_CONFIG, workflow.DEFAULT_SDC_CONFIG
        )

    def setUp(self) -> None:
        # Keep fixture paths short enough for Windows tools that are not
        # long-path aware while still exercising the mandatory N: storage gate.
        scratch = Path(r"N:\Experimental_Data\yujunchen\fmriprep_sdc_unit_tests")
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_all_25_subject_plans_and_frozen_strategies(self) -> None:
        remaining = self.cohort["cohort"]["remaining_subjects"]
        plans = [workflow.subject_plan(self.cohort, subject) for subject in remaining]
        self.assertEqual(len(plans), 25)
        self.assertEqual(
            [plan.subject for plan in plans],
            [1, 2, 7, 9, *range(11, 32)],
        )
        mixed = [plan.subject for plan in plans if plan.strategy == "isolated_mixed_session_branches"]
        self.assertEqual(mixed, [17, 18, 20, 21, 24, 26, 28, 29, 30, 31])
        self.assertTrue(all({run for session in plan.sessions for run in session.runs} == set(range(1, 11)) for plan in plans))
        expected_modes = {
            1: ("pepolar",),
            2: ("pepolar",),
            7: ("pepolar", "pepolar"),
            9: ("pepolar", "pepolar"),
            11: ("pepolar", "pepolar"),
            12: ("pepolar", "pepolar"),
            13: ("pepolar", "pepolar"),
            14: ("pepolar", "pepolar"),
            15: ("pepolar", "pepolar"),
            16: ("pepolar", "pepolar"),
            17: ("gre", "syn"),
            18: ("gre", "syn"),
            19: ("gre", "gre"),
            20: ("gre", "syn"),
            21: ("gre", "syn"),
            22: ("gre", "gre"),
            23: ("gre", "gre"),
            24: ("syn", "gre"),
            25: ("gre", "gre"),
            26: ("syn", "gre"),
            27: ("gre", "gre"),
            28: ("syn", "gre"),
            29: ("syn", "gre"),
            30: ("syn", "gre"),
            31: ("syn", "gre"),
        }
        self.assertEqual(set(expected_modes), set(remaining))
        for plan in plans:
            with self.subTest(subject=plan.subject):
                self.assertEqual(tuple(item.mode for item in plan.sessions), expected_modes[plan.subject])
                if len(plan.sessions) == 1:
                    self.assertEqual(plan.sessions[0].runs, tuple(range(1, 11)))
                else:
                    self.assertEqual(plan.sessions[0].runs, tuple(range(1, 7)))
                    self.assertEqual(plan.sessions[1].runs, tuple(range(7, 11)))

    def test_pepolar_metadata_association_is_same_session_and_complete(self) -> None:
        bids = self.root / "bids"
        pepolar_bids(bids)
        plan = workflow.subject_plan(self.cohort, 1)
        result = workflow.audit_bids_branch(bids, plan, plan.branches[0])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["runs"], list(range(1, 11)))
        self.assertEqual(result["associations"]["01"]["fieldmaps"], 2)

    def test_persisted_branch_identity_survives_json_round_trip(self) -> None:
        plan = workflow.subject_plan(self.cohort, 1)
        branch = plan.branches[0]
        persisted = json.loads(json.dumps(workflow.branch_payload(branch)))
        self.assertEqual(persisted, workflow.branch_payload(branch))
        self.assertEqual(persisted["sessions"], ["01"])
        self.assertEqual(persisted["runs"], list(range(1, 11)))

    def test_mixed_branch_keeps_all_anatomy_and_no_other_session_func_or_fmap(self) -> None:
        bids = self.root / "bids"
        mixed_bids(bids)
        plan = workflow.subject_plan(self.cohort, 17)
        for branch in plan.branches:
            destination = self.root / "branches" / branch.name
            workflow.build_isolated_branch(bids, destination, plan, branch, "copy")
            result = workflow.audit_bids_branch(destination, plan, branch)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                len(list(destination.glob("sub-17/ses-*/anat/*_T1w.nii.gz"))), 2
            )
            other = "02" if branch.sessions == ("01",) else "01"
            self.assertFalse((destination / "sub-17" / f"ses-{other}" / "func").exists())
            self.assertFalse((destination / "sub-17" / f"ses-{other}" / "fmap").exists())

    def test_syn_branch_rejects_measured_fieldmap_reference(self) -> None:
        bids = self.root / "bids"
        mixed_bids(bids)
        plan = workflow.subject_plan(self.cohort, 17)
        syn = next(branch for branch in plan.branches if branch.mode == "syn")
        destination = self.root / "syn"
        workflow.build_isolated_branch(bids, destination, plan, syn, "copy")
        bold_json = next(destination.glob("sub-17/ses-02/func/*_bold.json"))
        metadata = json.loads(bold_json.read_text())
        metadata["B0FieldSource"] = "sub-17_ses-01_fmap0"
        write_json(bold_json, metadata)
        with self.assertRaisesRegex(RuntimeError, "SyN BOLD references"):
            workflow.audit_bids_branch(destination, plan, syn)

    def test_local_preflight_builds_immutable_mixed_receipt_but_cannot_launch(self) -> None:
        bids = self.root / "bids"
        mixed_bids(bids)
        plan = workflow.subject_plan(self.cohort, 17)
        branch_root = self.root / "branches"
        output = self.root / "preflight"
        result = workflow.preflight_subject(
            plan,
            workflow.DEFAULT_COHORT_CONFIG,
            workflow.DEFAULT_SDC_CONFIG,
            bids,
            branch_root,
            output,
            "copy",
            False,
        )
        self.assertEqual(result["status"], "dry_run_local_only")
        self.assertEqual(len(result["branches"]), 2)
        self.assertTrue((output / "preflight_summary.json").is_file())
        with self.assertRaisesRegex(RuntimeError, "completed external"):
            workflow.verify_preflight_receipt(
                output,
                plan,
                workflow.DEFAULT_COHORT_CONFIG,
                workflow.DEFAULT_SDC_CONFIG,
            )

    def test_commands_pin_image_space_no_reconall_and_syn_error(self) -> None:
        plan = workflow.subject_plan(self.cohort, 17)
        syn = next(branch for branch in plan.branches if branch.mode == "syn")
        command = workflow.fmriprep_command(
            Path(r"N:\bids"),
            Path(r"N:\out"),
            Path(r"C:\license.txt"),
            "safe_volume",
            plan,
            syn,
            self.sdc,
        )
        self.assertIn(self.sdc["fmriprep"]["image"], command)
        self.assertIn("--fs-no-reconall", command)
        self.assertEqual(command[command.index("--use-syn-sdc") + 1], "error")
        self.assertEqual(command[command.index("--output-spaces") + 1], "MNI152NLin6Asym:res-2")
        self.assertIn("--bids-filter-file", command)
        self.assertEqual(
            command[:5],
            [r"C:\Windows\System32\wsl.exe", "-d", "Ubuntu-22.04", "--", "/usr/bin/docker"],
        )
        self.assertEqual(command[5:10], ["run", "--rm", "--pull", "never", "--name"])
        mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
        self.assertIn("type=bind,src=/mnt/n/bids,dst=/data,readonly", mounts)
        self.assertIn("type=bind,src=/mnt/n/out,dst=/out", mounts)
        self.assertIn(
            "type=bind,src=/mnt/c/license.txt,dst=/opt/freesurfer/license.txt,readonly",
            mounts,
        )
        self.assertTrue(any("src=/mnt/n/" in value and "dst=/bids-filter.json" in value for value in mounts))
        self.assertTrue(all("N:\\" not in value and not value.startswith("type=bind,src=\\\\") for value in mounts))

    def test_every_subject_branch_uses_wsl_docker_and_mnt_n_without_cross_mode_flags(self) -> None:
        prefix = [r"C:\Windows\System32\wsl.exe", "-d", "Ubuntu-22.04", "--", "/usr/bin/docker"]
        for subject in self.cohort["cohort"]["remaining_subjects"]:
            plan = workflow.subject_plan(self.cohort, subject)
            for branch in plan.branches:
                with self.subTest(subject=subject, branch=branch.name, mode=branch.mode):
                    bids = Path(rf"N:\runtime\bids\Sub{subject:02d}\{branch.name}")
                    output = Path(rf"N:\runtime\fmriprep\Sub{subject:02d}\{branch.name}")
                    command = workflow.fmriprep_command(
                        bids,
                        output,
                        Path(r"C:\Users\yujunchen\.cache\ai_iaps_fmriprep_license.txt"),
                        f"work_sub{subject:02d}_{branch.name}",
                        plan,
                        branch,
                        self.sdc,
                    )
                    self.assertEqual(command[:5], prefix)
                    self.assertEqual(command[command.index("--use-syn-sdc") + 1], "error" if branch.mode == "syn" else "warn")
                    self.assertEqual("--bids-filter-file" in command, branch.mode == "syn")
                    mounts = [
                        command[index + 1]
                        for index, value in enumerate(command)
                        if value == "--mount"
                    ]
                    persistent = [value for value in mounts if "dst=/data" in value or "dst=/out" in value or "dst=/bids-filter.json" in value]
                    self.assertTrue(persistent)
                    self.assertTrue(all("src=/mnt/n/" in value for value in persistent))
                    self.assertTrue(all("N:\\" not in value and "\\\\ad.ufl.edu" not in value for value in mounts))

                    probe = workflow.docker_probe_command(
                        bids,
                        plan,
                        branch,
                        self.sdc,
                        f"probe-sub{subject:02d}-{branch.name}",
                    )
                    self.assertEqual(probe[:5], prefix)
                    self.assertIn("--network", probe)
                    self.assertEqual(probe[probe.index("--network") + 1], "none")
                    probe_mounts = [
                        probe[index + 1]
                        for index, value in enumerate(probe)
                        if value == "--mount"
                    ]
                    self.assertTrue(all("src=/mnt/n/" in value for value in probe_mounts))
                    self.assertTrue(all("N:\\" not in value and "\\\\ad.ufl.edu" not in value for value in probe_mounts))

    def test_merge_requires_identical_anatomical_reference(self) -> None:
        plan = workflow.subject_plan(self.cohort, 17)
        outputs = {}
        for branch in plan.branches:
            path = self.root / "outputs" / branch.name
            derivative_branch(path, plan, branch)
            outputs[branch.name] = path
        destination = self.root / "final"
        result = workflow.merge_mixed_branches(plan, outputs, destination, "copy")
        self.assertEqual(result["strategy"], plan.strategy)
        self.assertEqual(
            len(list(destination.glob("sub-17/ses-*/func/*_desc-preproc_bold.nii.gz"))), 10
        )

        second = next(branch for branch in plan.branches if branch.mode == "syn")
        (outputs[second.name] / "sub-17" / "anat" / "common.txt").write_bytes(b"different")
        with self.assertRaisesRegex(RuntimeError, "different subject anatomical"):
            workflow.merge_mixed_branches(plan, outputs, self.root / "bad-final", "copy")

    def test_method_markers_are_exclusive(self) -> None:
        self.assertTrue(workflow.method_matches("pepolar", "PEB/PEPOLAR"))
        self.assertTrue(workflow.method_matches("gre", "FMB (fieldmap-based)"))
        self.assertTrue(workflow.method_matches("syn", "FLB (fieldmap-less, SyN-based)"))
        self.assertFalse(workflow.method_matches("gre", "fieldmap-less SyN"))
        self.assertFalse(workflow.method_matches("pepolar", "None"))

    def test_checksum_manifest_and_attestation_gate(self) -> None:
        derivative = self.root / "derivative"
        qc = self.root / "qc"
        write_file(derivative / "a.bin", b"abc")
        write_file(qc / "b.bin", b"def")
        manifest = self.root / "manifest.json"
        payload = post.create_checksum_manifest(
            (("fmriprep", derivative), ("qc", qc)), manifest
        )
        self.assertEqual(payload["file_count"], 2)
        self.assertEqual(post.verify_checksum_manifest(manifest)["status"], "pass")
        write_file(derivative / "a.bin", b"changed")
        with self.assertRaisesRegex(RuntimeError, "Checksum verification failed"):
            post.verify_checksum_manifest(manifest)

    def test_pending_visual_attestation_cannot_pass(self) -> None:
        plan = workflow.subject_plan(self.cohort, 17)
        path = self.root / "attestation.tsv"
        fields = [
            "subject", "session", "run", "expected_sdc_mode", "sdc_review",
            "coreg_review", "carpet_motion_review", "overall_decision", "reviewer",
            "reviewed_at_utc", "notes",
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for session in plan.sessions:
                for run in session.runs:
                    writer.writerow(
                        {
                            "subject": "Sub17", "session": f"ses-{session.session}",
                            "run": f"run-{run:02d}", "expected_sdc_mode": session.mode,
                            "sdc_review": "PENDING", "coreg_review": "PENDING",
                            "carpet_motion_review": "PENDING", "overall_decision": "PENDING",
                            "reviewer": "", "reviewed_at_utc": "", "notes": "",
                        }
                    )
        with self.assertRaisesRegex(RuntimeError, "incomplete or rejected"):
            post.validate_attestation(path, plan)

    def test_qc_wrapper_generates_pending_not_accepted_attestation(self) -> None:
        plan = workflow.subject_plan(self.cohort, 17)
        derivative = self.root / "derivative"
        derivative.mkdir()
        output = self.root / "qc"

        def fake_run(command, check):
            del check
            destination = Path(command[command.index("--output") + 1])
            destination.mkdir(parents=True)
            return mock.Mock(returncode=0)

        with mock.patch.object(post.subprocess, "run", side_effect=fake_run):
            result = post.run_qc_bundle(
                derivative,
                plan,
                output,
                Path(sys.executable),
                post.DEFAULT_QUANT_SCRIPT,
                post.DEFAULT_VISUAL_SCRIPT,
                None,
            )
        self.assertEqual(result["status"], "artifacts_generated_human_review_pending")
        attestation = output / "visual_review_attestation.tsv"
        with attestation.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["overall_decision"] for row in rows}, {"PENDING"})
        with self.assertRaisesRegex(RuntimeError, "incomplete or rejected"):
            post.validate_attestation(attestation, plan)

    def test_v6_tree_identity_includes_extended_length_windows_file(self) -> None:
        base = Path(tempfile.mkdtemp())
        try:
            long_file = base / ("a" * 90) / ("b" * 90) / ("c" * 90) / "payload.bin"
            workflow_v6.extended_path(long_file.parent).mkdir(parents=True)
            workflow_v6.extended_path(long_file).write_bytes(b"complete-long-path-file")
            self.assertGreater(len(str(long_file)), 260)
            self.assertEqual(len(workflow_v6.iter_files(base)), 1)
            identity = workflow_v6.tree_identity(base)
            self.assertEqual(identity["file_count"], 1)
            self.assertEqual(identity["total_bytes"], len(b"complete-long-path-file"))
        finally:
            shutil.rmtree(workflow_v6.extended_path(base), ignore_errors=True)

    def test_v6_recovery_publication_preserves_source_and_creates_parent(self) -> None:
        source = self.root / "prior" / "derivative"
        destination = self.root / "new" / "nested" / "Sub01"
        write_file(source / "dataset_description.json", b"{}")
        write_file(source / "sub-01" / "file.bin", b"verified")
        before = workflow_v6.tree_identity(source)
        with mock.patch.object(
            workflow_v6, "require_n_drive", side_effect=lambda path, *_args, **_kwargs: path
        ):
            result = workflow_v6.publish_recovered_tree(source, destination, "copy")
        self.assertTrue(source.is_dir())
        self.assertTrue(destination.is_dir())
        self.assertEqual(workflow_v6.tree_identity(source), before)
        self.assertEqual(workflow_v6.tree_identity(destination), before)
        self.assertEqual(result["strategy"], "verified_prior_attempt_publication")


if __name__ == "__main__":
    unittest.main()
