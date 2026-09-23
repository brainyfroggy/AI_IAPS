from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import aggregate_erp_results as aggregate
import cohort_analysis_common as common
import decode_erp_subject as subject_decoder
import run_glmsingle


def subject_rows(subject: int, method: str) -> pd.DataFrame:
    rows = []
    for pipeline in common.PIPELINES:
        for contrast in common.CONTRASTS:
            kind = "within" if contrast in common.WITHIN_CONTRASTS else "cross"
            for roi in common.ROI_ORDER:
                rows.append(
                    {
                        "erp_method": method,
                        "subject": common.subject_label(subject),
                        "pipeline": pipeline,
                        "analysis_kind": kind,
                        "contrast": contrast,
                        "contrast_label": contrast,
                        "roi": roi,
                        "n_voxels": 20,
                        "accuracy": 0.625,
                    }
                )
    return pd.DataFrame(rows)


def historical_fold_rows(subject: int) -> pd.DataFrame:
    rows = []
    for pipeline in common.PIPELINES:
        for contrast in common.CONTRASTS:
            folds = range(1, 5) if contrast in common.WITHIN_CONTRASTS else (0,)
            for roi in common.ROI_ORDER:
                for repeat in range(1, 21):
                    for fold in folds:
                        rows.append(
                            {
                                "erp_method": common.HISTORICAL_METHOD,
                                "subject": common.subject_label(subject),
                                "pipeline": pipeline,
                                "contrast": contrast,
                                "roi": roi,
                                "repeat": repeat,
                                "fold": fold,
                                "accuracy": 0.5,
                            }
                        )
    return pd.DataFrame(rows)


def runwise_fold_rows(subject: int) -> pd.DataFrame:
    rows = []
    for pipeline in common.PIPELINES:
        for contrast in common.CONTRASTS:
            for roi in common.ROI_ORDER:
                for held_out_run in range(1, 11):
                    rows.append(
                        {
                            "erp_method": common.RUNWISE_METHOD,
                            "subject": common.subject_label(subject),
                            "pipeline": pipeline,
                            "contrast": contrast,
                            "roi": roi,
                            "held_out_run": held_out_run,
                            "accuracy": 0.5,
                        }
                    )
    return pd.DataFrame(rows)


class FreezeAndSessionTests(unittest.TestCase):
    def test_freeze_is_exact_28_subject_contract(self) -> None:
        freeze = common.load_freeze(ROOT / "ANALYSIS_FREEZE.json")
        self.assertEqual(len(freeze["included_subjects"]), 28)
        self.assertEqual(tuple(freeze["erp_decoding"]["roi_order"]), common.ROI_ORDER)

    def test_session_indicators_are_config_driven_and_bids_checked(self) -> None:
        config = common.load_cohort_config(ROOT / "config" / "cohort.json")
        expected = common.expected_session_indicators(config, 11)
        self.assertEqual(expected, [1] * 6 + [2] * 4)
        paths = [
            Path(f"sub-11/ses-{session:02d}/func/sub-11_run-{run:02d}_bold.nii.gz")
            for run, session in enumerate(expected, start=1)
        ]
        observed, source = common.resolve_session_indicators(paths, 11, config)
        self.assertEqual(observed, expected)
        self.assertIn("cross-checked", source)
        paths[-1] = Path("sub-11/ses-01/func/sub-11_run-10_bold.nii.gz")
        with self.assertRaises(ValueError):
            common.resolve_session_indicators(paths, 11, config)

    def test_glmsingle_parser_accepts_any_frozen_subject(self) -> None:
        args = run_glmsingle.parse_args(
            [
                "--subject",
                "31",
                "--fmriprep-root",
                "derivatives",
                "--bids-root",
                "bids",
                "--validate-only",
            ]
        )
        self.assertEqual(args.subject, 31)

    def test_glmsingle_contract_exact(self) -> None:
        freeze = common.load_freeze(ROOT / "ANALYSIS_FREEZE.json")
        frozen = freeze["glmsingle"]
        pilot = SimpleNamespace(
            GLMSINGLE_COMMIT=frozen["commit"],
            GLMSINGLE_PATCH_SPECS=[
                ("a", Path("a"), frozen["ordered_patch_sha256"][0]),
                ("b", Path("b"), frozen["ordered_patch_sha256"][1]),
            ],
            DEFAULT_FRACS=tuple(frozen["fractional_ridge_grid"]),
            TR=freeze["acquisition"]["tr_seconds"],
            STIMDUR=freeze["acquisition"]["stimulus_duration_seconds"],
            SEED=freeze["random_seed_for_matched_nonhistorical_code"],
        )
        checks = run_glmsingle.validate_pinned_contract(pilot, freeze)
        self.assertTrue(all(checks.values()))
        pilot.GLMSINGLE_COMMIT = "wrong"
        with self.assertRaises(RuntimeError):
            run_glmsingle.validate_pinned_contract(pilot, freeze)

    def test_fracridge_vendor_is_record_pinned_and_importable(self) -> None:
        validation = run_glmsingle.validate_fracridge_vendor()
        self.assertTrue(validation["validated"])
        self.assertEqual(validation["package"], "fracridge")
        self.assertEqual(validation["version"], "2.0")
        self.assertEqual(
            validation["record_sha256"],
            run_glmsingle.FRACRIDGE_RECORD_SHA256,
        )
        self.assertEqual(validation["verified_hashed_file_count"], 12)
        self.assertTrue(
            Path(validation["module_path"]).is_relative_to(
                run_glmsingle.VENDORED_PYTHON_DEPENDENCIES.resolve()
            )
        )

    def test_production_destination_rejects_nonproduction_tokens(self) -> None:
        freeze = common.load_freeze(ROOT / "ANALYSIS_FREEZE.json")
        args = SimpleNamespace(
            nonproduction=False,
            n_pcs=10,
            fixed_frac=None,
            fracs=None,
            max_voxels=None,
            overwrite=False,
            output=Path("results") / "smoke_v1" / "Sub31",
        )
        with self.assertRaises(ValueError):
            run_glmsingle.enforce_production_options(args, freeze)

    def test_actual_pilot_erp_contract_and_self_test(self) -> None:
        freeze = common.load_freeze(ROOT / "ANALYSIS_FREEZE.json")
        pilot = subject_decoder.load_pilot_module()
        checks = subject_decoder.validate_pinned_contract(pilot, freeze)
        self.assertTrue(all(checks.values()))
        pilot.run_self_test()


class RowCompletenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subject = 31
        cls.historical_subject = subject_rows(
            cls.subject, common.HISTORICAL_METHOD
        )
        cls.runwise_subject = subject_rows(cls.subject, common.RUNWISE_METHOD)
        cls.historical_folds = historical_fold_rows(cls.subject)
        cls.runwise_folds = runwise_fold_rows(cls.subject)

    def test_exact_subject_keys(self) -> None:
        common.validate_subject_result_rows(
            self.historical_subject, self.subject, common.HISTORICAL_METHOD
        )
        broken = self.historical_subject.iloc[:-1].copy()
        with self.assertRaises(ValueError):
            common.validate_subject_result_rows(
                broken, self.subject, common.HISTORICAL_METHOD
            )

    def test_duplicate_subject_key_rejected(self) -> None:
        duplicate = pd.concat(
            [self.runwise_subject, self.runwise_subject.iloc[[0]]], ignore_index=True
        )
        with self.assertRaises(ValueError):
            common.validate_subject_result_rows(
                duplicate, self.subject, common.RUNWISE_METHOD
            )

    def test_historical_fold_completeness(self) -> None:
        self.assertEqual(len(self.historical_folds), 20_400)
        common.validate_fold_result_rows(
            self.historical_folds, self.subject, common.HISTORICAL_METHOD
        )
        broken = self.historical_folds.drop(index=0).reset_index(drop=True)
        with self.assertRaises(ValueError):
            common.validate_fold_result_rows(
                broken, self.subject, common.HISTORICAL_METHOD
            )
        wrong_key = self.historical_folds.copy()
        first_group = (
            (wrong_key["pipeline"] == common.PIPELINES[0])
            & (wrong_key["contrast"] == common.CONTRASTS[0])
            & (wrong_key["roi"] == common.ROI_ORDER[0])
            & (wrong_key["repeat"] == 20)
            & (wrong_key["fold"] == 4)
        )
        wrong_key.loc[first_group, "repeat"] = 21
        with self.assertRaises(ValueError):
            common.validate_fold_result_rows(
                wrong_key, self.subject, common.HISTORICAL_METHOD
            )

    def test_runwise_fold_completeness(self) -> None:
        self.assertEqual(len(self.runwise_folds), 4_080)
        common.validate_fold_result_rows(
            self.runwise_folds, self.subject, common.RUNWISE_METHOD
        )
        broken = self.runwise_folds[
            self.runwise_folds["held_out_run"] != 10
        ].copy()
        with self.assertRaises(ValueError):
            common.validate_fold_result_rows(
                broken, self.subject, common.RUNWISE_METHOD
            )

    def test_roi_count_exact_keys(self) -> None:
        rows = [
            {
                "subject": common.subject_label(self.subject),
                "pipeline": pipeline,
                "roi": roi,
                "analysis_voxels": 20,
            }
            for pipeline in common.PIPELINES
            for roi in common.ROI_ORDER
        ]
        common.validate_roi_count_rows(pd.DataFrame(rows), self.subject)

    def test_dynamic_labels_never_claim_three_subjects(self) -> None:
        frame = pd.concat(
            [
                self.historical_subject.assign(subject=f"Sub{subject}")
                for subject in range(1, 29)
            ],
            ignore_index=True,
        )
        labels = common.dynamic_pipeline_labels(frame)
        self.assertTrue(all("n=28" in label for label in labels.values()))
        summary_labels = common.dynamic_pipeline_labels(
            frame.groupby("pipeline", sort=False)
            .agg(n_subjects=("subject", "nunique"))
            .reset_index()
        )
        self.assertEqual(labels, summary_labels)
        for path in CODE.glob("*.py"):
            self.assertNotIn("n=3", path.read_text(encoding="utf-8"))


class RoiMapTests(unittest.TestCase):
    def test_roi_map_is_labeled_nonvoxelwise(self) -> None:
        names = [
            "V1v",
            "V1d",
            "V2v",
            "V2d",
            "V3v",
            "V3d",
            "hV4",
            "V3a",
            "V3b",
            "IPS0",
            "IPS1",
            "IPS2",
            "IPS3",
            "IPS4",
            "IPS5",
            "LO1",
            "LO2",
            "hMT",
            "VO1",
            "VO2",
            "PHC1",
            "PHC2",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas = root / "kastner.nii.gz"
            labels = root / "kastner.nii.txt"
            data = np.arange(1, len(names) + 1, dtype=np.int16).reshape(-1, 1, 1)
            nib.save(nib.Nifti1Image(data, np.eye(4)), atlas)
            labels.write_text(
                "\n".join(f"{index} {name}" for index, name in enumerate(names, 1)),
                encoding="utf-8",
            )
            summary = pd.DataFrame(
                [
                    {
                        "erp_method": common.HISTORICAL_METHOD,
                        "pipeline": "legacy_lsa_8mm",
                        "contrast": common.CONTRASTS[0],
                        "roi": roi,
                        "mean_accuracy": 0.6,
                        "n_subjects": 28,
                    }
                    for roi in common.ROI_ORDER
                ]
            )
            output = root / "maps"
            self.assertEqual(
                aggregate.export_roi_value_maps(summary, atlas, labels, output), 1
            )
            sidecar_path = next(output.glob("*.json"))
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertTrue(sidecar["NotVoxelwiseDecoding"])
            image = nib.load(str(next(output.glob("*.nii.gz"))))
            self.assertTrue(np.allclose(image.get_fdata(), 0.6))


class AggregationIntegrationTests(unittest.TestCase):
    def test_one_subject_partial_aggregation_is_atomic_and_roi_only(self) -> None:
        subject = 4
        pilot = subject_decoder.load_pilot_module()
        reference = pilot.load_historical(aggregate.DEFAULT_HISTORICAL)
        reference = reference[reference["subject"] == common.subject_label(subject)]
        lookup = reference.set_index(["contrast", "roi"])["accuracy"]

        historical = subject_rows(subject, common.HISTORICAL_METHOD)
        legacy_mask = historical["pipeline"] == "legacy_lsa_8mm"
        historical.loc[legacy_mask, "accuracy"] = [
            lookup.loc[(contrast, roi)]
            for contrast, roi in historical.loc[
                legacy_mask, ["contrast", "roi"]
            ].itertuples(index=False)
        ]
        runwise = subject_rows(subject, common.RUNWISE_METHOD)
        counts = pd.DataFrame(
            [
                {
                    "subject": common.subject_label(subject),
                    "pipeline": pipeline,
                    "roi": roi,
                    "analysis_voxels": 20,
                }
                for pipeline in common.PIPELINES
                for roi in common.ROI_ORDER
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject_root = root / "subjects" / common.subject_label(subject)
            subject_root.mkdir(parents=True)
            historical.to_csv(
                subject_root / "historical_avg_random_subject_results.csv",
                index=False,
            )
            historical_fold_rows(subject).to_csv(
                subject_root / "historical_avg_random_fold_results.csv", index=False
            )
            runwise.to_csv(subject_root / "runwise_loro_subject_results.csv", index=False)
            runwise_fold_rows(subject).to_csv(
                subject_root / "runwise_loro_fold_results.csv", index=False
            )
            counts.to_csv(subject_root / "roi_voxel_counts.csv", index=False)
            (subject_root / "validation.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            (subject_root / "provenance.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "request": {
                            "freeze": {
                                "sha256": common.sha256_file(
                                    ROOT / "ANALYSIS_FREEZE.json"
                                )
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            subject_decoder.validate_existing_output(subject_root, subject)
            output = root / "aggregate"
            aggregate.run(
                Namespace(
                    subject_results_root=[root / "subjects"],
                    subjects=[subject],
                    output=output,
                    freeze=ROOT / "ANALYSIS_FREEZE.json",
                    historical_result=aggregate.DEFAULT_HISTORICAL,
                    pilot_script=subject_decoder.PILOT_SCRIPT,
                    allow_partial=True,
                    export_roi_maps=False,
                    atlas=Path(r"C:\MRIcroGL\Resources\atlas\kastner.nii.gz"),
                    atlas_labels=Path(r"C:\MRIcroGL\Resources\atlas\kastner.nii.txt"),
                )
            )
            validation = json.loads(
                (output / "validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validation["n_subjects"], 1)
            self.assertFalse(validation["exact_subject_set"])
            summary = pd.read_csv(output / "group_summary.csv")
            self.assertEqual(set(summary["roi"]), set(common.ROI_ORDER))
            self.assertTrue((output / "kastner_roi_historical_avg_random.png").is_file())
            self.assertTrue((output / "kastner_roi_runwise_loro.png").is_file())


if __name__ == "__main__":
    unittest.main()
