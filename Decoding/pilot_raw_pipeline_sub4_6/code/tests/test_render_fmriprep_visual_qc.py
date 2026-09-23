"""Synthetic tests for read-only fMRIPrep visual-reportlet rendering."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import render_fmriprep_visual_qc as visual_qc  # noqa: E402


FEBRUARY_RUNS = (1, 3, 4, 5)


def synthetic_svg(*, layered: bool) -> str:
    if layered:
        body = """
  <g class="background-svg"><rect width="40" height="20" fill="navy"/></g>
  <g class="foreground-svg"><rect width="40" height="20" fill="orange"/></g>
"""
    else:
        body = '<rect width="40" height="20" fill="green"/>'
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20">'
        + body
        + "</svg>\n"
    )


def reportlet_path(
    root: Path,
    run: int,
    description: str,
    *,
    subject_label: str = "05",
    run_label: str | None = None,
) -> Path:
    label = run_label if run_label is not None else f"{run:02d}"
    return root / "sub-05" / "figures" / (
        f"sub-{subject_label}_ses-01_task-iaps_run-{label}_"
        f"desc-{description}_bold.svg"
    )


def make_derivative(root: Path, runs: tuple[int, ...] = FEBRUARY_RUNS) -> None:
    for run in runs:
        for description in visual_qc.TARGET_DESCRIPTIONS:
            path = reportlet_path(root, run, description)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                synthetic_svg(layered=description in visual_qc.LAYERED_DESCRIPTIONS),
                encoding="utf-8",
            )


def snapshot(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


class FakeRenderer:
    def __init__(self, derivative: Path) -> None:
        self.derivative = derivative.resolve()
        self.sources: list[Path] = []

    def __call__(self, source: Path, output: Path) -> None:
        source = source.resolve()
        self.sources.append(source)
        if source.name.endswith(("_background.svg", "_foreground.svg")):
            self.assert_static_layer(source)
            if source.is_relative_to(self.derivative):
                raise AssertionError("Layered SVG should be a temporary copy")
        with Image.new("RGBA", (40, 20), (20, 80, 120, 255)) as image:
            image.save(output, format="PNG")

    @staticmethod
    def assert_static_layer(source: Path) -> None:
        text = source.read_text(encoding="utf-8")
        counts = {
            layer: visual_qc._layer_class_count(text, layer)
            for layer in visual_qc.LAYERS
        }
        if counts != {"background": 0, "foreground": 0}:
            raise AssertionError(f"Temporary SVG retained a flicker-layer class: {counts}")


class VisualQcRendererTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("inkscape"), "Inkscape is not installed")
    def test_real_inkscape_backend_selects_synthetic_flicker_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            source = root / "source.svg"
            source.write_text(synthetic_svg(layered=True), encoding="utf-8")
            render = visual_qc._inkscape_render_function(
                Path(shutil.which("inkscape")), timeout_seconds=30
            )
            expected_colors = {
                "background": (0, 0, 128),
                "foreground": (255, 165, 0),
            }
            for layer, expected_color in expected_colors.items():
                static_svg = root / f"{layer}.svg"
                output = root / f"{layer}.png"
                visual_qc.write_static_layer_svg(source, static_svg, layer)
                render(static_svg, output)
                self.assertEqual(visual_qc._validate_png(output), (40, 20))
                with Image.open(output) as image:
                    self.assertEqual(image.convert("RGB").getpixel((20, 10)), expected_color)

    def test_february_matrix_renders_atomically_and_preserves_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            derivative = root / "fmriprep"
            output = root / "reports" / "february_syn"
            make_derivative(derivative)
            before = snapshot(derivative)
            renderer = FakeRenderer(derivative)

            result = visual_qc.run_visual_qc(
                derivative,
                5,
                "01",
                output,
                expected_runs=FEBRUARY_RUNS,
                task="iaps",
                contact_prefix="february_syn",
                columns=2,
                render_svg=renderer,
            )

            self.assertEqual(result, output.resolve())
            self.assertEqual(snapshot(derivative), before)
            self.assertFalse(any(output.parent.glob(".february_syn.staging-*")))
            self.assertEqual(len(renderer.sources), len(FEBRUARY_RUNS) * 6)

            per_run_pngs = sorted(output.glob("run-*.png"))
            contact_pngs = sorted(output.glob("*_contact.png"))
            self.assertEqual(len(per_run_pngs), 24)
            self.assertEqual(len(contact_pngs), 6)
            self.assertEqual(
                {path.name for path in contact_pngs},
                {
                    "february_syn_sdc_background_contact.png",
                    "february_syn_sdc_foreground_contact.png",
                    "february_syn_coreg_background_contact.png",
                    "february_syn_coreg_foreground_contact.png",
                    "february_syn_rois_contact.png",
                    "february_syn_carpetplots_contact.png",
                },
            )
            for path in per_run_pngs + contact_pngs:
                with Image.open(path) as image:
                    image.load()
                    self.assertGreater(image.width, 0)
                    self.assertGreater(image.height, 0)

            with (output / "visual_qc_manifest.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 24)
            self.assertEqual({row["run"] for row in rows}, {"01", "03", "04", "05"})
            self.assertEqual(
                {row["description"] for row in rows}, set(visual_qc.TARGET_DESCRIPTIONS)
            )

            provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["expected_runs"], [1, 3, 4, 5])
            self.assertEqual(provenance["session"], "01")
            self.assertTrue(provenance["source_hashes_rechecked_before_publish"])
            self.assertFalse(provenance["source_derivative_modified"])

    def test_missing_reportlet_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            derivative = root / "fmriprep"
            output = root / "reports" / "february_syn"
            make_derivative(derivative)
            reportlet_path(derivative, 3, "sdc").unlink()

            with self.assertRaisesRegex(RuntimeError, r"sdc: missing=\[3\]"):
                visual_qc.run_visual_qc(
                    derivative,
                    5,
                    "01",
                    output,
                    expected_runs=FEBRUARY_RUNS,
                    render_svg=FakeRenderer(derivative),
                )
            self.assertFalse(output.exists())

    def test_duplicate_run_reportlet_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            derivative = root / "fmriprep"
            make_derivative(derivative)
            original = reportlet_path(derivative, 1, "coreg")
            duplicate = reportlet_path(
                derivative, 1, "coreg", subject_label="5", run_label="1"
            )
            shutil.copyfile(original, duplicate)

            with self.assertRaisesRegex(RuntimeError, r"coreg:.*duplicates=.*1"):
                visual_qc.discover_reportlets(
                    derivative, 5, "01", "iaps", FEBRUARY_RUNS
                )

    def test_unexpected_run_fails_exact_run_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            derivative = root / "fmriprep"
            make_derivative(derivative, runs=FEBRUARY_RUNS + (6,))

            with self.assertRaisesRegex(RuntimeError, r"unexpected=\[6\]"):
                visual_qc.discover_reportlets(
                    derivative, 5, "01", "iaps", FEBRUARY_RUNS
                )

    def test_malformed_flicker_layers_fail_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            derivative = root / "fmriprep"
            output = root / "reports" / "february_syn"
            make_derivative(derivative)
            bad = reportlet_path(derivative, 4, "sdc")
            bad.write_text(synthetic_svg(layered=False), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Expected one background-svg"):
                visual_qc.run_visual_qc(
                    derivative,
                    5,
                    "01",
                    output,
                    expected_runs=FEBRUARY_RUNS,
                    render_svg=FakeRenderer(derivative),
                )
            self.assertFalse(output.exists())
            self.assertFalse(any(output.parent.glob(".february_syn.staging-*")))

    def test_renderer_failure_does_not_publish_partial_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            derivative = root / "fmriprep"
            output = root / "reports" / "february_syn"
            make_derivative(derivative)

            def fail_renderer(source: Path, rendered: Path) -> None:
                raise RuntimeError("synthetic renderer failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic renderer failure"):
                visual_qc.run_visual_qc(
                    derivative,
                    5,
                    "01",
                    output,
                    expected_runs=FEBRUARY_RUNS,
                    render_svg=fail_renderer,
                )
            self.assertFalse(output.exists())
            self.assertFalse(any(output.parent.glob(".february_syn.staging-*")))

    def test_output_inside_derivative_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            derivative = Path(tmp_name) / "fmriprep"
            make_derivative(derivative)
            with self.assertRaisesRegex(ValueError, "outside the fMRIPrep derivative"):
                visual_qc.run_visual_qc(
                    derivative,
                    5,
                    "01",
                    derivative / "qc",
                    expected_runs=FEBRUARY_RUNS,
                    render_svg=FakeRenderer(derivative),
                )


if __name__ == "__main__":
    unittest.main()
