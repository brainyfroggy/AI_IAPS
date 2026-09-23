#!/usr/bin/env python3
"""Render a fail-closed, read-only visual-QC bundle from fMRIPrep reportlets.

The utility renders the two static layers of fMRIPrep's SDC and BOLD-to-T1w
coregistration flicker reportlets, plus the ROI and carpet-plot reportlets.  It
never edits a derivative: layer selection is injected into a temporary SVG,
and source hashes are checked again before the completed output is published.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image, ImageDraw, ImageOps


TARGET_DESCRIPTIONS = ("sdc", "coreg", "rois", "carpetplot")
LAYERED_DESCRIPTIONS = ("sdc", "coreg")
LAYERS = ("background", "foreground")
REPORTLET_PATTERN = re.compile(
    r"^sub-(?P<subject>\d+)_ses-(?P<session>[A-Za-z0-9]+)_"
    r"task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>\d+)_"
    r"desc-(?P<description>sdc|coreg|rois|carpetplot)_bold\.svg$"
)


@dataclass(frozen=True)
class Reportlet:
    run: int
    description: str
    path: Path
    sha256: str


RenderFunction = Callable[[Path, Path], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_runs(expected_runs: Sequence[int]) -> tuple[int, ...]:
    runs = tuple(int(run) for run in expected_runs)
    if not runs or len(set(runs)) != len(runs) or any(run < 1 for run in runs):
        raise ValueError(f"Expected runs must be unique positive integers; received {runs}")
    return tuple(sorted(runs))


def _subject_directory(root: Path, subject: int) -> Path:
    candidates = [root / f"sub-{subject:02d}", root / f"sub-{subject}"]
    hits = list(dict.fromkeys(path.resolve() for path in candidates if path.is_dir()))
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one subject directory for Sub{subject} under {root}; found {hits}"
        )
    return hits[0]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def discover_reportlets(
    fmriprep_root: Path,
    subject: int,
    session: str,
    task: str,
    expected_runs: Sequence[int],
) -> dict[int, dict[str, Reportlet]]:
    """Find exactly the expected run-by-description reportlet matrix."""
    root = fmriprep_root.resolve()
    runs = _normalized_runs(expected_runs)
    session = str(session)
    task = str(task)
    subject_dir = _subject_directory(root, subject)
    figures = subject_dir / "figures"
    if not figures.is_dir():
        raise FileNotFoundError(f"Missing fMRIPrep figures directory: {figures}")

    hits: dict[str, dict[int, list[Path]]] = {
        description: {} for description in TARGET_DESCRIPTIONS
    }
    for path in sorted(figures.glob("*.svg")):
        match = REPORTLET_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        if int(match.group("subject")) != int(subject):
            continue
        if match.group("session") != session or match.group("task") != task:
            continue
        description = match.group("description")
        run = int(match.group("run"))
        hits[description].setdefault(run, []).append(path.resolve())

    expected_set = set(runs)
    problems: list[str] = []
    for description in TARGET_DESCRIPTIONS:
        observed_set = set(hits[description])
        missing = sorted(expected_set - observed_set)
        unexpected = sorted(observed_set - expected_set)
        duplicates = {
            run: paths for run, paths in hits[description].items() if len(paths) != 1
        }
        if missing or unexpected or duplicates:
            problems.append(
                f"{description}: missing={missing}, unexpected={unexpected}, "
                f"duplicates={duplicates}"
            )
    if problems:
        raise RuntimeError(
            "Reportlet matrix is not exact for "
            f"Sub{subject} ses-{session} task-{task}, expected runs {list(runs)}; "
            + "; ".join(problems)
        )

    matrix: dict[int, dict[str, Reportlet]] = {}
    for run in runs:
        matrix[run] = {}
        for description in TARGET_DESCRIPTIONS:
            path = hits[description][run][0]
            matrix[run][description] = Reportlet(
                run=run,
                description=description,
                path=path,
                sha256=_sha256(path),
            )
    return matrix


def _layer_class_count(svg: str, layer: str) -> int:
    return len(
        re.findall(
            rf"class\s*=\s*['\"][^'\"]*\b{re.escape(layer)}-svg\b[^'\"]*['\"]",
            svg,
            flags=re.IGNORECASE,
        )
    )


def write_static_layer_svg(source: Path, output: Path, layer: str) -> None:
    """Write a temporary SVG with exactly one flicker layer made visible."""
    if layer not in LAYERS:
        raise ValueError(f"Unknown reportlet layer {layer!r}; expected one of {LAYERS}")
    try:
        tree = ET.parse(source)
    except ET.ParseError as exc:
        raise RuntimeError(f"Cannot parse SVG reportlet {source}: {exc}") from exc
    root = tree.getroot()
    matches = {
        name: [
            element
            for element in root.iter()
            if f"{name}-svg" in element.attrib.get("class", "").split()
        ]
        for name in LAYERS
    }
    counts = {name: len(elements) for name, elements in matches.items()}
    if counts != {"background": 1, "foreground": 1}:
        raise RuntimeError(
            f"Expected one background-svg and one foreground-svg in {source}; found {counts}"
        )
    other = "foreground" if layer == "background" else "background"
    parents = {child: parent for parent in root.iter() for child in parent}
    unwanted = matches[other][0]
    if unwanted not in parents:
        raise RuntimeError(f"Cannot remove the {other} SVG layer from {source}")
    parents[unwanted].remove(unwanted)

    # Removing the retained class also disables fMRIPrep's hover animation.
    retained = matches[layer][0]
    remaining_classes = [
        name for name in retained.attrib.get("class", "").split() if name != f"{layer}-svg"
    ]
    if remaining_classes:
        retained.set("class", " ".join(remaining_classes))
    else:
        retained.attrib.pop("class", None)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def _resolve_inkscape(renderer: str | Path | None) -> Path:
    requested = str(renderer) if renderer is not None else "inkscape"
    found = shutil.which(requested)
    if found is None:
        raise FileNotFoundError(
            f"Cannot find SVG renderer {requested!r}; install Inkscape or pass --inkscape"
        )
    return Path(found).resolve()


def _inkscape_version(renderer: Path) -> str:
    completed = subprocess.run(
        [str(renderer), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (completed.stdout or completed.stderr).strip()


def _inkscape_render_function(renderer: Path, timeout_seconds: int) -> RenderFunction:
    def render(source: Path, output: Path) -> None:
        completed = subprocess.run(
            [
                str(renderer),
                str(source),
                "--export-type=png",
                f"--export-filename={output}",
                "--export-background=white",
                "--export-background-opacity=255",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Inkscape failed for {source} with exit code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )

    return render


def _validate_png(path: Path) -> tuple[int, int]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Renderer did not create a non-empty PNG: {path}")
    with Image.open(path) as image:
        image.load()
        size = image.size
    if size[0] < 1 or size[1] < 1:
        raise RuntimeError(f"Invalid rendered PNG dimensions for {path}: {size}")
    return size


def _make_contact_sheet(
    image_paths: Sequence[Path],
    run_labels: Sequence[str],
    output: Path,
    columns: int,
) -> None:
    if not image_paths or len(image_paths) != len(run_labels):
        raise ValueError("Contact-sheet images and labels must be non-empty and have equal length")
    if columns < 1:
        raise ValueError(f"Contact-sheet columns must be positive; received {columns}")

    loaded: list[Image.Image] = []
    try:
        for path in image_paths:
            with Image.open(path) as image:
                loaded.append(image.convert("RGBA"))
        tile_width = max(image.width for image in loaded)
        tile_height = max(image.height for image in loaded)
        label_height = 34
        margin = 16
        rows = math.ceil(len(loaded) / columns)
        sheet = Image.new(
            "RGBA",
            (
                margin + columns * (tile_width + margin),
                margin + rows * (label_height + tile_height + margin),
            ),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for index, (image, label) in enumerate(zip(loaded, run_labels)):
            row, column = divmod(index, columns)
            left = margin + column * (tile_width + margin)
            top = margin + row * (label_height + tile_height + margin)
            draw.text((left + 4, top + 8), label, fill="black")
            fitted = ImageOps.contain(
                image,
                (tile_width, tile_height),
                method=Image.Resampling.LANCZOS,
            )
            image_left = left + (tile_width - fitted.width) // 2
            image_top = top + label_height + (tile_height - fitted.height) // 2
            sheet.alpha_composite(fitted, (image_left, image_top))
        sheet.save(output, format="PNG", optimize=True)
        sheet.close()
    finally:
        for image in loaded:
            image.close()
    _validate_png(output)


def _callable_label(render_svg: RenderFunction) -> str:
    module = getattr(render_svg, "__module__", "unknown")
    name = getattr(render_svg, "__qualname__", getattr(render_svg, "__name__", "callable"))
    return f"python-callable:{module}.{name}"


def run_visual_qc(
    fmriprep_root: Path,
    subject: int,
    session: str,
    output: Path,
    *,
    expected_runs: Sequence[int],
    task: str = "iaps",
    contact_prefix: str | None = None,
    columns: int | None = None,
    renderer: str | Path | None = None,
    renderer_timeout_seconds: int = 120,
    render_svg: RenderFunction | None = None,
) -> Path:
    """Render and atomically publish a complete visual-QC bundle."""
    root = fmriprep_root.resolve()
    destination = output.resolve()
    runs = _normalized_runs(expected_runs)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    if _is_relative_to(destination, root):
        raise ValueError(
            f"Output must be outside the fMRIPrep derivative to preserve read-only inputs: {destination}"
        )
    if columns is None:
        columns = math.ceil(math.sqrt(len(runs)))
    if columns < 1:
        raise ValueError(f"Contact-sheet columns must be positive; received {columns}")
    prefix = contact_prefix or f"ses-{session}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError(f"Unsafe contact-sheet prefix: {prefix!r}")

    # Discover and validate the complete matrix before creating any output.
    matrix = discover_reportlets(root, subject, session, task, runs)

    if render_svg is None:
        renderer_path = _resolve_inkscape(renderer)
        renderer_label = _inkscape_version(renderer_path)
        render_function = _inkscape_render_function(
            renderer_path, int(renderer_timeout_seconds)
        )
    else:
        renderer_label = _callable_label(render_svg)
        render_function = render_svg

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    manifest_rows: list[dict[str, object]] = []
    rendered: dict[str, list[Path]] = {
        "sdc_background": [],
        "sdc_foreground": [],
        "coreg_background": [],
        "coreg_foreground": [],
        "rois": [],
        "carpetplots": [],
    }
    contact_paths: dict[str, Path] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="fmriprep-reportlet-layers-") as temp_name:
            temp = Path(temp_name)
            for run in runs:
                for description in TARGET_DESCRIPTIONS:
                    reportlet = matrix[run][description]
                    layers = LAYERS if description in LAYERED_DESCRIPTIONS else ("static",)
                    for layer in layers:
                        suffix = f"_{layer}" if layer != "static" else ""
                        output_png = staging / f"run-{run:02d}_{description}{suffix}.png"
                        if layer in LAYERS:
                            render_source = temp / (
                                f"run-{run:02d}_{description}_{layer}.svg"
                            )
                            write_static_layer_svg(reportlet.path, render_source, layer)
                        else:
                            render_source = reportlet.path
                        render_function(render_source, output_png)
                        width, height = _validate_png(output_png)
                        category = (
                            f"{description}_{layer}"
                            if layer in LAYERS
                            else ("carpetplots" if description == "carpetplot" else description)
                        )
                        rendered[category].append(output_png)
                        manifest_rows.append(
                            {
                                "run": f"{run:02d}",
                                "session": session,
                                "task": task,
                                "description": description,
                                "layer": layer,
                                "source_svg": str(reportlet.path),
                                "source_sha256": reportlet.sha256,
                                "output_png": output_png.name,
                                "width_pixels": width,
                                "height_pixels": height,
                            }
                        )

        labels = [f"run-{run:02d}" for run in runs]
        for category, image_paths in rendered.items():
            if len(image_paths) != len(runs):
                raise RuntimeError(
                    f"Internal completeness error for {category}: "
                    f"expected {len(runs)} images, found {len(image_paths)}"
                )
            contact = staging / f"{prefix}_{category}_contact.png"
            _make_contact_sheet(image_paths, labels, contact, columns)
            contact_paths[category] = contact

        # Re-hash every source after rendering, detecting any accidental or
        # concurrent input change before publication.
        changed = []
        for run in runs:
            for description in TARGET_DESCRIPTIONS:
                reportlet = matrix[run][description]
                observed_hash = _sha256(reportlet.path)
                if observed_hash != reportlet.sha256:
                    changed.append(
                        {
                            "source": str(reportlet.path),
                            "before": reportlet.sha256,
                            "after": observed_hash,
                        }
                    )
        if changed:
            raise RuntimeError(f"Source reportlets changed during rendering: {changed}")

        manifest_path = staging / "visual_qc_manifest.tsv"
        with manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(manifest_rows[0]),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        provenance = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_fmriprep_root": str(root),
            "subject": int(subject),
            "session": session,
            "task": task,
            "expected_runs": list(runs),
            "expected_descriptions": list(TARGET_DESCRIPTIONS),
            "layered_descriptions": list(LAYERED_DESCRIPTIONS),
            "contact_prefix": prefix,
            "contact_columns": columns,
            "renderer": renderer_label,
            "source_reportlets": [
                {
                    "run": run,
                    "description": description,
                    "path": str(matrix[run][description].path),
                    "sha256": matrix[run][description].sha256,
                }
                for run in runs
                for description in TARGET_DESCRIPTIONS
            ],
            "contact_sheets": {
                category: path.name for category, path in contact_paths.items()
            },
            "source_hashes_rechecked_before_publish": True,
            "source_derivative_modified": False,
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )

        expected_png_count = len(runs) * 6 + len(rendered)
        pngs = sorted(staging.glob("*.png"))
        if len(pngs) != expected_png_count:
            raise RuntimeError(
                f"Expected {expected_png_count} PNGs before publication; found {len(pngs)}"
            )
        for path in pngs:
            _validate_png(path)

        staging.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmriprep-root", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--task", default="iaps")
    parser.add_argument("--expected-runs", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-prefix")
    parser.add_argument("--columns", type=int)
    parser.add_argument("--inkscape", help="Inkscape executable (default: PATH lookup)")
    parser.add_argument("--renderer-timeout-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = run_visual_qc(
        args.fmriprep_root,
        args.subject,
        args.session,
        args.output,
        expected_runs=args.expected_runs,
        task=args.task,
        contact_prefix=args.contact_prefix,
        columns=args.columns,
        renderer=args.inkscape,
        renderer_timeout_seconds=args.renderer_timeout_seconds,
    )
    print(f"Visual-QC bundle written to {output}")


if __name__ == "__main__":
    main()
