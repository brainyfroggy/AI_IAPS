#!/usr/bin/env python3
"""Run the validated pilot GLMsingle model on one fMRIPrep spatial branch.

The numerical implementation is imported from the completed Subject 4-6
pilot. Only file discovery is generalized beyond MNI res-2.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PILOT_RUNNER = (
    PROJECT_ROOT
    / "Decoding"
    / "pilot_raw_pipeline_sub4_6"
    / "code"
    / "run_glmsingle.py"
)

BRANCH_GLOBS = {
    "bold_acquired_grid": "*_desc-preproc_bold.nii*",
    "subject_t1w": "*_space-T1w_desc-preproc_bold.nii*",
    "mni_res_native": "*_space-MNI152NLin6Asym_desc-preproc_bold.nii*",
    "mni_res_2": "*_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii*",
}

# fMRIPrep 25.1.3 omits the space entity entirely for the acquired/native
# ("func") grid, so the bold_acquired_grid glob above also matches every
# space-* derivative. Exclude those explicitly.
ACQUIRED_GRID_EXCLUDE_TOKEN = "_space-"


def load_pilot() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ai_iaps_pilot_glmsingle", PILOT_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pilot runner: {PILOT_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_number(path: Path) -> int:
    match = re.search(r"_run-(\d+)_", path.name)
    if match is None:
        raise ValueError(f"Cannot parse run from {path}")
    return int(match.group(1))


def discover_branch(root: Path, subject: int, branch: str) -> list[dict[str, Path]]:
    pattern = BRANCH_GLOBS[branch]
    candidates: list[Path] = []
    for subdir in (root / f"sub-{subject:02d}", root / f"sub-{subject}"):
        if subdir.exists():
            candidates.extend(subdir.glob(f"**/func/{pattern}"))
    if branch == "bold_acquired_grid":
        candidates = [path for path in candidates if ACQUIRED_GRID_EXCLUDE_TOKEN not in path.name]

    # A .nii and .nii.gz pair represent the same derivative. Prefer .nii.
    by_run: dict[int, Path] = {}
    for path in sorted(candidates):
        run = run_number(path)
        if run not in by_run or path.suffix == ".nii":
            by_run[run] = path
    if sorted(by_run) != list(range(1, 11)):
        raise RuntimeError(
            f"{branch}: expected one derivative for runs 1-10; found {sorted(by_run)}"
        )

    records: list[dict[str, Path]] = []
    for run in range(1, 11):
        bold = by_run[run]
        base = bold.name.removesuffix(".gz").removesuffix(".nii")
        prefix, spatial = base.split("_desc-preproc_bold", maxsplit=1)[0], ""
        del spatial
        mask_base = prefix + "_desc-brain_mask"
        mask_hits = [
            path
            for path in (bold.parent / f"{mask_base}.nii", bold.parent / f"{mask_base}.nii.gz")
            if path.exists()
        ]
        if len(mask_hits) == 2:
            mask_hits = [path for path in mask_hits if path.suffix == ".nii"]
        pre_space_prefix = prefix.split("_space-", maxsplit=1)[0]
        confounds = bold.parent / f"{pre_space_prefix}_desc-confounds_timeseries.tsv"
        metadata = bold.parent / f"{base}.json"
        if len(mask_hits) != 1 or not confounds.is_file() or not metadata.is_file():
            raise RuntimeError(
                f"Missing companions for {bold}: mask={mask_hits}, "
                f"confounds={confounds}, metadata={metadata}"
            )
        records.append(
            {
                "bold": bold,
                "mask": mask_hits[0],
                "confounds": confounds,
                "json": metadata,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmriprep-root", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--branch", choices=sorted(BRANCH_GLOBS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunklen", type=int, default=10000)
    parser.add_argument("--n-pcs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pilot = load_pilot()

    def replacement(root: Path, subject: int, _space: str):
        return discover_branch(root, subject, args.branch)

    pilot.discover_run_files = replacement
    pilot_args = argparse.Namespace(
        subject=4,
        fmriprep_root=args.fmriprep_root,
        bids_root=args.bids_root,
        space=args.branch,
        output=args.output,
        chunklen=args.chunklen,
        n_pcs=args.n_pcs,
        fixed_frac=None,
        fracs=None,
        max_voxels=None,
        overwrite=False,
    )
    pilot.run(pilot_args)


if __name__ == "__main__":
    main()

