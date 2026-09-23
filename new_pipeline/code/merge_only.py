#!/usr/bin/env python3
"""Re-run just the merge step for a mixed-SDC-mode subject whose branches
already completed individually, without re-running fMRIPrep.

Exists because fmriprep_sdc_workflow_v6.run_subject() has no "branch output
already exists, skip straight to merge" path -- it unconditionally
FileExistsErrors if a branch's output directory is already there (the only
alternative is the narrow recovery-attestation mechanism, built for a
different, specific historical failure and not applicable here). This script
reuses wave_orchestrator.py's own module-loading and monkeypatch machinery
(load_v6_module, patch_merge_mixed_branches) and calls
v6.merge_mixed_branches() directly against the two already-completed branch
directories, so none of that compute is wasted or repeated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wave_orchestrator as wo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--cohort-config", type=Path, default=wo.DEFAULT_COHORT_CONFIG)
    parser.add_argument("--sdc-config", type=Path, default=wo.DEFAULT_SDC_CONFIG)
    parser.add_argument("--derivatives-root", type=Path, default=wo.DEFAULT_DERIVATIVES_ROOT)
    parser.add_argument("--link-mode", choices=("hardlink", "copy", "auto"), default="auto")
    args = parser.parse_args()

    v6 = wo.load_v6_module()
    cohort, sdc = v6.load_configs(args.cohort_config, args.sdc_config)
    wo.patch_basic_derivative_gate(v6, sdc)
    wo.patch_merge_mixed_branches(v6)

    plan = v6.subject_plan(cohort, args.subject)
    if plan.strategy != "isolated_mixed_session_branches":
        raise ValueError(f"Subject {args.subject}'s strategy is {plan.strategy!r}, not a mixed-branch merge case")

    paths = wo.subject_paths(args.subject, args.derivatives_root)
    branch_output_root = paths["branch_output_root"]
    branch_outputs = {}
    for branch in plan.branches:
        candidate = branch_output_root / "branches" / branch.name
        if not candidate.is_dir():
            raise FileNotFoundError(f"Branch output missing, cannot merge-only: {candidate}")
        branch_outputs[branch.name] = candidate
        print(f"Reusing already-completed branch output for {branch.name}: {candidate}")

    final_root = paths["final_root"]
    if final_root.exists():
        raise FileExistsError(f"final_root already exists, refusing to overwrite: {final_root}")

    print(f"Merging {len(branch_outputs)} branches into {final_root} ...")
    result = v6.merge_mixed_branches(plan, branch_outputs, final_root, args.link_mode)
    print("MERGE_PASS")
    import json

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
