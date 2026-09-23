#!/usr/bin/env python3
"""Small non-numerical helpers used by exact production stage contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def windows_to_wsl(path: Path) -> str:
    # Do not call Path.resolve(): mapped N: resolves to its UNC backing path on
    # this workstation, while WSL intentionally consumes it through /mnt/n.
    absolute = Path(os.path.abspath(path))
    drive = absolute.drive.rstrip(":").lower()
    if len(drive) == 1 and drive.isalpha():
        tail = absolute.as_posix().split(":", 1)[1].lstrip("/")
        return f"/mnt/{drive}/{tail}"

    # The audited stage runner resolves its N:-resident root to the backing
    # UNC share before rendering commands.  Docker/WSL must still consume the
    # exact same bytes through /mnt/n, which is the route proven by Sub4-6.
    # Accept only paths beneath this workstation's resolved N: root; arbitrary
    # UNC shares remain fail-closed.
    n_root = Path("N:\\").resolve(strict=True)
    try:
        relative = absolute.resolve(strict=False).relative_to(n_root)
    except ValueError as error:
        raise ValueError(
            f"Only drive-letter paths or the resolved N: share can be mapped to WSL: {absolute}"
        ) from error
    return "/mnt/n/" + relative.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_existing_bids(output: Path, subject: int, config: Path, selection: Path) -> dict:
    """Adopt an atomically built BIDS tree after full byte verification."""

    output = output.resolve(strict=True)
    code = output / "code"
    provenance_path = code / "build_provenance.json"
    generated_path = code / "generated_bids_file_manifest.tsv"
    built_path = code / "built_source_manifest.tsv"
    approved_path = code / "approved_source_selection.tsv"
    for required in (provenance_path, generated_path, built_path, approved_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "config_sha256": sha256_file(config),
        "selection_sha256": sha256_file(selection),
        "subjects": [subject],
        "subject_count": 1,
        "bold_run_count": 10,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"Existing BIDS provenance mismatch at {key}: "
                f"{provenance.get(key)!r} != {value!r}"
            )

    verified: set[str] = set()
    with generated_path.open("r", encoding="utf-8-sig", newline="") as stream:
        generated = list(csv.DictReader(stream, delimiter="\t"))
    for row in generated:
        relative = row["bids_relpath"]
        path = output / Path(relative)
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Generated BIDS file size mismatch: {relative}")
        if sha256_file(path) != row["sha256"].casefold():
            raise ValueError(f"Generated BIDS file hash mismatch: {relative}")
        verified.add(relative.replace("\\", "/"))

    with built_path.open("r", encoding="utf-8-sig", newline="") as stream:
        built = list(csv.DictReader(stream, delimiter="\t"))
    if len(built) != int(provenance.get("asset_count", -1)):
        raise ValueError("Built-source manifest asset count mismatch")
    for row in built:
        relative = row["bids_relpath"]
        path = output / Path(relative)
        if not path.is_file() or sha256_file(path) != row["source_sha256"].casefold():
            raise ValueError(f"BIDS image hash mismatch: {relative}")
        verified.add(relative.replace("\\", "/"))

    # The builder writes these two deterministic metadata files after its
    # generated-file manifest is frozen. Verify their exact bytes explicitly.
    static_files = {
        "participants.tsv": f"participant_id\nsub-{subject:02d}\n".encode("utf-8"),
        "README": (
            "Session-correct AI-IAPS BIDS staging built from an approved exact-path manifest.\n"
            "Do not run fMRIPrep until the official BIDS validator passes and every measured\n"
            "fieldmap association is confirmed in the fMRIPrep report.\n"
        ).encode("utf-8"),
    }
    for relative, expected_bytes in static_files.items():
        path = output / relative
        if not path.is_file() or path.read_bytes() != expected_bytes:
            raise ValueError(f"Static BIDS metadata mismatch: {relative}")
        verified.add(relative)

    actual_data = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and code not in path.parents
    }
    if actual_data != verified:
        raise ValueError(
            f"Existing BIDS data-file set mismatch; missing={sorted(verified - actual_data)}, "
            f"extra={sorted(actual_data - verified)}"
        )
    return {
        "status": "adopted_after_full_verification",
        "subject": subject,
        "files": len(actual_data),
        "bytes": sum((output / Path(item)).stat().st_size for item in actual_data),
        "provenance_sha256": sha256_file(provenance_path),
    }


def bids_build(args: argparse.Namespace) -> int:
    output = Path(os.path.abspath(args.output))
    if output.exists():
        result = verify_existing_bids(output, args.subject, args.config, args.selection)
        print("BIDS_ADOPTION_PASS: " + json.dumps(result, sort_keys=True))
        return 0
    command = [
        str(args.wsl_executable.resolve(strict=True)),
        "-d", args.distribution, "--", args.python_linux_path,
        args.builder_linux_path,
        "--config", args.config_linux_path,
        "--selection", args.selection_linux_path,
        "--output", windows_to_wsl(output),
        "--subject", str(args.subject),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"BIDS builder exited {completed.returncode}")
    print(f"BIDS_BUILD_PASS: {output}")
    return 0


def count_validator_errors(payload: object) -> int:
    if not isinstance(payload, dict):
        raise ValueError("Validator report is not a JSON object")
    issues = payload.get("issues")
    if isinstance(issues, dict):
        issues = issues.get("issues")
    if not isinstance(issues, list):
        raise ValueError("Validator report has no issues list")
    return sum(
        isinstance(issue, dict) and str(issue.get("severity", "")).lower() == "error"
        for issue in issues
    )


def bids_validate(args: argparse.Namespace) -> int:
    bids_root = Path(os.path.abspath(args.bids_root))
    if not bids_root.is_dir():
        raise FileNotFoundError(f"BIDS root does not exist: {bids_root}")
    if not (bids_root / "dataset_description.json").is_file():
        raise ValueError(f"Not a BIDS root: {bids_root}")
    if args.output.exists():
        raise FileExistsError(
            f"Validator output already exists; immutable stage output cannot be replaced: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".json", dir=args.output.parent
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    temp_path.unlink()
    command = [
        str(args.wsl_executable.resolve(strict=True)),
        "-d",
        args.distribution,
        "--",
        args.validator_linux_path,
        "--format",
        "json_pp",
        "--outfile",
        windows_to_wsl(temp_path),
        windows_to_wsl(bids_root),
    ]
    try:
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"BIDS validator exited {completed.returncode}")
        payload = json.loads(temp_path.read_text(encoding="utf-8"))
        error_count = count_validator_errors(payload)
        if error_count:
            raise RuntimeError(f"BIDS validation report contains {error_count} errors")
        os.replace(temp_path, args.output)
        print(f"BIDS_VALIDATION_PASS: {args.output.resolve()}")
        return 0
    finally:
        if temp_path.exists():
            temp_path.unlink()


def blocked(args: argparse.Namespace) -> int:
    print(
        f"BLOCKED_STAGE {args.stage} Sub{args.subject:02d}: {args.reason}",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validator = commands.add_parser("bids-validate")
    validator.add_argument("--wsl-executable", type=Path, required=True)
    validator.add_argument("--distribution", required=True)
    validator.add_argument("--validator-linux-path", required=True)
    validator.add_argument("--bids-root", type=Path, required=True)
    validator.add_argument("--output", type=Path, required=True)
    builder = commands.add_parser("bids-build")
    builder.add_argument("--wsl-executable", type=Path, required=True)
    builder.add_argument("--distribution", required=True)
    builder.add_argument("--python-linux-path", required=True)
    builder.add_argument("--builder-linux-path", required=True)
    builder.add_argument("--config-linux-path", required=True)
    builder.add_argument("--selection-linux-path", required=True)
    builder.add_argument("--config", type=Path, required=True)
    builder.add_argument("--selection", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    builder.add_argument("--subject", type=int, required=True)
    fail = commands.add_parser("blocked")
    fail.add_argument("--stage", required=True)
    fail.add_argument("--subject", type=int, required=True)
    fail.add_argument("--reason", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command == "bids-validate":
            return bids_validate(args)
        if args.command == "bids-build":
            return bids_build(args)
        return blocked(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
