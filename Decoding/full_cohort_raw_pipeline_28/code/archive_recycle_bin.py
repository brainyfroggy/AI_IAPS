#!/usr/bin/env python3
"""Archive one exact Windows Recycle Bin SID tree, verify it, then optionally clear it.

Regular files and directories are stored in an uncompressed ZIP64 recovery
archive on another drive. Reparse points are recorded but never followed or
archived. Clearing removes only entries captured by the unchanged pre-archive
snapshot; new files are left alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
DIRECTORY_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_DIRECTORY", 0x10)


@dataclass(frozen=True)
class Entry:
    relative_path: str
    kind: str
    size_bytes: int
    mtime_ns: int
    file_attributes: int


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def safe_source(path: Path) -> Path:
    absolute = path.absolute()
    expected_parent = Path(r"C:\$Recycle.Bin").absolute()
    try:
        relative = absolute.relative_to(expected_parent)
    except ValueError as error:
        raise ValueError(f"Source is outside the exact C: Recycle Bin: {absolute}") from error
    if len(relative.parts) != 1 or not relative.name.startswith("S-1-"):
        raise ValueError(f"Source must be one exact user SID directory: {absolute}")
    if not absolute.is_dir():
        raise FileNotFoundError(absolute)
    # Recycled development trees contain paths longer than the Win32 MAX_PATH
    # limit. The extended-length prefix prevents silent truncation during the
    # recursive snapshot and ZIP read.
    if os.name == "nt" and not str(absolute).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(absolute))
    return absolute


def safe_output(path: Path) -> Path:
    absolute = path.absolute()
    if absolute.drive.casefold() == "c:":
        raise ValueError("Recovery archive must not be written to C:")
    if absolute.exists() or absolute.with_suffix(absolute.suffix + ".partial").exists():
        raise FileExistsError(f"Refusing an existing archive or partial archive: {absolute}")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def snapshot(root: Path) -> list[Entry]:
    records: list[Entry] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name.casefold())
        for child in children:
            child_path = Path(child.path)
            try:
                info = child.stat(follow_symlinks=False)
            except OSError:
                records.append(
                    Entry(
                        relative_path=child_path.relative_to(root).as_posix(),
                        kind="unreadable_entry",
                        size_bytes=0,
                        mtime_ns=0,
                        file_attributes=0,
                    )
                )
                continue
            attributes = int(getattr(info, "st_file_attributes", 0))
            is_reparse = bool(attributes & REPARSE_ATTRIBUTE)
            is_directory = bool(attributes & DIRECTORY_ATTRIBUTE) or child.is_dir(
                follow_symlinks=False
            )
            relative = child_path.relative_to(root).as_posix()
            if is_reparse:
                kind = "reparse_directory" if is_directory else "reparse_file"
            elif is_directory:
                kind = "directory"
            elif child.is_file(follow_symlinks=False):
                kind = "file"
            else:
                kind = "other"
            record_index = len(records)
            record = Entry(
                relative_path=relative,
                kind=kind,
                size_bytes=int(info.st_size),
                mtime_ns=int(info.st_mtime_ns),
                file_attributes=attributes,
            )
            records.append(record)
            if kind == "directory":
                try:
                    visit(child_path)
                except OSError:
                    records[record_index] = Entry(
                        relative_path=record.relative_path,
                        kind="unreadable_directory",
                        size_bytes=record.size_bytes,
                        mtime_ns=record.mtime_ns,
                        file_attributes=record.file_attributes,
                    )

    visit(root)
    return records


def snapshot_digest(records: list[Entry]) -> str:
    payload = json.dumps(
        [asdict(record) for record in records],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_inventory(path: Path, records: list[Entry]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def make_archive(source: Path, partial: Path, records: list[Entry]) -> None:
    archiveable = [record for record in records if record.kind in {"file", "directory"}]
    with zipfile.ZipFile(
        partial,
        mode="x",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
        strict_timestamps=False,
    ) as archive:
        for index, record in enumerate(archiveable, start=1):
            source_path = source.joinpath(*record.relative_path.split("/"))
            archive_name = record.relative_path + ("/" if record.kind == "directory" else "")
            archive.write(source_path, arcname=archive_name)
            if index % 1000 == 0:
                print(f"ARCHIVE {index}/{len(archiveable)}", flush=True)
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())


def verify_archive(partial: Path, records: list[Entry]) -> dict[str, object]:
    expected = {
        record.relative_path + ("/" if record.kind == "directory" else ""): record
        for record in records
        if record.kind in {"file", "directory"}
    }
    with zipfile.ZipFile(partial, mode="r", allowZip64=True) as archive:
        infos = archive.infolist()
        observed = {info.filename: info for info in infos}
        if set(observed) != set(expected):
            raise RuntimeError(
                f"Archive entry mismatch: missing={len(set(expected) - set(observed))}, "
                f"unexpected={len(set(observed) - set(expected))}"
            )
        for name, record in expected.items():
            if record.kind == "file" and observed[name].file_size != record.size_bytes:
                raise RuntimeError(f"Archive size mismatch: {name}")
        print("VERIFY CRC: reading all archived file data", flush=True)
        first_bad = archive.testzip()
        if first_bad is not None:
            raise RuntimeError(f"ZIP CRC verification failed at {first_bad}")
    return {
        "zip_entry_count": len(expected),
        "zip_file_count": sum(record.kind == "file" for record in records),
        "zip_directory_count": sum(record.kind == "directory" for record in records),
        "zip_crc_test": "passed",
    }


def clear_snapshot_entries(source: Path, records: list[Entry]) -> dict[str, object]:
    removed = 0
    failures: list[str] = []
    for record in sorted(
        records,
        key=lambda item: (item.relative_path.count("/"), item.relative_path.casefold()),
        reverse=True,
    ):
        if record.kind not in {
            "file",
            "directory",
            "reparse_file",
            "reparse_directory",
        }:
            continue
        target = source.joinpath(*record.relative_path.split("/"))
        if not os.path.lexists(target):
            continue
        try:
            if record.kind in {"directory", "reparse_directory"}:
                os.rmdir(target)
            else:
                try:
                    os.unlink(target)
                except PermissionError:
                    os.chmod(target, stat.S_IWRITE, follow_symlinks=False)
                    os.unlink(target)
            removed += 1
        except OSError as error:
            failures.append(f"{record.relative_path}: {type(error).__name__}: {error}")
    removable_paths_remaining = [
        record.relative_path
        for record in records
        if record.kind
        in {"file", "directory", "reparse_file", "reparse_directory"}
        if os.path.lexists(source.joinpath(*record.relative_path.split("/")))
    ]
    return {
        "captured_entries_removed": removed,
        "captured_entries_remaining": len(removable_paths_remaining),
        "removal_failures": failures[:100],
        "new_or_uncaptured_entries_left_untouched": len(snapshot(source)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--clear-after-verify", action="store_true")
    args = parser.parse_args()

    started = time.time()
    source = safe_source(args.source)
    final_archive = safe_output(args.archive)
    partial = final_archive.with_suffix(final_archive.suffix + ".partial")
    inventory = final_archive.with_suffix(final_archive.suffix + ".inventory.jsonl")
    summary_path = final_archive.with_suffix(final_archive.suffix + ".summary.json")
    if inventory.exists() or summary_path.exists():
        raise FileExistsError("Refusing existing inventory/summary paths")

    before = snapshot(source)
    if not before:
        raise RuntimeError("Recycle Bin SID directory is empty")
    before_digest = snapshot_digest(before)
    write_inventory(inventory, before)
    make_archive(source, partial, before)
    verification = verify_archive(partial, before)
    after = snapshot(source)
    after_digest = snapshot_digest(after)
    if after_digest != before_digest:
        raise RuntimeError("Recycle Bin contents changed during archive; archive preserved but nothing cleared")

    archive_hash = sha256_file(partial)
    os.replace(partial, final_archive)
    cleanup = {"requested": bool(args.clear_after_verify), "performed": False}
    if args.clear_after_verify:
        cleanup = {
            "requested": True,
            "performed": True,
            **clear_snapshot_entries(source, before),
        }

    summary = {
        "status": "verified",
        "source": str(source),
        "archive": str(final_archive),
        "archive_size_bytes": final_archive.stat().st_size,
        "archive_sha256": archive_hash,
        "inventory": str(inventory),
        "inventory_sha256": sha256_file(inventory),
        "source_snapshot_sha256": before_digest,
        "captured_entry_count": len(before),
        "captured_regular_file_bytes": sum(
            record.size_bytes for record in before if record.kind == "file"
        ),
        "reparse_points_skipped": [
            asdict(record) for record in before if record.kind.startswith("reparse_")
        ],
        "unreadable_entries_left_on_c": [
            asdict(record) for record in before if record.kind.startswith("unreadable_")
        ],
        "verification": verification,
        "cleanup": cleanup,
        "elapsed_seconds": time.time() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if cleanup.get("captured_entries_remaining", 0):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
