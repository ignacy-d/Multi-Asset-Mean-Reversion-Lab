"""Freeze an explicit preregistration by binding its exact bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

LOCK_SCHEMA_VERSION = "mr-lab-study-lock-v1"
STUDY_ID_PATTERN = re.compile(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\Z")
STATUS_FIELD_PATTERN = re.compile(
    rb"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?status\s*:(?:\*\*)?\s*"
    rb"[\"'`*_]*([A-Za-z]+)"
)
STUDY_ID_FIELD_PATTERN = re.compile(
    rb"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?study_id\s*:(?:\*\*)?\s*"
    rb"[\"'`*_]*([A-Za-z0-9_-]+)"
)


class SpecLockError(ValueError):
    """Raised when a preregistration cannot be safely frozen."""


def validate_study_id(study_id: str) -> str:
    """Return a canonical study ID or reject unsafe/path-like values."""
    if not STUDY_ID_PATTERN.fullmatch(study_id):
        raise SpecLockError(
            "study_id must use uppercase letters, digits, and single hyphens"
        )
    return study_id


def hash_spec_bytes(spec_bytes: bytes) -> str:
    """Hash exact preregistration bytes with SHA-256."""
    return hashlib.sha256(spec_bytes).hexdigest()


def _reject_draft_status(spec_bytes: bytes) -> None:
    statuses = (
        match.group(1).upper() for match in STATUS_FIELD_PATTERN.finditer(spec_bytes)
    )
    if b"DRAFT" in statuses:
        raise SpecLockError("refusing to freeze a DRAFT preregistration")


def _validate_declared_study_id(study_id: str, spec_bytes: bytes) -> None:
    declared = {
        match.group(1).decode("ascii")
        for match in STUDY_ID_FIELD_PATTERN.finditer(spec_bytes)
    }
    if declared and declared != {study_id}:
        raise SpecLockError(
            "study_id does not match every study_id declared by the preregistration"
        )


def build_spec_lock(study_id: str, spec_path: Path) -> dict[str, str]:
    """Build lock content from one explicitly supplied preregistration path."""
    validate_study_id(study_id)
    try:
        descriptor = os.open(spec_path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as error:
        raise SpecLockError(f"missing or unreadable spec: {spec_path}") from error
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise SpecLockError(f"spec is not a regular file: {spec_path}")
        spec_bytes = stream.read()
    _reject_draft_status(spec_bytes)
    _validate_declared_study_id(study_id, spec_bytes)
    return {
        "lock_schema_version": LOCK_SCHEMA_VERSION,
        "spec_path": Path(os.path.normpath(spec_path)).as_posix(),
        "spec_sha256": hash_spec_bytes(spec_bytes),
        "study_id": study_id,
    }


def freeze_study(study_id: str, spec_path: Path, output_path: Path) -> Path:
    """Create a new lock exclusively and refuse every existing output path."""
    lock = build_spec_lock(study_id, spec_path)
    payload = json.dumps(lock, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        descriptor = os.open(
            output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise SpecLockError(
            f"refusing to overwrite existing lock: {output_path}"
        ) from error
    except OSError as error:
        raise SpecLockError(f"cannot create lock: {output_path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except BaseException:
        with suppress(OSError):
            output_path.unlink()
        raise
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bind an empirical study to exact preregistration bytes"
    )
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = freeze_study(args.study_id, args.spec, args.output)
    except SpecLockError as error:
        parser.error(str(error))
    print(f"study_lock={output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
