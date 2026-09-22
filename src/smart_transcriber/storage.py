"""Atomic local artifacts and independent caches for paid API stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class StageStore:
    """Raw responses are immutable; a small index selects the last completed attempt."""

    def __init__(self, directory: Path):
        self.directory = directory

    def _index(self, stage: str, settings: dict) -> Path:
        return self.directory / "cache" / f"{stage}-{fingerprint(settings)}.json"

    def load(self, stage: str, settings: dict) -> tuple[dict, Path] | None:
        try:
            index = self._index(stage, settings)
            if not index.is_file():
                # A process can stop after the raw response commits but before its
                # index does. The filename binds that response to exact settings.
                candidates = sorted((self.directory / "raw").glob(f"{stage}-{fingerprint(settings)}-*.json"),
                                    key=lambda p: p.stat().st_mtime_ns, reverse=True)
                for raw in candidates:
                    try:
                        value = json.loads(raw.read_text(encoding="utf-8"))
                        if isinstance(value, dict):
                            return value, raw.resolve()
                    except (OSError, ValueError):
                        continue
                return None
            record = json.loads(index.read_text(encoding="utf-8"))
            if record["settings"] != settings:
                return None
            raw = (self.directory / record["raw"]).resolve()
            if not raw.is_relative_to((self.directory / "raw").resolve()):
                return None
            if file_hash(raw) != record["sha256"]:
                return None
            value = json.loads(raw.read_text(encoding="utf-8"))
            return (value, raw) if isinstance(value, dict) else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, stage: str, settings: dict, response: dict) -> Path:
        raw = self.directory / "raw" / f"{stage}-{fingerprint(settings)}-{uuid.uuid4().hex}.json"
        atomic_json(raw, response)
        atomic_json(self._index(stage, settings), {
            "settings": settings, "raw": raw.relative_to(self.directory).as_posix(),
            "sha256": file_hash(raw),
        })
        return raw
