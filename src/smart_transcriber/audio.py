"""Lossless audio preparation with explicit original-recording offsets."""

from __future__ import annotations

import base64
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from smart_transcriber.storage import atomic_json, file_hash
from smart_transcriber.utils import MAX_UPLOAD_BYTES

# Below both interpretations of the documented 25 MB upload limit.
UPLOAD_LIMIT = MAX_UPLOAD_BYTES
SUPPORTED_EXTENSIONS = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}
PREPARATION_VERSION = 1


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    offset: float
    duration: float


def _run(command: list[str]) -> str:
    if not shutil.which(command[0]):
        raise RuntimeError(f"{command[0]} not found on PATH; install ffmpeg (including ffprobe).")
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed: {result.stderr.strip()[-2000:]}")
    return result.stdout


def probe_audio(path: Path) -> dict:
    value = json.loads(_run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,channels,duration",
        "-of", "json", str(path),
    ]))
    streams = [s for s in value.get("streams", []) if s.get("codec_type") == "audio"]
    if len(streams) != 1:
        raise ValueError("Input must contain one audio track; export the intended track first. Multiple channels are supported.")
    duration = float(streams[0].get("duration") or value.get("format", {}).get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Cannot determine a positive audio duration: {path}")
    return {"duration": duration, "channels": streams[0].get("channels"),
            "has_video": any(s.get("codec_type") == "video" for s in value.get("streams", []))}


def prepare_audio(path: Path, directory: Path, chunk_seconds: int, source_hash: str) -> tuple[list[AudioChunk], dict]:
    if chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be positive.")
    info = probe_audio(path)
    if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.stat().st_size <= UPLOAD_LIMIT and not info["has_video"]:
        return [AudioChunk(path, 0.0, info["duration"])], info
    directory.mkdir(parents=True, exist_ok=True)
    settings = {"version": PREPARATION_VERSION, "source_sha256": source_hash, "chunk_seconds": chunk_seconds}
    manifest = directory / "manifest.json"
    try:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["settings"] == settings and saved["chunks"]:
            chunks = []
            for item in saved["chunks"]:
                candidate = (directory / item["file"]).resolve()
                if not candidate.is_relative_to(directory.resolve()) or not 0 < candidate.stat().st_size <= UPLOAD_LIMIT or file_hash(candidate) != item["sha256"]:
                    break
                chunks.append(AudioChunk(candidate, item["offset"], item["duration"]))
            else:
                return chunks, info
    except (OSError, ValueError, KeyError, TypeError):
        pass

    # Preserve channels and sample rate. Try one lossless audio-only upload first.
    whole = directory / "audio.flac"
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0:a:0", "-vn", "-c:a", "flac", str(whole)])
    if whole.stat().st_size <= UPLOAD_LIMIT:
        chunks = [AudioChunk(whole, 0.0, info["duration"])]
    else:
        chunks = []
        offset = 0.0
        while offset < info["duration"] - 0.000001:
            length = min(float(chunk_seconds), info["duration"] - offset)
            target = directory / f"chunk_{len(chunks):05d}.flac"
            while True:
                # Output-side seeking decodes up to the exact original offset.
                _run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ss", f"{offset:.6f}",
                      "-t", f"{length:.6f}", "-map", "0:a:0", "-vn", "-c:a", "flac", str(target)])
                if 0 < target.stat().st_size <= UPLOAD_LIMIT:
                    break
                length /= 2
                if length < 0.1:
                    raise RuntimeError("Could not prepare an audio chunk below the upload limit.")
            chunks.append(AudioChunk(target, offset, length))
            offset += length
    atomic_json(manifest, {"settings": settings, "chunks": [
        {"file": c.path.name, "offset": c.offset, "duration": c.duration, "sha256": file_hash(c.path)} for c in chunks
    ]})
    return chunks, info


def known_speaker_references(items: list[str]) -> list[dict[str, str]]:
    if len(items) > 4:
        raise ValueError("Provide at most four known-speaker references.")
    speakers = []
    names = set()
    for item in items:
        name, separator, filename = item.partition("=")
        name = name.strip()
        path = Path(filename.strip())
        if not separator or not name or not filename.strip() or not path.is_file():
            raise ValueError("Known speaker must be NAME=PATH pointing to an existing audio file.")
        if name in names:
            raise ValueError("Known-speaker names must be distinct.")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or path.stat().st_size > UPLOAD_LIMIT:
            raise ValueError("Known-speaker clips must use a supported audio format below 25 MB.")
        info = probe_audio(path)
        if not 2 <= info["duration"] <= 10 or info["has_video"]:
            raise ValueError("Known-speaker references must be audio-only clips lasting 2 to 10 seconds.")
        # Explicit MIME values avoid Windows' inconsistent extension registry.
        mime = {".mp3": "audio/mpeg", ".mpga": "audio/mpeg", ".mpeg": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".webm": "audio/webm", ".ogg": "audio/ogg", ".flac": "audio/flac", ".wav": "audio/wav"}[path.suffix.lower()]
        speakers.append({"name": name, "sha256": file_hash(path),
                         "data_url": f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"})
        names.add(name)
    return speakers
