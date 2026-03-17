"""Pure formatting and normalization utilities."""

from __future__ import annotations

from typing import Any


MAX_UPLOAD_BYTES = 26_214_400
DEFAULT_CHUNK_SECONDS = 600
DEFAULT_AUDIO_EXTS = [
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".webm",
    ".mp4",
    ".mov",
    ".mkv",
    ".aiff",
    ".aif",
    ".caf",
    ".wma",
]


def format_timestamp(seconds: float | None, precision: str = "ms") -> str:
    if seconds is None:
        return "00:00:00" if precision == "s" else "00:00:00.000"
    if precision == "s":
        total = int(round(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def normalize_text_ascii(text: str) -> str:
    replacements = {
        "â€“": "-",
        "â€”": "--",
        "â€˜": "'",
        "â€™": "'",
        "â€œ": "\"",
        "â€�": "\"",
        "â€¦": "...",
        "Â": "",
    }
    for src, dest in replacements.items():
        text = text.replace(src, dest)
    return text


def parse_time_of_day(value: str) -> int:
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("time must be HH:MM or HH:MM:SS")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2]) if len(parts) == 3 else 0
    if not (0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds <= 59):
        raise ValueError("time must be a valid 24h clock time")
    return hours * 3600 + minutes * 60 + seconds


def format_time_of_day(offset_seconds: float | None, start_seconds: int) -> str:
    if offset_seconds is None:
        return "--:--:--"
    total = int(round(offset_seconds)) + start_seconds
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{hours:02d}:{minutes:02d}:{secs:02d} (+{days}d)"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_annotation_timestamp(value: Any, start_time_seconds: int | None) -> str:
    if isinstance(value, (int, float)):
        seconds = float(value)
        if start_time_seconds is not None:
            return format_time_of_day(seconds, start_time_seconds)
        return format_timestamp(seconds, precision="s")
    if isinstance(value, str):
        try:
            seconds = float(value)
            if start_time_seconds is not None:
                return format_time_of_day(seconds, start_time_seconds)
            return format_timestamp(seconds, precision="s")
        except ValueError:
            return normalize_text_ascii(value)
    return normalize_text_ascii(str(value))
