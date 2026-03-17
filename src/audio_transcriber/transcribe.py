"""Audio transcription via OpenAI API — chunking, merging, API calls."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


def merge_display_segments(
    segments: List[Dict[str, Any]],
    speaker_map: Dict[int, str],
    merge_gap_seconds: int,
    max_merge_seconds: int,
    max_merge_words: int,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = speaker_map.get(idx) or "Speaker"
        start = seg.get("start")
        end = seg.get("end")

        if not merged:
            merged.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                }
            )
            continue

        last = merged[-1]
        same_speaker = last["speaker"] == speaker
        last_end = last.get("end")
        gap_ok = False
        if last_end is not None and start is not None:
            gap_ok = (start - last_end) <= merge_gap_seconds

        if same_speaker and gap_ok:
            candidate_text = f"{last['text']} {text}"
            word_ok = True
            if max_merge_words > 0:
                word_ok = len(candidate_text.split()) <= max_merge_words

            duration_ok = True
            if max_merge_seconds > 0:
                start_time = last.get("start")
                candidate_end = end if end is not None else last_end
                if start_time is not None and candidate_end is not None:
                    duration_ok = (candidate_end - start_time) <= max_merge_seconds

            if word_ok and duration_ok:
                last["text"] = candidate_text
                if end is not None:
                    last["end"] = end
            else:
                merged.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                        "text": text,
                    }
                )
        else:
            merged.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                }
            )
    return merged


def normalize_response(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def select_transcribe_format(model: str) -> tuple[str, List[str] | None]:
    if model.startswith("gpt-4o-mini-transcribe") or model.startswith("gpt-4o-transcribe"):
        return "json", None
    return "verbose_json", ["segment"]


def split_audio_ffmpeg(audio_path: Path, chunk_seconds: int, output_dir: Path) -> List[Path]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    ext = audio_path.suffix or ".m4a"
    pattern = output_dir / f"chunk_%03d{ext}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        "-c",
        "copy",
        str(pattern),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chunks = sorted(output_dir.glob(f"chunk_*{ext}"))
    if not chunks:
        raise RuntimeError("ffmpeg produced no chunks")
    return chunks


def merge_transcripts(
    transcripts: List[Dict[str, Any]],
    chunk_seconds: int,
) -> Dict[str, Any]:
    merged_text_parts: List[str] = []
    merged_segments: List[Dict[str, Any]] = []
    offset = 0.0
    for t in transcripts:
        text = (t.get("text") or "").strip()
        if text:
            merged_text_parts.append(text)
        segments = t.get("segments") or []
        if segments:
            for seg in segments:
                start = seg.get("start")
                end = seg.get("end")
                merged_segments.append(
                    {
                        **seg,
                        "start": (start + offset) if start is not None else start,
                        "end": (end + offset) if end is not None else end,
                    }
                )
            last_end = segments[-1].get("end")
            offset += last_end if last_end is not None else float(chunk_seconds)
        else:
            offset += float(chunk_seconds)
    merged: Dict[str, Any] = {"text": "\n".join(merged_text_parts)}
    if merged_segments:
        merged["segments"] = merged_segments
    return merged


def call_transcription(
    client: OpenAI,
    audio_path: Path,
    model: str,
    language: str | None,
    prompt: str | None,
) -> Dict[str, Any]:
    response_format, timestamp_granularities = select_transcribe_format(model)
    params: Dict[str, Any] = {
        "model": model,
        "file": audio_path.open("rb"),
        "response_format": response_format,
    }
    if timestamp_granularities:
        params["timestamp_granularities"] = timestamp_granularities
    if language:
        params["language"] = language
    if prompt:
        params["prompt"] = prompt
    try:
        response = client.audio.transcriptions.create(**params)
        return normalize_response(response)
    finally:
        params["file"].close()
