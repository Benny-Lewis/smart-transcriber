"""Audio transcription via OpenAI API — chunking, merging, API calls."""

from __future__ import annotations

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
        speaker = seg.get("speaker") or speaker_map.get(idx) or "Speaker"
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
        same_speaker = last["speaker"] == speaker and speaker != "Speaker"
        last_end = last.get("end")
        gap_ok = False
        if last_end is not None and start is not None:
            gap_ok = 0 <= (start - last_end) <= merge_gap_seconds

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
    if model == "gpt-4o-transcribe-diarize":
        return "diarized_json", None
    if model in {
        "gpt-transcribe", "gpt-4o-transcribe", "gpt-4o-mini-transcribe",
        "gpt-4o-mini-transcribe-2025-12-15",
    }:
        return "json", None
    if model == "whisper-1":
        return "verbose_json", ["segment"]
    raise ValueError(f"Unsupported transcription model: {model}. Check current capabilities before adding a model.")


def transcription_parameters(
    model: str, language: str | None = None, prompt: str | None = None,
    keywords: List[str] | None = None,
    known_speakers: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Validate capabilities before opening/uploading audio. No unsupported hints are dropped."""
    fmt, granularity = select_transcribe_format(model)
    params: Dict[str, Any] = {"model": model, "response_format": fmt}
    extra: Dict[str, Any] = {}
    if model == "gpt-4o-transcribe-diarize":
        if prompt or keywords:
            raise ValueError("The diarization model does not support prompts or glossary terms; use meeting or text mode.")
        params["chunking_strategy"] = "auto"
    elif known_speakers:
        raise ValueError("Known-speaker references require the diarization model.")
    if keywords:
        if model != "gpt-transcribe":
            raise ValueError("Structured glossary terms require gpt-transcribe.")
        if any(not term.strip() or any(c in term for c in "<>\r\n") for term in keywords):
            raise ValueError("Glossary entries must be nonempty single lines without < or >.")
        extra["keywords"] = keywords
    if language and language != "auto":
        if model == "gpt-transcribe":
            extra["languages"] = [language]
        else:
            params["language"] = language
    if prompt:
        params["prompt"] = prompt
    if known_speakers:
        names = [s["name"] for s in known_speakers]
        if len(names) > 4 or len(set(names)) != len(names):
            raise ValueError("Provide at most four distinct known-speaker names.")
        extra["known_speaker_names"] = names
        extra["known_speaker_references"] = [s["data_url"] for s in known_speakers]
    if extra:
        params["extra_body"] = extra
    if granularity:
        params["timestamp_granularities"] = granularity
    return params


def merge_transcripts(
    transcripts: List[Dict[str, Any]],
    chunk_seconds: int,
    *, offsets: List[float] | None = None,
    known_names: List[str] | None = None,
) -> Dict[str, Any]:
    merged_text_parts: List[str] = []
    merged_segments: List[Dict[str, Any]] = []
    if offsets is not None and len(offsets) != len(transcripts):
        raise ValueError("Each transcript needs its original audio offset.")
    for chunk_index, t in enumerate(transcripts):
        offset = offsets[chunk_index] if offsets is not None else float(chunk_index * chunk_seconds)
        text = (t.get("text") or "").strip()
        if text:
            merged_text_parts.append(text)
        segments = t.get("segments") or []
        if segments:
            for seg_index, seg in enumerate(segments):
                start = seg.get("start")
                end = seg.get("end")
                speaker = seg.get("speaker")
                identity = {}
                if speaker:
                    identity = {"speaker": speaker, "speaker_source": "audio"}
                    if len(transcripts) > 1 and speaker not in (known_names or []):
                        identity.update(provider_speaker=speaker, speaker=f"chunk_{chunk_index + 1}:{speaker}")
                merged_segments.append(
                    {
                        **seg,
                        **identity,
                        "id": f"chunk_{chunk_index + 1}:{seg.get('id', seg_index)}" if len(transcripts) > 1 else seg.get("id", str(seg_index)),
                        "start": (start + offset) if start is not None else start,
                        "end": (end + offset) if end is not None else end,
                    }
                )
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
    *, keywords: List[str] | None = None,
    known_speakers: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    params = transcription_parameters(model, language, prompt, keywords, known_speakers)
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(file=audio_file, **params)
        return normalize_response(response)
