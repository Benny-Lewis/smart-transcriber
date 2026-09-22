"""Reusable OpenAI-only transcription pipeline shared by CLI and skill launcher."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from smart_transcriber.audio import PREPARATION_VERSION, prepare_audio
from smart_transcriber.reconcile import VERSION as RECONCILIATION_VERSION, reconcile, render_review
from smart_transcriber.storage import StageStore, atomic_json, atomic_text, file_hash
from smart_transcriber.transcribe import call_transcription, merge_transcripts, transcription_parameters

WORDING_MODEL = "gpt-transcribe"
DIARIZATION_MODEL = "gpt-4o-transcribe-diarize"
ANALYSIS_MODEL = "gpt-6-astra"


@dataclass
class TranscriptionOptions:
    mode: str = "meeting"
    transcribe_model: str = WORDING_MODEL
    diarize_model: str = DIARIZATION_MODEL
    language: str | None = "en"
    prompt: str | None = None
    glossary: list[str] = field(default_factory=list)
    known_speakers: list[dict[str, str]] = field(default_factory=list)
    chunk_seconds: int = 600
    force: bool = False


def read_glossary(path: Path | None) -> list[str]:
    if path is None:
        return []
    # Preserve input order while eliminating duplicates and blank lines.
    terms = list(dict.fromkeys(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()))
    if any("<" in term or ">" in term for term in terms):
        raise ValueError("Glossary entries cannot contain < or >.")
    return terms


def validate_options(options: TranscriptionOptions) -> None:
    if options.mode not in {"meeting", "text", "diarized"}:
        raise ValueError("Mode must be meeting, text, or diarized.")
    if options.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be positive.")
    if options.mode == "text" and options.known_speakers:
        raise ValueError("--known-speaker requires meeting or diarized mode.")
    if options.mode == "diarized" and (options.prompt or options.glossary):
        raise ValueError("Diarized mode cannot use --prompt or --glossary. Choose meeting mode to use both capabilities.")
    if options.mode != "text":
        if options.diarize_model != DIARIZATION_MODEL:
            raise ValueError("No supported diarized response contract is registered for this --diarize-model.")
        transcription_parameters(options.diarize_model, options.language, known_speakers=options.known_speakers)
    if options.mode != "diarized":
        if options.transcribe_model == DIARIZATION_MODEL:
            raise ValueError("Use --mode diarized and --diarize-model for the speaker-labeling model.")
        transcription_parameters(options.transcribe_model, options.language, options.prompt, options.glossary)


def _validate_response(response: Any, diarized: bool, duration: float) -> None:
    if not isinstance(response, dict) or not isinstance(response.get("text"), str):
        raise ValueError("Transcription API returned an invalid text response. The original response has been retained.")
    if diarized:
        segments = response.get("segments")
        if not isinstance(segments, list) or (response["text"].strip() and not segments):
            raise ValueError("Diarization response is missing speaker segments. The original response has been retained.")
        for segment in segments:
            if not isinstance(segment, dict) or not isinstance(segment.get("text"), str) or not isinstance(segment.get("speaker"), str) or not segment["speaker"]:
                raise ValueError("Invalid speaker segment in diarization response.")
            start, end = segment.get("start"), segment.get("end")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)) or not 0 <= start <= end <= duration + 1:
                raise ValueError("Invalid segment timestamps in diarization response.")


def run_transcription(client: Any, audio: Path, directory: Path, options: TranscriptionOptions,
                      progress: Callable[[str], None] = print) -> dict:
    validate_options(options)
    if not audio.is_file() or audio.stat().st_size == 0:
        raise ValueError(f"Audio file is missing or empty: {audio}")
    source_sha = file_hash(audio)
    job = directory / source_sha
    store = StageStore(job)
    chunks, info = prepare_audio(audio, job / "prepared" / str(options.chunk_seconds), options.chunk_seconds, source_sha)
    progress(f"Mode: {options.mode}; {len(chunks)} audio part(s). Completed matching API stages will be reused.")
    if len(chunks) > 1 and options.mode != "text":
        progress("Unknown speaker IDs are scoped to each audio part; names supplied with voice references may span parts.")
    raw_paths: dict[str, list[str]] = {}
    results = {}
    for stage in (["diarization"] if options.mode == "diarized" else ["wording"] if options.mode == "text" else ["diarization", "wording"]):
        is_diarized = stage == "diarization"
        model = options.diarize_model if is_diarized else options.transcribe_model
        references = options.known_speakers if is_diarized else []
        prompt = None if is_diarized else options.prompt
        keywords = [] if is_diarized else options.glossary
        responses = []
        raw_paths[stage] = []
        for index, chunk in enumerate(chunks):
            settings = {
                "request_version": 1, "preparation_version": PREPARATION_VERSION,
                "source_sha256": source_sha, "upload_sha256": file_hash(chunk.path),
                "offset": chunk.offset, "duration": chunk.duration,
                "model": model, "language": options.language, "prompt": prompt, "keywords": keywords,
                "known_speakers": [{"name": s["name"], "sha256": s["sha256"]} for s in references],
            }
            cached = None if options.force else store.load(stage, settings)
            if cached:
                try:
                    _validate_response(cached[0], is_diarized, chunk.duration)
                except ValueError:
                    cached = None
            if cached:
                response, raw = cached
                progress(f"Reusing {stage} part {index + 1}/{len(chunks)} ({model}).")
            else:
                progress(f"Transcribing {stage} part {index + 1}/{len(chunks)} ({model})...")
                response = call_transcription(client, chunk.path, model, options.language, prompt,
                                              keywords=keywords, known_speakers=references)
                # Checkpoint before interpreting the response or attempting any subsequent stage.
                raw = store.save(stage, settings, response)
                _validate_response(response, is_diarized, chunk.duration)
            responses.append(response)
            raw_paths[stage].append(str(raw.resolve()))
        result = merge_transcripts(responses, options.chunk_seconds, offsets=[c.offset for c in chunks],
                                   known_names=[s["name"] for s in references])
        result["duration"] = info["duration"]
        results[stage] = result
        atomic_json(job / f"{stage}.json", result)
    if options.mode == "meeting":
        try:
            transcript = reconcile(results["diarization"], results["wording"])
        except Exception as exc:
            # Paid originals remain intact, even if a new/odd response exposes an alignment bug.
            transcript = dict(results["diarization"])
            transcript["wording_text"] = results["wording"]["text"]
            transcript["reconciliation"] = {"status": "failed", "accepted_count": 0, "review_count": 1,
                "changes": [{"disposition": "review", "reason": f"reconciliation_failed_{type(exc).__name__}",
                             "before": transcript["text"], "after": transcript["wording_text"], "segment_ids": []}]}
    else:
        transcript = results["diarization" if options.mode == "diarized" else "wording"]
    transcript.update(schema_version=1, meta={
        "audio_file": str(audio.resolve()), "source_sha256": source_sha, "duration_seconds": info["duration"],
        "channels": info["channels"], "mode": options.mode, "provider": "openai",
        "transcribe_model": options.diarize_model if options.mode == "diarized" else options.transcribe_model,
        "diarize_model": options.diarize_model if options.mode != "text" else None,
        "analysis_model": None, "language": options.language, "raw_responses": raw_paths,
        "job_directory": str(job.resolve()),
        "chunks": [{"offset": c.offset, "duration": c.duration} for c in chunks],
    })
    if options.mode == "meeting":
        history = StageStore(job / "derived")
        settings = {"version": RECONCILIATION_VERSION, "raw_responses": raw_paths}
        prior = history.load("reconciliation", settings)
        record = prior[1] if prior and prior[0] == transcript else history.save("reconciliation", settings, transcript)
        transcript["meta"]["reconciliation_record"] = str(record.resolve())
    atomic_json(job / "transcript.json", transcript)
    if options.mode == "meeting":
        atomic_text(job / "review.md", render_review(transcript))
    progress(f"Saved reusable transcript: {job / 'transcript.json'}")
    return transcript
