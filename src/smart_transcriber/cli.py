"""CLI entry point and argument parsing."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from smart_transcriber import __version__
from smart_transcriber.analyze import analyze_transcript
from smart_transcriber.render import render_markdown, render_outline_markdown
from smart_transcriber.transcribe import call_transcription, merge_transcripts, split_audio_ffmpeg
from smart_transcriber.utils import (
    DEFAULT_AUDIO_EXTS,
    DEFAULT_CHUNK_SECONDS,
    MAX_UPLOAD_BYTES,
    parse_time_of_day,
)


def resolve_path_with_extensions(path: Path, exts: List[str]) -> Path | None:
    if path.exists():
        return path
    if path.suffix:
        return None
    for ext in exts:
        candidate = path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a meeting and produce a summary with speaker annotations."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument("--out", help="Markdown output path.")
    parser.add_argument("--json-out", help="JSON output path for analysis + transcript.")
    parser.add_argument(
        "--transcript-json",
        help="Optional path to save raw transcription JSON from the API.",
    )
    parser.add_argument(
        "--transcribe-model",
        default="whisper-1",
        help=(
            "Transcription model (e.g., whisper-1, gpt-4o-mini-transcribe). "
            "Whisper supports segment timestamps via verbose_json."
        ),
    )
    parser.add_argument(
        "--analysis-model",
        default="gpt-5.2",
        help="Model to summarize and annotate the transcript.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language hint (default: en).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Optional hint for number of speakers in the meeting.",
    )
    parser.add_argument(
        "--prompt",
        help="Optional transcription prompt (keywords, names, jargon).",
    )
    parser.add_argument(
        "--style",
        choices=["outline", "report"],
        default="report",
        help="Output style (default: report).",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata section in outline style.",
    )
    parser.add_argument(
        "--disclaimer",
        help="Optional disclaimer text to include at the top of the notes.",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip summary/annotation step and only transcribe.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip transcription and only run analysis on a JSON transcript.",
    )
    parser.add_argument(
        "--transcript-input",
        help=(
            "Path to a JSON file containing a transcript (raw or json-out format). "
            "Required with --analysis-only."
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Chunk length in seconds if the file is too large (default: {DEFAULT_CHUNK_SECONDS}).",
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=int,
        default=2,
        help="Merge adjacent segments from the same speaker if the gap is <= this many seconds.",
    )
    parser.add_argument(
        "--max-merge-seconds",
        type=int,
        default=45,
        help="Max total seconds per merged transcript line (default: 45).",
    )
    parser.add_argument(
        "--max-merge-words",
        type=int,
        default=80,
        help="Max total words per merged transcript line (default: 80).",
    )
    parser.add_argument(
        "--start-time",
        help="Optional start time (HH:MM or HH:MM:SS) to render time-of-day instead of offsets.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    if args.analysis_only and args.no_analysis:
        print("--analysis-only cannot be combined with --no-analysis.", file=sys.stderr)
        return 1

    if args.analysis_only and not args.transcript_input:
        print("--analysis-only requires --transcript-input.", file=sys.stderr)
        return 1

    start_time_seconds: int | None = None
    if args.start_time:
        try:
            start_time_seconds = parse_time_of_day(args.start_time)
        except ValueError as exc:
            print(f"Invalid --start-time: {exc}", file=sys.stderr)
            return 1

    audio_path: Path | None = None
    if not args.analysis_only:
        if not args.audio_file:
            print("audio_file is required unless --analysis-only is set.", file=sys.stderr)
            return 1
        audio_path = Path(args.audio_file)
        resolved_audio = resolve_path_with_extensions(audio_path, DEFAULT_AUDIO_EXTS)
        if not resolved_audio:
            print(f"Audio file not found: {audio_path}", file=sys.stderr)
            return 1
        if resolved_audio != audio_path:
            print(f"Resolved audio file: {resolved_audio}")
        audio_path = resolved_audio

    if args.out:
        out_path = Path(args.out)
    elif audio_path:
        out_path = Path.cwd() / (audio_path.stem + ".md")
    else:
        out_path = Path.cwd() / "notes.md"
    json_out_path = Path(args.json_out) if args.json_out else None
    raw_transcript_path = Path(args.transcript_json) if args.transcript_json else None

    client = OpenAI()

    transcript: Dict[str, Any]
    transcript_text = ""
    segments: List[Dict[str, Any]] = []
    duration_seconds = None
    audio_file_label = str(audio_path) if audio_path else "transcript.json"

    if args.analysis_only:
        transcript_path = Path(args.transcript_input)
        resolved_transcript = resolve_path_with_extensions(transcript_path, [".json"])
        if not resolved_transcript:
            print(f"Transcript file not found: {transcript_path}", file=sys.stderr)
            return 1
        if resolved_transcript != transcript_path:
            print(f"Resolved transcript file: {resolved_transcript}")
        transcript_path = resolved_transcript
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        if "transcript" in transcript and isinstance(transcript["transcript"], dict):
            transcript_text = transcript["transcript"].get("text", "") or ""
            segments = transcript["transcript"].get("segments", []) or []
        else:
            transcript_text = transcript.get("text", "") or ""
            segments = transcript.get("segments", []) or []

        meta_in = transcript.get("meta") or {}
        if meta_in.get("audio_file"):
            audio_file_label = str(meta_in["audio_file"])
        if meta_in.get("duration_seconds") is not None:
            duration_seconds = meta_in.get("duration_seconds")
    else:
        if audio_path.stat().st_size > MAX_UPLOAD_BYTES:
            print(
                f"Audio file is larger than {MAX_UPLOAD_BYTES} bytes; chunking with ffmpeg..."
            )
            try:
                with tempfile.TemporaryDirectory(prefix="audio_chunks_") as tmpdir:
                    chunks = split_audio_ffmpeg(audio_path, args.chunk_seconds, Path(tmpdir))
                    transcripts: List[Dict[str, Any]] = []
                    total = len(chunks)
                    for idx, chunk in enumerate(chunks, start=1):
                        print(f"Transcribing chunk {idx}/{total}: {chunk.name}")
                        t = call_transcription(
                            client,
                            chunk,
                            model=args.transcribe_model,
                            language=args.language,
                            prompt=args.prompt,
                        )
                        transcripts.append(t)
                    transcript = merge_transcripts(transcripts, args.chunk_seconds)
                    print("Transcription complete.")
            except RuntimeError as exc:
                print(
                    "Chunking failed. Install ffmpeg or reduce the file size, "
                    "then try again.",
                    file=sys.stderr,
                )
                print(f"Details: {exc}", file=sys.stderr)
                return 1
        else:
            print("Transcribing audio...")
            transcript = call_transcription(
                client,
                audio_path,
                model=args.transcribe_model,
                language=args.language,
                prompt=args.prompt,
            )
            print("Transcription complete.")

        transcript_text = transcript.get("text", "")
        segments = transcript.get("segments", [])
        if segments:
            duration_seconds = segments[-1].get("end")

    indexed_segments = []
    for idx, seg in enumerate(segments):
        indexed_segments.append(
            {
                "index": idx,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text"),
            }
        )
    payload = {
        "audio_file": audio_file_label,
        "duration_seconds": duration_seconds,
        "num_speakers_hint": args.num_speakers,
        "transcript_text": transcript_text,
        "segments": indexed_segments,
    }

    analysis: Dict[str, Any]
    if args.no_analysis:
        analysis = {}
    else:
        print("Analyzing transcript...")
        analysis = analyze_transcript(client, args.analysis_model, payload)
        print("Analysis complete.")

    meta = {
        "audio_file": audio_file_label,
        "duration_seconds": duration_seconds,
        "transcribe_model": args.transcribe_model,
        "analysis_model": args.analysis_model,
    }
    if args.style == "outline":
        markdown = render_outline_markdown(
            analysis,
            transcript_text,
            segments,
            meta,
            start_time_seconds,
            args.merge_gap_seconds,
            args.max_merge_seconds,
            args.max_merge_words,
            args.include_metadata,
            args.disclaimer,
        )
    else:
        markdown = render_markdown(
            analysis,
            transcript_text,
            segments,
            meta,
            start_time_seconds,
            args.merge_gap_seconds,
            args.max_merge_seconds,
            args.max_merge_words,
            args.disclaimer,
        )

    print(f"Writing {out_path}...")
    out_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {out_path}")

    if json_out_path:
        json_payload = {
            "meta": meta,
            "analysis": analysis,
            "transcript": {
                "text": transcript_text,
                "segments": segments,
            },
        }
        print(f"Writing {json_out_path}...")
        json_out_path.write_text(
            json.dumps(json_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"Wrote {json_out_path}")

    if raw_transcript_path and not args.analysis_only:
        print(f"Writing {raw_transcript_path}...")
        raw_transcript_path.write_text(
            json.dumps(transcript, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"Wrote {raw_transcript_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
