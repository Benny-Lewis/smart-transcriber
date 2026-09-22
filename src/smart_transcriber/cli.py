"""CLI orchestration; API access occurs only for missing paid stages."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

from openai import OpenAI, OpenAIError

from smart_transcriber import __version__
from smart_transcriber.analyze import ANALYSIS_PROMPT_VERSION, analyze_transcript, preserve_analysis_speakers, validate_analysis
from smart_transcriber.audio import known_speaker_references, probe_audio
from smart_transcriber.pipeline import (ANALYSIS_MODEL, DIARIZATION_MODEL, WORDING_MODEL,
    TranscriptionOptions, read_glossary, run_transcription, validate_options)
from smart_transcriber.reconcile import render_review
from smart_transcriber.render import render_markdown, render_outline_markdown, render_transcript_markdown
from smart_transcriber.storage import StageStore, atomic_json, atomic_text
from smart_transcriber.utils import DEFAULT_AUDIO_EXTS, DEFAULT_CHUNK_SECONDS, parse_time_of_day


class LazyOpenAI:
    """Cached runs and local rendering do not require credentials."""

    def __init__(self):
        self.client = None

    def __getattr__(self, name: str):
        if self.client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is not set. Set it locally to run an uncached API stage.")
            # A timed-out request may already have incurred charges: do not retry invisibly.
            self.client = OpenAI(max_retries=0)
        return getattr(self.client, name)


def resolve_path_with_extensions(path: Path, exts: list[str]) -> Path | None:
    if path.is_file():
        return path
    if not path.suffix:
        for ext in exts:
            candidate = path.with_suffix(ext)
            if candidate.is_file():
                return candidate
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI transcription with reusable audio-derived speaker labels and optional analysis.")
    parser.add_argument("audio_file", nargs="?", help="Audio/video file; extension may be omitted.")
    parser.add_argument("--mode", choices=["meeting", "text", "diarized"], default="meeting", help="meeting: two complementary audio passes (default); text or diarized: one pass.")
    parser.add_argument("--transcribe-model", default=WORDING_MODEL, help="Wording model (meeting/text modes).")
    parser.add_argument("--diarize-model", default=DIARIZATION_MODEL, help="Speaker model (scheduled retirement: 2027-02-26).")
    parser.add_argument("--analysis-model", default=ANALYSIS_MODEL, help="Optional report/analysis model.")
    parser.add_argument("--language", default="en", help="Language hint, or auto to omit (default: en).")
    parser.add_argument("--prompt", help="Recording context for the wording pass.")
    parser.add_argument("--glossary", help="UTF-8 file containing one terminology hint per line.")
    parser.add_argument("--known-speaker", action="append", default=[], metavar="NAME=PATH", help="Reference clip of 2–10 seconds; repeat up to four times.")
    parser.add_argument("--num-speakers", type=int, help="Legacy analysis context only; does not constrain audio diarization.")
    parser.add_argument("--out", help="Markdown output path (default: <audio_stem>.md).")
    parser.add_argument("--json-out", help="Combined analysis and transcript JSON output.")
    parser.add_argument("--transcript-json", help="Reusable transcript output (default: <output_stem>.transcript.json); raw responses are retained in the work directory.")
    parser.add_argument("--work-dir", help="Persistent work directory (default: <audio_stem>.transcribe beside Markdown output).")
    parser.add_argument("--force", action="store_true", help="Intentionally rerun active API stages; retain earlier raw responses.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and inspect audio without API calls or output writes.")
    parser.add_argument("--style", choices=["outline", "report", "transcript"], default="report")
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--disclaimer")
    parser.add_argument("--no-analysis", action="store_true", help="Skip analysis. Transcript style also needs no analysis call.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--analysis-only", action="store_true", help="Analyze saved JSON without processing audio.")
    group.add_argument("--render-only", action="store_true", help="Render saved JSON locally without API calls.")
    parser.add_argument("--transcript-input", help="Raw, reusable, or combined JSON for --analysis-only/--render-only.")
    parser.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--merge-gap-seconds", type=int, default=2)
    parser.add_argument("--max-merge-seconds", type=int, default=45)
    parser.add_argument("--max-merge-words", type=int, default=80)
    parser.add_argument("--start-time", help="Wall-clock start time, HH:MM or HH:MM:SS.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def load_transcript(path: Path) -> tuple[dict, dict]:
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(root, dict):
        raise ValueError("Transcript JSON must be an object.")
    transcript = copy.deepcopy(root.get("transcript", root))
    if not isinstance(transcript, dict) or not isinstance(transcript.get("text", ""), str) or not isinstance(transcript.get("segments", []), list):
        raise ValueError("Transcript must have string text and a list of segments.")
    if any(value is not None and not isinstance(value, dict) for value in (transcript.get("meta"), root.get("meta"))):
        raise ValueError("Transcript metadata must be an object.")
    meta = {**(transcript.get("meta") or {}), **(root.get("meta") or {})}
    meta.setdefault("audio_file", str(path))
    meta.setdefault("duration_seconds", transcript.get("duration"))
    meta.setdefault("transcribe_model", "unknown (imported transcript)")
    meta.setdefault("analysis_model", None)
    duration = meta.get("duration_seconds")
    if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0):
        raise ValueError("Invalid saved transcript duration.")
    transcript["meta"] = meta
    analysis = root.get("analysis") or {}
    validate_analysis(analysis)
    legacy = {a.get("segment_index"): a.get("speaker") for a in (analysis.get("segment_speakers") or []) if isinstance(a.get("segment_index"), int)}
    for index, segment in enumerate(transcript.get("segments", [])):
        if not isinstance(segment, dict) or not isinstance(segment.get("text", ""), str):
            raise ValueError("Invalid saved transcript segment.")
        for field in ("start", "end"):
            value = segment.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid saved segment {field}.")
        if segment.get("start") is not None and segment.get("end") is not None and segment["start"] > segment["end"]:
            raise ValueError("Saved segment ends before it starts.")
        if segment.get("speaker") is not None and not isinstance(segment["speaker"], str):
            raise ValueError("Saved speaker must be a string.")
        if not segment.get("speaker") and legacy.get(index):
            segment.update(speaker=legacy[index], speaker_source="legacy_inferred")
            meta["speaker_warning"] = "Imported speaker assignments were inferred by the old analysis workflow; they are not verified audio diarization."
    return transcript, analysis


def _run(args: argparse.Namespace) -> int:
    saved_input = args.analysis_only or args.render_only
    if args.analysis_only and args.no_analysis:
        raise ValueError("--analysis-only cannot be combined with --no-analysis.")
    if saved_input and not args.transcript_input:
        raise ValueError("--analysis-only/--render-only requires --transcript-input.")
    if args.analysis_only and args.style == "transcript":
        raise ValueError("Use --render-only for a saved transcript, or report/outline style for analysis.")
    if args.transcript_input and not saved_input:
        raise ValueError("Use --analysis-only or --render-only with --transcript-input.")
    if saved_input and args.audio_file:
        raise ValueError("Do not supply audio_file when processing saved JSON.")
    if args.num_speakers is not None and args.num_speakers <= 0:
        raise ValueError("--num-speakers must be positive.")
    if args.merge_gap_seconds < 0 or args.max_merge_seconds < 0 or args.max_merge_words < 0:
        raise ValueError("Merge limits must be nonnegative.")
    start_time = parse_time_of_day(args.start_time) if args.start_time else None
    if saved_input:
        source = resolve_path_with_extensions(Path(args.transcript_input), [".json"])
        if not source:
            raise ValueError(f"Transcript file not found: {args.transcript_input}")
    else:
        if not args.audio_file:
            raise ValueError("audio_file is required unless using saved JSON.")
        source = resolve_path_with_extensions(Path(args.audio_file), DEFAULT_AUDIO_EXTS)
        if not source:
            raise ValueError(f"Audio file not found: {args.audio_file}")
    out = Path(args.out) if args.out else Path.cwd() / ("notes.md" if saved_input else source.stem + ".md")
    transcript_out = Path(args.transcript_json) if args.transcript_json else out.with_suffix(".transcript.json")
    json_out = Path(args.json_out) if args.json_out else None
    review_out, wording_out = out.with_suffix(".review.md"), out.with_suffix(".wording.txt")
    # Load before checking output collisions so imported meeting artifacts get the same protection.
    transcript, analysis = load_transcript(source) if saved_input else ({}, {})
    has_review = "reconciliation" in transcript if saved_input else args.mode == "meeting"
    outputs = [out, transcript_out] + ([json_out] if json_out else []) + ([review_out, wording_out] if has_review else [])
    readonly = [source] + ([Path(args.glossary)] if args.glossary else [])
    readonly += [Path(item.partition("=")[2].strip()) for item in args.known_speaker if "=" in item]
    resolved = [p.resolve() for p in outputs]
    if len(set(resolved)) != len(resolved) or any(p.resolve() in resolved for p in readonly):
        raise ValueError("Output paths must be distinct and cannot overwrite an input file.")
    work_dir = Path(args.work_dir) if args.work_dir else out.parent / (source.stem + ".transcribe")
    client = LazyOpenAI()
    if not saved_input:
        options = TranscriptionOptions(mode=args.mode, transcribe_model=args.transcribe_model, diarize_model=args.diarize_model,
            language=args.language, prompt=args.prompt, glossary=read_glossary(Path(args.glossary) if args.glossary else None),
            known_speakers=known_speaker_references(args.known_speaker), chunk_seconds=args.chunk_seconds, force=args.force)
        validate_options(options)
        if args.num_speakers:
            print("Note: --num-speakers provides analysis context only; OpenAI diarization has no speaker-count parameter.", file=sys.stderr)
        if args.mode != "text":
            print("Note: gpt-4o-transcribe-diarize is scheduled to retire 2027-02-26. No equivalent newer speaker API is documented.", file=sys.stderr)
        if args.dry_run:
            print(json.dumps({"mode": args.mode, "transcribe_model": args.transcribe_model if args.mode != "diarized" else None,
                "diarize_model": args.diarize_model if args.mode != "text" else None, "audio": probe_audio(source),
                "glossary_terms": len(options.glossary), "known_speakers": [s["name"] for s in options.known_speakers]}, indent=2))
            return 0
        transcript = run_transcription(client, source, work_dir, options)
    if args.dry_run:
        print("Saved transcript is valid. No API calls or output writes.")
        return 0
    # Durable normalized export precedes analysis and presentation work.
    atomic_json(transcript_out, transcript)
    if has_review:
        atomic_text(review_out, render_review(transcript))
        atomic_text(wording_out, transcript.get("wording_text", ""))
    text, segments = transcript.get("text", ""), transcript.get("segments", [])
    meta = dict(transcript.get("meta") or {})
    if not args.render_only and not args.no_analysis and args.style != "transcript":
        payload = {"audio_file": meta.get("audio_file"), "duration_seconds": meta.get("duration_seconds"),
            "num_speakers_hint": args.num_speakers, "transcript_text": text,
            "segments": [{**s, "index": i} for i, s in enumerate(segments)],
            "reconciliation": transcript.get("reconciliation", {})}
        settings = {"prompt_version": ANALYSIS_PROMPT_VERSION, "model": args.analysis_model, "payload": payload}
        # Never write to arbitrary paths embedded in imported JSON.
        store = StageStore(work_dir / "analysis")
        cached = None if args.force else store.load("analysis", settings)
        if cached:
            try:
                validate_analysis(cached[0])
            except ValueError:
                cached = None
        if cached:
            analysis = preserve_analysis_speakers(cached[0], payload)
            print("Reusing saved analysis.")
        else:
            print(f"Analyzing saved transcript ({args.analysis_model})...")
            analysis = dict(analyze_transcript(client, args.analysis_model, payload,
                save_response=lambda response: store.save("analysis-response", settings, response)))
            validate_analysis(analysis)
            store.save("analysis", settings, analysis)
        meta["analysis_model"] = args.analysis_model
    elif not args.render_only:
        analysis = {}
        meta["analysis_model"] = None
    common = [analysis, text, segments]
    tail = [start_time, args.merge_gap_seconds, args.max_merge_seconds, args.max_merge_words]
    if args.style == "transcript":
        markdown = render_transcript_markdown(*common, *tail, args.disclaimer)
    elif args.style == "outline":
        markdown = render_outline_markdown(*common, meta, *tail, args.include_metadata, args.disclaimer)
    else:
        markdown = render_markdown(*common, meta, *tail, args.disclaimer)
    if meta.get("speaker_warning"):
        markdown += "\n> " + meta["speaker_warning"] + "\n"
    if has_review:
        report = transcript["reconciliation"]
        markdown += f"\nWording reconciliation: {report.get('accepted_count', 0)} substitutions; {report.get('review_count', 0)} differences requiring review. See {review_out.name}. Speaker labels and segment timing remain from the audio pass.\n"
    atomic_text(out, markdown)
    if json_out:
        atomic_json(json_out, {"meta": meta, "analysis": analysis, "transcript": transcript})
    print(f"Wrote {out}\nReusable transcript: {transcript_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return _run(args)
    except (ValueError, OSError, RuntimeError, OpenAIError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Completed API stages remain in the work directory. Rerun without --force to reuse them; an interrupted in-flight request cannot be recovered from OpenAI.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Completed stages are saved; rerun without --force to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
