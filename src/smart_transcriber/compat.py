"""Legacy skill CLI arguments, delegating all transcription to the package pipeline."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from openai import OpenAIError

from smart_transcriber.audio import known_speaker_references, probe_audio
from smart_transcriber.cli import LazyOpenAI
from smart_transcriber.pipeline import (DIARIZATION_MODEL, WORDING_MODEL, TranscriptionOptions,
                                      run_transcription, validate_options)
from smart_transcriber.storage import atomic_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compatibility launcher. For two-pass meetings use: python -m smart_transcriber --mode meeting")
    parser.add_argument("audio", nargs="+")
    parser.add_argument("--model", default=WORDING_MODEL)
    parser.add_argument("--response-format", choices=["text", "json", "diarized_json"], default="text")
    parser.add_argument("--language", default="en")
    parser.add_argument("--prompt")
    parser.add_argument("--known-speaker", action="append", default=[])
    parser.add_argument("--chunking-strategy", default="auto", choices=["auto"])
    parser.add_argument("--out")
    parser.add_argument("--out-dir")
    parser.add_argument("--work-dir")
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.out and args.out_dir:
            raise ValueError("Choose either --out or --out-dir.")
        if (args.out or args.stdout) and len(args.audio) != 1:
            raise ValueError("--out and --stdout require a single audio file.")
        if args.stdout and (args.out or args.out_dir):
            raise ValueError("--stdout cannot be combined with --out or --out-dir.")
        diarized = args.model == DIARIZATION_MODEL
        if args.response_format == "diarized_json" and not diarized:
            raise ValueError("diarized_json requires the diarization model.")
        options = TranscriptionOptions(mode="diarized" if diarized else "text",
            transcribe_model=WORDING_MODEL if diarized else args.model, diarize_model=DIARIZATION_MODEL,
            language=args.language, prompt=args.prompt, known_speakers=known_speaker_references(args.known_speaker), force=args.force)
        validate_options(options)
        audios = [Path(p) for p in args.audio]
        outputs = []
        for audio in audios:
            if not audio.is_file():
                raise ValueError(f"Audio file not found: {audio}")
            extension = ".txt" if args.response_format == "text" else ".json"
            output = Path(args.out) if args.out else Path(args.out_dir or ".") / f"{audio.stem}.transcript{extension}"
            if output.is_dir():
                output = output / f"{audio.stem}.transcript{extension}"
            elif not output.suffix:
                output = output.with_suffix(extension)
            outputs.append(output)
        paths = [p.resolve() for p in outputs]
        inputs = [p.resolve() for p in audios] + [Path(item.partition("=")[2]).resolve() for item in args.known_speaker]
        if not args.stdout and (len(set(paths)) != len(paths) or set(paths).intersection(inputs)):
            raise ValueError("Output paths must be distinct and cannot overwrite input files.")
        if args.dry_run:
            for path in audios:
                print(json.dumps({"file": str(path), "audio": probe_audio(path), "model": args.model,
                    "response_format": "diarized_json" if diarized else "json", "export_format": args.response_format,
                    "known_speakers": [s["name"] for s in options.known_speakers]}, indent=2))
            return 0
        for audio, output in zip(audios, outputs):
            work = Path(args.work_dir) if args.work_dir else output.parent / f"{audio.stem}.transcribe"
            with contextlib.redirect_stdout(sys.stderr):
                transcript = run_transcription(LazyOpenAI(), audio, work, options)
            rendered = transcript["text"] if args.response_format == "text" else json.dumps(transcript, indent=2, ensure_ascii=False)
            if args.stdout:
                print(rendered)
            else:
                atomic_text(output, rendered)
                print(f"Wrote {output}", file=sys.stderr)
        return 0
    except (ValueError, OSError, RuntimeError, OpenAIError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
