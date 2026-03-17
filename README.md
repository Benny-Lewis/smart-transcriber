# smart-transcriber

Transcribe meeting audio and generate structured notes using OpenAI APIs.

## Install

```bash
pip install smart-transcriber
```

## Quick Start

```bash
export OPENAI_API_KEY="sk-..."
transcribe meeting.mp3
```

This produces `meeting.md` with a summary, decisions, action items, speaker-labeled transcript, and more.

## Requirements

- **Python 3.11+**
- **OpenAI API key** — set `OPENAI_API_KEY` environment variable
- **ffmpeg** — required for audio files larger than 25 MB (auto-detected)

Install ffmpeg: https://ffmpeg.org/download.html

## Usage

```bash
transcribe <audio_file> [options]
```

### Examples

```bash
# Basic transcription with summary
transcribe meeting.mp3

# Specify output path
transcribe meeting.mp3 --out notes.md --json-out notes.json

# Outline style instead of report
transcribe meeting.wav --style outline

# Use a different analysis model
transcribe meeting.m4a --analysis-model gpt-5-mini

# Transcription only (no summary/analysis)
transcribe meeting.mp3 --no-analysis

# Re-analyze a saved transcript
transcribe --analysis-only --transcript-input raw.json --out notes.md

# Render wall-clock timestamps
transcribe meeting.mp3 --start-time 09:30

# Provide hints for better results
transcribe meeting.mp3 --num-speakers 4 --prompt "Acme Corp, Project Phoenix"
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `audio_file` | — | Path to audio file (extension auto-detected) |
| `--out` | `<audio_stem>.md` | Markdown output path |
| `--json-out` | — | JSON output (analysis + transcript) |
| `--transcript-json` | — | Save raw transcription JSON |
| `--transcribe-model` | `whisper-1` | Transcription model |
| `--analysis-model` | `gpt-5.2` | Analysis/summary model |
| `--language` | `en` | Language hint for transcription |
| `--num-speakers` | — | Speaker count hint |
| `--prompt` | — | Transcription prompt (names, jargon) |
| `--style` | `report` | Output style: `report` or `outline` |
| `--include-metadata` | off | Show metadata in outline style |
| `--disclaimer` | — | Disclaimer text at top of notes |
| `--no-analysis` | off | Skip analysis, transcribe only |
| `--analysis-only` | off | Skip transcription, analyze saved JSON |
| `--transcript-input` | — | JSON input for `--analysis-only` |
| `--chunk-seconds` | `600` | Chunk length for large files |
| `--merge-gap-seconds` | `2` | Max gap to merge same-speaker segments |
| `--max-merge-seconds` | `45` | Max duration per merged line |
| `--max-merge-words` | `80` | Max words per merged line |
| `--start-time` | — | Render wall-clock timestamps (HH:MM or HH:MM:SS) |

## How It Works

The tool makes two OpenAI API calls per file:

1. **Transcription** — audio → text with timestamps (Whisper API)
2. **Analysis** — text → structured summary, speaker labels, decisions, action items (Chat API)

For large files (>25 MB), audio is automatically split into chunks using ffmpeg, transcribed separately, and merged.

## Performance

Expect processing time to scale with audio duration. For faster results:

```bash
# Use a smaller analysis model
transcribe meeting.mp3 --analysis-model gpt-5-mini

# Skip analysis entirely
transcribe meeting.mp3 --no-analysis
```

## License

[MPL 2.0](LICENSE)
