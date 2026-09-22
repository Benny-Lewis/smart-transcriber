# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- OpenAI-only `meeting` (default), `text`, and `diarized` modes, with `gpt-transcribe` wording and `gpt-4o-transcribe-diarize` native speakers/timing.
- Glossary/context support, up to four known-speaker reference clips, and model-specific request validation.
- Conservative local reconciliation, readable review reports, immutable original responses and derived change history.
- Atomic per-stage checkpoints, independent reuse, explicit `--work-dir`, `--force`, `--dry-run`, and local `--render-only`.
- Shared package-backed Codex skill, compatibility launcher, dated capability sources and diarization retirement guidance.
- Request serialization, real stereo audio preparation, boundary/repetition, interruption, cache, and legacy-import regression coverage.

### Changed
- Development version 0.3.0; optional analysis defaults to configurable `gpt-6-astra`.
- Default meeting mode now makes two paid audio passes; reports add a text-analysis call. Transcript style works with `--no-analysis`.
- Speaker labels come from audio; text analysis cannot reassign them. Legacy inferred labels are marked on import.
- Prefer whole uploads, preserve channels, enforce upload sizes, and use original source offsets for necessary chunks, including silent tails.
- `--transcript-json` exports a reusable normalized transcript; original provider responses are retained in the work directory.
- The SDK no longer retries requests invisibly. Resume completed work without `--force`.

### Needs validation
- No representative recording accuracy evaluation yet. Reconciled changes are hypotheses; originals and review records are retained.
- The diarization model retires 2027-02-26; current OpenAI documentation has no equivalent newer native speaker-label replacement.

## [0.1.0] - 2026-03-16

### Added
- Initial release as a PyPI package
- Transcribe audio files using OpenAI Whisper API
- Analyze transcripts for summaries, decisions, action items, and speaker labels
- Two output styles: report (default) and outline
- Automatic chunking of large files via ffmpeg
- Time-of-day timestamp rendering with `--start-time`
- Analysis-only mode for re-processing saved transcripts
