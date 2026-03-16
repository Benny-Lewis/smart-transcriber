# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.1.0] - 2026-03-16

### Added
- Initial release as a PyPI package
- Transcribe audio files using OpenAI Whisper API
- Analyze transcripts for summaries, decisions, action items, and speaker labels
- Two output styles: report (default) and outline
- Automatic chunking of large files via ffmpeg
- Time-of-day timestamp rendering with `--start-time`
- Analysis-only mode for re-processing saved transcripts
