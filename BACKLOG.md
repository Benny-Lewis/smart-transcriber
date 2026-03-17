# Backlog

Post-MVP ideas for smart-transcriber. Items roughly ordered by expected value.

## Near-term

- [ ] Robust audio duration detection (currently derived from last segment end time)
- [ ] `RenderConfig` dataclass to bundle shared render parameters
- [ ] Model alias defaults (e.g., `gpt-5` instead of `gpt-5.2`) to reduce deprecation risk
- [ ] `--verbose` / `--quiet` flags (convert print() to logging module)
- [ ] Progress indicators for long transcriptions
- [ ] `py.typed` marker + mypy enforcement
- [ ] Config file support (`~/.smart-transcriber.toml`)
- [ ] Mocked API integration tests
- [ ] JSON schema versioning for intermediate transcript format

## Medium-term

- [ ] Multi-provider support (Anthropic, local Whisper, Deepgram)
- [ ] Custom output templates
- [ ] Additional output formats (HTML, plain text, JSON-only)
- [ ] Speaker diarization via dedicated models
- [ ] Batch mode (multiple files)
- [ ] `.env` file support
- [ ] Stable programmatic API (if demand exists)

## Long-term

- [ ] Real-time / streaming transcription
- [ ] Web UI frontend
- [ ] Docker image
- [ ] GitHub Action for automated meeting notes
