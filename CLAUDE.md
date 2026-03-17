# smart-transcriber

PyPI package `smart-transcriber` — CLI command is `transcribe`.

## Development

- Package source: `src/smart_transcriber/`
- Activate venv: `source .venv/Scripts/activate`
- Install editable: `pip install -e ".[dev]"`
- Run tests: `pytest tests/ -v` (81 tests)
- Build: `pip install build && python -m build`
- CLI verify: `transcribe --version` / `transcribe --help`

## Architecture

- `utils.py` — pure formatting (timestamps, text normalization). No I/O.
- `transcribe.py` — OpenAI transcription API, chunking, segment merging
- `analyze.py` — analysis prompt + AnalysisResult TypedDict schema contract
- `render.py` — markdown output (report + outline styles). Two independent entry points, shared helpers.
- `cli.py` — argparse + main() orchestration. Three modes: full pipeline, transcribe-only, analysis-only.

## Gotchas

- `normalize_text_ascii()` contains mojibake byte sequences — copy from source, don't retype
- `split_audio_ffmpeg()` takes `output_dir` param — caller manages `TemporaryDirectory` lifecycle
- PyPI name is `smart-transcriber`, Python import is `smart_transcriber`
- CI runs on Python 3.11, 3.12, 3.13 via GitHub Actions
