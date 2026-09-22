# smart-transcriber

OpenAI-only transcription for recorded interviews and meetings, with technical vocabulary hints, audio-derived speaker labels, reusable results, and optional analysis.

## Install from this checkout

Requires Python 3.11+, **ffmpeg and ffprobe on PATH**, and your existing `OPENAI_API_KEY` for uncached API work. No additional provider account is required.

```powershell
python -m pip install -e '.[dev]'
python -m smart_transcriber --version
```

The PyPI package and CLI are named `smart-transcriber` and `transcribe`, respectively. These changes are version **0.3.0 in this checkout**; installing an older published version will not provide this workflow. [FFmpeg installation](https://ffmpeg.org/download.html).

## Choose the workflow

```powershell
# Interviews/meetings: two audio passes, transcript only
transcribe interview.m4a --mode meeting --style transcript --no-analysis --glossary terms.txt --out interview.md

# Full report: the same audio passes followed by optional text analysis
transcribe meeting.wav --out notes.md --json-out notes.json

# One pass when you only need text or only need audio-based speaker labels
transcribe interview.m4a --mode text --prompt 'Technical interview about Kubernetes' --no-analysis
transcribe meeting.wav --mode diarized --style transcript --no-analysis
```

| Mode | Audio model(s) | Capabilities |
|---|---|---|
| `meeting` (default) | `gpt-transcribe` + `gpt-4o-transcribe-diarize` | Contextual wording plus native speakers/timing; conservative local reconciliation |
| `text` | `gpt-transcribe` | Wording with vocabulary/context; no invented speakers or timestamps |
| `diarized` | `gpt-4o-transcribe-diarize` | Native speaker labels and timed segments; no prompt/glossary support |

The existing full-report default is preserved: `--style report` runs analysis using configurable `--analysis-model gpt-6-astra`. Use `--style transcript --no-analysis` for later review without paying for summaries. Transcript style always skips analysis; `--style outline` is also available.

**Cost change:** meeting mode intentionally transcribes the audio twice. As checked September 17, 2026, the wording pass costs $0.0045/minute ($0.27/hour), **plus separate diarization and any analysis charges**. One-pass modes reduce processing. The tool retains API usage metadata; it does not estimate a fixed total bill. [Wording model pricing](https://developers.openai.com/api/docs/models/gpt-transcribe).

**Retirement:** `gpt-4o-transcribe-diarize` is scheduled for removal on **February 26, 2027**. Current OpenAI documentation does not establish an equivalent newer native diarization replacement. The dependency is explicit; availability errors never trigger an automatic model substitution or text-inferred speakers. [OpenAI deprecations](https://developers.openai.com/api/docs/deprecations#2026-08-26-transcription-models).

## Technical terminology and known speakers

`--glossary terms.txt` reads UTF-8, one term per line. Blank lines and duplicates are ignored; `<` and `>` are unsupported. Preserve meaningful spelling and punctuation, such as `kubectl`, `C++`, `C#`, and `.NET`. `--prompt` supplies recording context. Both go only to the wording model. Hints can bias recognition; use terms supported by the recording's context.

```powershell
transcribe interview.wav --glossary terms.txt --prompt 'Engineering interview with Jane about Kubernetes operations' --known-speaker 'Jane=jane-reference.wav' --style transcript --no-analysis
```

Up to four distinct speaker names may have 2–10-second audio-only reference clips. No references are required. `--language en` is the default; `--language auto` omits the hint. The package maps the hint to `languages` for `gpt-transcribe` and `language` for diarization. Unsupported combinations fail before any audio API request. `--num-speakers` remains a legacy analysis hint, **not an audio diarization constraint**.

## Saved work and reuse

For `--out interview.md`, outputs are:

| Artifact | Contents |
|---|---|
| `interview.md` | Readable transcript or report |
| `interview.transcript.json` | Reusable normalized transcript, provenance, native segment metadata and reconciliation records |
| `interview.wording.txt` | Complete wording-pass alternative (meeting mode) |
| `interview.review.md` | Accepted substitutions and unresolved differences (meeting mode) |
| `<audio_stem>.transcribe/` | Durable work directory: immutable raw API responses, stage indexes, prepared audio, derived history, analysis results |

The work directory defaults beside the Markdown output. Use `--work-dir output/interview-work` to keep it fixed when changing output locations. `--transcript-json` overrides the normalized export path; it is no longer a raw-response export. Original provider responses remain under the source-hash job's `raw/` directory, with paths recorded in transcript metadata. Historical reconciliations are retained under `derived/`; the readable exports show the latest result.

Completed stages are written atomically and reused when audio bytes, model, and relevant request settings match. Changing the glossary reruns **only wording**. Changing analysis settings never retranscribes matching audio. A failed analysis leaves both paid transcripts available. Resume with the same work directory **without `--force`**. An interrupted request with no returned response cannot be recovered; that one request may require processing again.

`--force` intentionally reprocesses active API stages and retains earlier originals/history. The CLI disables automatic SDK retries so a timeout does not silently repeat a potentially paid request. Avoid concurrent runs against the same job: caching is resumable but does not lock out duplicate in-flight requests. Saved artifacts contain recording content and should be handled like the recording itself.

```powershell
# Analyze an existing transcript; no audio calls
transcribe --analysis-only --transcript-input interview.transcript.json --out notes.md

# Render saved JSON locally, without an API key
transcribe --render-only --transcript-input notes.json --style transcript --out review-copy.md

# Validate configuration and inspect media without API calls or output writes
transcribe interview.wav --mode meeting --glossary terms.txt --dry-run
```

Imports accept older combined `{meta, analysis, transcript}` JSON, flat transcript exports, and raw transcription JSON. Existing files are not migrated or deleted. Imported legacy text-inferred speaker assignments are explicitly marked as unverified. Use a different output path to preserve an older human-edited report; explicit output paths are replaced atomically. Inputs cannot be overwritten by outputs.

## Reconciliation and audio preparation

The diarization response supplies authoritative speaker IDs and segment times. The companion transcript can supply alternative wording, never a new speaker or word timestamp. Only 1–4-token substitutions wholly inside one native segment, with unique exact context on both sides, are eligible for automatic incorporation. Punctuation is retained. The originals and all change records remain available.

Numbers/versions (including common English number words), negations, insertions, deletions, overlapping speech, speaker boundaries, and ambiguous/repeated context remain unchanged and go into the review report. These are conservative heuristics, not a semantic guarantee for every language. Alignment has token/work limits; if exceeded or reconciliation fails, the native transcript remains usable with a review flag.

**Needs validation:** two passes provide complementary evidence, not proof of better wording. No representative recording has yet been compared against a human-checked reference. Evaluate terminology errors, speaker attribution, unresolved changes, and total cost on a representative excerpt before describing the derived transcript as more accurate.

Supported audio under the conservative 24,000,000-byte upload ceiling is sent whole. Other formats/video are converted to lossless audio-only FLAC while preserving channels and sample rate. Large audio is split only if a whole upload remains too large; every chunk's size is checked. Offsets come from the original audio timeline, including silence. Unidentified speakers across separate uploads are named `chunk_1:A`, `chunk_2:A`, etc.; matching letters do not establish identity. Reference-backed names can span chunks. Multiple audio tracks require choosing/exporting the intended track first.

`--chunk-seconds` (default 600) caps necessary chunks, not all recordings. Display merging uses `--merge-gap-seconds 2`, `--max-merge-seconds 45`, and `--max-merge-words 80`. It retains native identities and does not merge overlapping segments. `--start-time 09:30` renders wall-clock times. See `transcribe --help` for all options.

## Codex skill and maintenance

The maintained skill source lives in [`skills/transcribe`](skills/transcribe/SKILL.md). The installed personal skill at `C:\Users\Ben\.codex\skills\transcribe` now calls this package using the project virtual environment. Its old `scripts/transcribe_diarize.py` entry point is a thin launcher for `smart_transcriber.compat`, preserving old one-pass arguments and local text/JSON export. Legacy `--chunking-strategy` supports only `auto`; custom strategies are rejected. `SMART_TRANSCRIBER_PYTHON` can select another interpreter with the package installed.

Skill guidance selects capabilities before models, defaults to transcript-only output for transcription requests, and checks current documentation when asked for best/latest models, when requirements change, or when availability fails. Package code owns requests, chunking, caching and reconciliation. Keep repository and installed skill copies synchronized when editing the skill. Existing skill icons and license are retained.

Research checked **2026-09-17**: [transcription guide](https://developers.openai.com/api/docs/guides/speech-to-text), [request reference](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create), [diarization model](https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize), [analysis model](https://developers.openai.com/api/docs/models/gpt-6-astra). The current guide recommends `gpt-transcribe` for recorded wording; native speaker metadata still needs the diarization model. General audio chat is not a documented drop-in replacement for that response contract. See the [dated capability notes](skills/transcribe/references/api.md).

## Development

```powershell
python -m pytest tests/ -v
python -m pip install build
python -m build
transcribe --version
transcribe --help
```

Tests use mock API responses/HTTP transport and generated audio, with no paid requests. FFmpeg tests require ffmpeg/ffprobe and run in CI. API account/model availability and recording accuracy need live validation separately.

## License

[MPL 2.0](LICENSE). The bundled Codex skill retains its [original license](skills/transcribe/LICENSE.txt).
