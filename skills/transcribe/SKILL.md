---
name: transcribe
description: Transcribe recorded interviews, meetings, or other audio/video with technical vocabulary hints, audio-derived speaker labels, and reusable outputs. Use for transcription or review of recorded speech.
---

# Recorded audio transcription

Use the maintained `smart-transcriber` package for processing. This skill chooses the workflow; it does not maintain a separate API implementation.

## Choose capabilities, then models

- Respect explicit user model/provider choices. This installation uses OpenAI only, with `OPENAI_API_KEY`.
- For Ben's technical interviews and meetings, use `--mode meeting`: one wording pass with `gpt-transcribe`, plus one audio diarization pass with `gpt-4o-transcribe-diarize`. Two passes are an intentional quality tradeoff, not proof that every disagreement is a correction.
- For text without speakers, choose `--mode text` and provide relevant terminology/context. For one speaker-labeling pass, choose `--mode diarized`; it cannot accept a glossary or prompt.
- When asked for the best/latest model, when the recording's needs change, or when a model becomes unavailable, check current official model capabilities and deprecations. Do not recommend a model merely because it appears in this skill. Reuse a recently validated choice for equivalent recordings rather than researching or benchmarking every file.
- The diarization model is scheduled to retire **2027-02-26**. As researched **2026-09-17**, OpenAI documents no equivalent newer native speaker-label response. Surface that limitation; never substitute guessed speakers from a text model.
- Read [references/api.md](references/api.md) for model constraints, sources, output meanings, and the retirement gap.

## Run the shared implementation

Preferred interpreter for this installation (PowerShell):

```powershell
$transcriberPython = 'C:\Users\Ben\dev\audio-transcriber\.venv\Scripts\python.exe'
& $transcriberPython -m smart_transcriber --help
& $transcriberPython -m smart_transcriber 'interview.m4a' --mode meeting --glossary 'terms.txt' --style transcript --no-analysis --out 'output/transcribe/interview/transcript.md'
```

Omit `--glossary` when no terminology list is available. Supply only names and terms supported by the user's context; do not invent a glossary or force expected words into the output. `--prompt` supplies recording context to the wording pass. Use `--language auto` if the language is unknown. Up to four `--known-speaker 'Name=reference.wav'` clips may be supplied when available; otherwise preserve generic speaker IDs.

For another machine, use an interpreter with this repository installed: `python -m pip install -e '.[dev]'`. Verify package version 0.3.0 or newer. Do not fall back to an unrelated stale `transcribe` executable on PATH. If the key is missing, ask the user to set it locally, never paste it in chat.

Transcribe without an analysis call unless notes or analysis are requested. Use saved JSON for later work:

```powershell
& $transcriberPython -m smart_transcriber --analysis-only --transcript-input 'transcript.transcript.json' --out 'notes.md'
& $transcriberPython -m smart_transcriber --render-only --transcript-input 'transcript.transcript.json' --style transcript --out 'review-copy.md'
```

## Review and reuse

- Inspect the transcript, companion wording text, and review report. Preserve uncertainty around technical terms, numbers, negations, overlap, and speaker boundaries. Native speaker timing is not word-level alignment.
- Both original API responses and per-stage settings are saved in the work directory before analysis. Reconciliation history is retained under `derived/`. Rerun without `--force` to reuse completed stages; `--work-dir` keeps that location explicit. A glossary change reruns only wording. Use `--force` only for intentional reprocessing.
- Unknown speaker IDs are scoped to their chunk when splitting is necessary. Do not identify speakers across chunks from matching letters or conversation content.
- Accepted reconciliation edits are small anchored substitutions. Unresolved differences retain original diarized wording and appear in the review report. Do not silently apply speculative rewrites or launch more audio passes.
- A live accuracy comparison needs a representative audio sample; mocked tests only verify tooling behavior. Report whether audio quality was actually evaluated.

`scripts/transcribe_diarize.py` is a compatibility launcher for the old script's arguments. It delegates to `smart_transcriber.compat`; new meeting work should use the package CLI above.
