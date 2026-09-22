# OpenAI recorded-audio capabilities

Research checked 2026-09-17. These are dated findings, not a permanent ranking.

| Requirement | Model / behavior |
|---|---|
| Current recorded-speech wording | `gpt-transcribe`, JSON response, `prompt`, `keywords`, and `languages` hints; no native speaker segments |
| Audio-based speakers and segment timing | `gpt-4o-transcribe-diarize`, `diarized_json`, `chunking_strategy=auto`; no prompt/glossary support |
| Optional analysis | `gpt-6-astra`, text analysis of saved results; cannot hear the recording or establish new speaker identities |

- OpenAI's guide recommends `gpt-transcribe` for recorded speech. Its listed price is $0.0045/minute ($0.27/hour). Diarization and optional text analysis add separate charges. This tool records original usage responses; it does not promise a fixed total price.
- API uploads are limited to 25 MB. The package uses a conservative 24,000,000-byte ceiling, preserves channels, and splits with original-recording offsets only when needed. ffmpeg and ffprobe are required for media inspection/preparation.
- Diarization needs server chunking for audio longer than 30 seconds. Server chunking does not remove the upload limit.
- Known speakers: at most four distinct names with matching 2–10 second audio references. Unidentified speakers remain generic. Separate requests do not guarantee consistent unidentified speaker IDs.
- For `gpt-transcribe`, send `languages`, not singular `language`; the package maps `--language` correctly. Glossary entries cannot contain `<`, `>`, or embedded newlines.
- `gpt-4o-transcribe` and mini use JSON API responses. Text export is a local formatting choice, not a reason to send an unsupported API response format.
- Whisper and GPT-4o transcription models, including diarize, are deprecated with scheduled removal on **2027-02-26**. The current replacement recommendations do not document equivalent native diarization. General audio-chat prompting is not a validated replacement for speaker/timestamp metadata.

## Reconciliation contract

Meeting mode saves both raw responses. The derived transcript retains diarizer speaker IDs and segment times. Only substitutions of 1–4 tokens wholly inside a segment, bounded by unique exact 3–8-token context on both sides, are eligible. Numeric/version and negation changes, overlap, insertions/deletions, missing/repeated anchors, and cross-segment changes require review. Matching text is location evidence, not acoustic proof. Work limits can leave the entire comparison for review.

The companion wording text and review report preserve alternatives. The transcript JSON records dispositions, spans and anchor evidence. Do not call changes verified unless someone actually checked the audio.

## Sources

- [OpenAI file transcription guide](https://developers.openai.com/api/docs/guides/speech-to-text)
- [Transcription request reference](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)
- [GPT-Transcribe model and pricing](https://developers.openai.com/api/docs/models/gpt-transcribe)
- [GPT-4o Transcribe Diarize](https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize)
- [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)
- [Deprecations](https://developers.openai.com/api/docs/deprecations#2026-08-26-transcription-models)

Other providers were considered during research, but this installation intentionally requires only OpenAI. Retain that choice unless the user changes it.
