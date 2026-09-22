"""Exercise installed SDK multipart serialization with an in-memory HTTP transport."""

import json
from email import policy
from email.parser import BytesParser

import httpx
from openai import OpenAI

from smart_transcriber.transcribe import call_transcription


def test_real_sdk_request_compatibility_without_network(tmp_path):
    captured = []

    def respond(request):
        multipart = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + request.headers["Content-Type"].encode() + b"\r\n\r\n" + request.read())
        fields = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True).decode()
                  for part in multipart.iter_parts()}
        captured.append(fields)
        response = {"text": "Hello", "segments": [{"id": "s", "speaker": "Jane", "start": 0, "end": 1, "text": "Hello"}]}
        return httpx.Response(200, json=response)

    source = tmp_path / "fixture.wav"
    source.write_bytes(b"synthetic")
    with OpenAI(api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        call_transcription(client, source, "gpt-transcribe", "en", "Engineering", keywords=["C++", ".NET"])
        result = call_transcription(client, source, "gpt-4o-transcribe-diarize", "en", None,
            known_speakers=[{"name": "Jane", "data_url": "data:audio/wav;base64,AA=="}])
    wording, diarized = captured
    assert wording["response_format"] == "json"
    assert wording["languages[]"] == "en"
    assert wording["prompt"] == "Engineering"
    assert "language" not in wording and "timestamp_granularities[]" not in wording
    assert diarized["response_format"] == "diarized_json"
    assert diarized["language"] == "en" and diarized["chunking_strategy"] == "auto"
    assert diarized["known_speaker_names[]"] == "Jane"
    assert diarized["known_speaker_references[]"] == "data:audio/wav;base64,AA=="
    assert "prompt" not in diarized and "keywords[]" not in diarized
    assert result["segments"][0]["speaker"] == "Jane"
