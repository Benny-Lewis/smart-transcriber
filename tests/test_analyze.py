"""Tests for smart_transcriber.analyze."""

import json
from types import SimpleNamespace

import pytest

from smart_transcriber.analyze import AnalysisResult, build_analysis_prompt
from smart_transcriber.analyze import analyze_transcript, validate_analysis


@pytest.mark.parametrize("value", [[], {"meta": {"participants": 5}}, {"sections": [{"heading": 7}]},
    {"meta": {"topics": [1]}}, {"qa": [{"answers": "wrong"}]}, {"annotations": [{"note": {}}]},
    {"annotations": [{"timestamp": "inf"}]}])
def test_invalid_nested_analysis_is_rejected(value):
    with pytest.raises(ValueError):
        validate_analysis(value)


def test_malformed_paid_analysis_is_saved_before_parsing():
    from openai.types.chat import ChatCompletion
    response = ChatCompletion(id="fixture", created=0, model="test", object="chat.completion",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "not valid json"}}])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    saved = []
    with pytest.raises(ValueError):
        analyze_transcript(client, "test", {}, save_response=saved.append)
    assert saved[0]["choices"][0]["message"]["content"] == "not valid json"


def test_analysis_never_reassigns_native_speakers():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
        "summary": "Summary", "speakers": [{"label": "invented"}, {"label": "A"}],
        "annotations": [{"speaker": "Alice", "note": "Unsupported identity"}, {"speaker": "A", "note": "Known ID"}],
        "segment_speakers": [{"segment_index": 0, "speaker": "invented"}]})))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    result = analyze_transcript(client, "test", {"segments": [{"index": 0, "speaker": "A"}, {"index": 1}]})
    assert result["segment_speakers"] == [{"segment_index": 0, "speaker": "A"}]
    assert result["speakers"] == [{"label": "A"}]
    assert result["annotations"][0]["speaker"] is None
    assert result["annotations"][1]["speaker"] == "A"


class TestAnalysisResultTypedDict:
    def test_has_expected_keys(self):
        """AnalysisResult TypedDict should define all expected schema keys."""
        expected_keys = {
            "summary", "meta", "sections", "qa", "speakers",
            "decisions", "action_items", "annotations", "segment_speakers",
        }
        assert set(AnalysisResult.__annotations__.keys()) == expected_keys


class TestBuildAnalysisPrompt:
    def test_contains_schema_keys(self):
        payload = {
            "audio_file": "test.mp3",
            "duration_seconds": 60.0,
            "num_speakers_hint": None,
            "transcript_text": "Hello world",
            "segments": [],
        }
        result = build_analysis_prompt(payload)
        for key in ["summary", "meta", "sections", "qa", "speakers",
                     "decisions", "action_items", "annotations", "segment_speakers"]:
            assert key in result, f"Missing key '{key}' in prompt"

    def test_contains_payload_json(self):
        payload = {
            "audio_file": "test.mp3",
            "duration_seconds": 60.0,
            "num_speakers_hint": 2,
            "transcript_text": "Hello",
            "segments": [{"index": 0, "start": 0.0, "end": 1.0, "text": "Hello"}],
        }
        result = build_analysis_prompt(payload)
        assert '"audio_file": "test.mp3"' in result
        assert '"num_speakers_hint": 2' in result

    def test_empty_segments(self):
        payload = {
            "audio_file": "test.mp3",
            "duration_seconds": None,
            "num_speakers_hint": None,
            "transcript_text": "",
            "segments": [],
        }
        result = build_analysis_prompt(payload)
        assert "Input JSON:" in result

    def test_returns_string(self):
        payload = {"audio_file": "x", "duration_seconds": 0,
                   "num_speakers_hint": None, "transcript_text": "", "segments": []}
        assert isinstance(build_analysis_prompt(payload), str)

    def test_transcript_only_includes_speaker_keys(self):
        payload = {
            "audio_file": "test.mp3",
            "duration_seconds": 60.0,
            "num_speakers_hint": None,
            "transcript_text": "Hello",
            "segments": [{"index": 0, "start": 0.0, "end": 1.0, "text": "Hello"}],
        }
        result = build_analysis_prompt(payload, transcript_only=True)
        assert "speakers" in result
        assert "segment_speakers" in result

    def test_transcript_only_excludes_full_analysis_keys(self):
        payload = {
            "audio_file": "test.mp3",
            "duration_seconds": 60.0,
            "num_speakers_hint": None,
            "transcript_text": "Hello",
            "segments": [{"index": 0, "start": 0.0, "end": 1.0, "text": "Hello"}],
        }
        result = build_analysis_prompt(payload, transcript_only=True)
        assert "summary" not in result
        assert "sections" not in result
        assert "decisions" not in result
        assert "action_items" not in result
        assert "annotations" not in result
