"""Tests for smart_transcriber.analyze."""

import json

from smart_transcriber.analyze import AnalysisResult, build_analysis_prompt


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
