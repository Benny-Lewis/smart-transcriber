"""Tests for smart_transcriber.transcribe."""

from smart_transcriber.transcribe import (
    merge_display_segments,
    merge_transcripts,
    normalize_response,
    select_transcribe_format,
)


class TestNormalizeResponse:
    def test_dict_passthrough(self):
        d = {"text": "hello"}
        assert normalize_response(d) == d

    def test_model_dump(self):
        class FakeResponse:
            def model_dump(self):
                return {"text": "hello"}

        assert normalize_response(FakeResponse()) == {"text": "hello"}

    def test_dict_method(self):
        class OldResponse:
            def dict(self):
                return {"text": "hello"}

        assert normalize_response(OldResponse()) == {"text": "hello"}


class TestSelectTranscribeFormat:
    def test_whisper(self):
        fmt, gran = select_transcribe_format("whisper-1")
        assert fmt == "verbose_json"
        assert gran == ["segment"]

    def test_gpt4o_mini_transcribe(self):
        fmt, gran = select_transcribe_format("gpt-4o-mini-transcribe")
        assert fmt == "json"
        assert gran is None

    def test_gpt4o_transcribe(self):
        fmt, gran = select_transcribe_format("gpt-4o-transcribe")
        assert fmt == "json"
        assert gran is None


class TestMergeTranscripts:
    def test_single_transcript(self):
        transcripts = [{"text": "Hello", "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
        ]}]
        result = merge_transcripts(transcripts, 600)
        assert result["text"] == "Hello"
        assert len(result["segments"]) == 1
        assert result["segments"][0]["start"] == 0.0

    def test_two_chunks_offset_correction(self):
        transcripts = [
            {"text": "First", "segments": [
                {"start": 0.0, "end": 5.0, "text": "First"},
            ]},
            {"text": "Second", "segments": [
                {"start": 0.0, "end": 3.0, "text": "Second"},
            ]},
        ]
        result = merge_transcripts(transcripts, 600)
        assert result["text"] == "First\nSecond"
        assert len(result["segments"]) == 2
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 5.0
        # Second chunk offset by first chunk's last end (5.0)
        assert result["segments"][1]["start"] == 5.0
        assert result["segments"][1]["end"] == 8.0

    def test_no_segments_uses_chunk_seconds_for_offset(self):
        transcripts = [
            {"text": "First"},
            {"text": "Second", "segments": [
                {"start": 0.0, "end": 2.0, "text": "Second"},
            ]},
        ]
        result = merge_transcripts(transcripts, 600)
        # First has no segments, offset advances by chunk_seconds (600)
        assert result["segments"][0]["start"] == 600.0

    def test_empty_input(self):
        result = merge_transcripts([], 600)
        assert result["text"] == ""
        assert "segments" not in result


class TestMergeDisplaySegments:
    def test_single_segment(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        speaker_map = {0: "Speaker 1"}
        result = merge_display_segments(segments, speaker_map, 2, 45, 80)
        assert len(result) == 1
        assert result[0]["speaker"] == "Speaker 1"
        assert result[0]["text"] == "Hello"

    def test_same_speaker_merges(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.5, "end": 2.5, "text": "world"},
        ]
        speaker_map = {0: "Speaker 1", 1: "Speaker 1"}
        result = merge_display_segments(segments, speaker_map, 2, 45, 80)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"
        assert result[0]["end"] == 2.5

    def test_different_speakers_no_merge(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.5, "end": 2.5, "text": "Hi"},
        ]
        speaker_map = {0: "Speaker 1", 1: "Speaker 2"}
        result = merge_display_segments(segments, speaker_map, 2, 45, 80)
        assert len(result) == 2

    def test_gap_too_large_no_merge(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 5.0, "end": 6.0, "text": "world"},
        ]
        speaker_map = {0: "Speaker 1", 1: "Speaker 1"}
        result = merge_display_segments(segments, speaker_map, 2, 45, 80)
        assert len(result) == 2

    def test_word_limit_prevents_merge(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "word " * 70},
            {"start": 1.5, "end": 2.5, "text": "word " * 20},
        ]
        speaker_map = {0: "Speaker 1", 1: "Speaker 1"}
        result = merge_display_segments(segments, speaker_map, 2, 45, 80)
        assert len(result) == 2

    def test_duration_limit_prevents_merge(self):
        segments = [
            {"start": 0.0, "end": 40.0, "text": "Long segment"},
            {"start": 41.0, "end": 50.0, "text": "Another"},
        ]
        speaker_map = {0: "Speaker 1", 1: "Speaker 1"}
        result = merge_display_segments(segments, speaker_map, 2, 45, 80)
        assert len(result) == 2

    def test_empty_text_skipped(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": ""},
            {"start": 1.0, "end": 2.0, "text": "Hello"},
        ]
        speaker_map = {0: "Speaker 1", 1: "Speaker 1"}
        result = merge_display_segments(segments, speaker_map, 2, 45, 80)
        assert len(result) == 1
        assert result[0]["text"] == "Hello"

    def test_missing_speaker_defaults(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        speaker_map = {}
        result = merge_display_segments(segments, speaker_map, 2, 45, 80)
        assert result[0]["speaker"] == "Speaker"
