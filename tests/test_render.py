"""Tests for smart_transcriber.render."""

from smart_transcriber.render import (
    build_annotated_transcript_lines,
    filter_participants,
    format_list,
    render_markdown,
    render_outline_markdown,
    render_transcript_markdown,
)


class TestFilterParticipants:
    def test_all_mentioned(self):
        participants = ["Alice", "Bob"]
        text = "Alice said hello. Alice agreed. Bob replied. Bob nodded."
        result = filter_participants(participants, text)
        assert result == ["Alice", "Bob"]

    def test_one_not_mentioned_enough(self):
        participants = ["Alice", "Bob"]
        text = "Alice said hello. Alice agreed. Bob replied."
        result = filter_participants(participants, text)
        assert result == ["Alice"]

    def test_none_mentioned_returns_all(self):
        participants = ["Alice", "Bob"]
        text = "Someone spoke."
        result = filter_participants(participants, text)
        assert result == ["Alice", "Bob"]

    def test_empty_transcript(self):
        result = filter_participants(["Alice"], "")
        assert result == ["Alice"]

    def test_empty_participants(self):
        result = filter_participants([], "Hello world")
        assert result == []


class TestFormatList:
    def test_string_items(self):
        result = format_list(["item1", "item2"])
        assert result == ["item1", "item2"]

    def test_dict_items_with_item_key(self):
        result = format_list([{"item": "Do X", "owner": "Alice", "due": "Friday"}])
        assert result == ["Do X - Alice - Friday"]

    def test_dict_items_partial(self):
        result = format_list([{"item": "Do X"}])
        assert result == ["Do X"]

    def test_mixed(self):
        result = format_list(["plain", {"item": "task", "owner": "Bob"}])
        assert len(result) == 2
        assert result[0] == "plain"
        assert result[1] == "task - Bob"

    def test_empty(self):
        assert format_list([]) == []


class TestBuildAnnotatedTranscriptLines:
    def test_with_segments_and_speakers(self):
        analysis = {
            "segment_speakers": [
                {"segment_index": 0, "speaker": "Alice"},
                {"segment_index": 1, "speaker": "Bob"},
            ]
        }
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 6.0, "end": 10.0, "text": "Hi there"},
        ]
        lines = build_annotated_transcript_lines(
            analysis, "Hello Hi there", segments,
            start_time_seconds=None,
            merge_gap_seconds=2, max_merge_seconds=45, max_merge_words=80,
        )
        content_lines = [l for l in lines if l.strip()]
        assert len(content_lines) == 2
        assert "Alice" in content_lines[0]
        assert "Bob" in content_lines[1]

    def test_no_segments_uses_transcript_text(self):
        lines = build_annotated_transcript_lines(
            {}, "Hello world", [],
            start_time_seconds=None,
            merge_gap_seconds=2, max_merge_seconds=45, max_merge_words=80,
        )
        content_lines = [l for l in lines if l.strip()]
        assert len(content_lines) == 1
        assert "Hello world" in content_lines[0]

    def test_empty_analysis_empty_segments(self):
        lines = build_annotated_transcript_lines(
            {}, "", [],
            start_time_seconds=None,
            merge_gap_seconds=2, max_merge_seconds=45, max_merge_words=80,
        )
        assert lines == []


class TestRenderMarkdown:
    MINIMAL_ANALYSIS = {
        "summary": "Test summary",
        "meta": {"title": "Test Meeting", "date": "2026-03-16"},
        "decisions": ["Decision 1"],
        "action_items": [{"item": "Do X", "owner": "Alice"}],
        "annotations": [],
        "sections": [],
        "qa": [],
        "segment_speakers": [],
    }
    META = {
        "audio_file": "test.mp3",
        "duration_seconds": 60.0,
        "transcribe_model": "whisper-1",
        "analysis_model": "gpt-5.2",
    }

    def test_contains_summary(self):
        md = render_markdown(
            self.MINIMAL_ANALYSIS, "Hello", [], self.META,
            None, 2, 45, 80, None,
        )
        assert "Test summary" in md

    def test_contains_metadata(self):
        md = render_markdown(
            self.MINIMAL_ANALYSIS, "Hello", [], self.META,
            None, 2, 45, 80, None,
        )
        assert "test.mp3" in md
        assert "whisper-1" in md

    def test_contains_decisions(self):
        md = render_markdown(
            self.MINIMAL_ANALYSIS, "Hello", [], self.META,
            None, 2, 45, 80, None,
        )
        assert "Decision 1" in md

    def test_contains_action_items(self):
        md = render_markdown(
            self.MINIMAL_ANALYSIS, "Hello", [], self.META,
            None, 2, 45, 80, None,
        )
        assert "Do X" in md
        assert "Alice" in md

    def test_disclaimer(self):
        md = render_markdown(
            self.MINIMAL_ANALYSIS, "Hello", [], self.META,
            None, 2, 45, 80, "DRAFT ONLY",
        )
        assert "DRAFT ONLY" in md

    def test_empty_analysis(self):
        md = render_markdown(
            {}, "Hello", [], self.META,
            None, 2, 45, 80, None,
        )
        assert "# Meeting Notes" in md
        assert "_No summary produced._" in md


class TestRenderOutlineMarkdown:
    MINIMAL_ANALYSIS = {
        "summary": "Test summary",
        "meta": {"title": "Test Meeting", "date": "2026-03-16"},
        "decisions": ["Decision 1"],
        "action_items": [],
        "sections": [{"heading": "Topic 1", "intro": "", "bullets": ["Point A"]}],
        "qa": [],
        "segment_speakers": [],
    }
    META = {
        "audio_file": "test.mp3",
        "duration_seconds": 60.0,
        "transcribe_model": "whisper-1",
        "analysis_model": "gpt-5.2",
    }

    def test_title_with_date(self):
        md = render_outline_markdown(
            self.MINIMAL_ANALYSIS, "Hello", [], self.META,
            None, 2, 45, 80, False, None,
        )
        assert "Test Meeting" in md
        assert "2026-03-16" in md

    def test_includes_section_heading(self):
        md = render_outline_markdown(
            self.MINIMAL_ANALYSIS, "Hello", [], self.META,
            None, 2, 45, 80, False, None,
        )
        assert "Topic 1" in md
        assert "Point A" in md

    def test_metadata_excluded_by_default(self):
        md = render_outline_markdown(
            self.MINIMAL_ANALYSIS, "Hello", [], self.META,
            None, 2, 45, 80, False, None,
        )
        assert "whisper-1" not in md

    def test_metadata_included_when_requested(self):
        md = render_outline_markdown(
            self.MINIMAL_ANALYSIS, "Hello", [], self.META,
            None, 2, 45, 80, True, None,
        )
        assert "whisper-1" in md


class TestRenderTranscriptMarkdown:
    ANALYSIS = {
        "segment_speakers": [
            {"segment_index": 0, "speaker": "Speaker 1"},
            {"segment_index": 1, "speaker": "Speaker 2"},
        ],
        "speakers": [
            {"label": "Speaker 1", "notes": "project lead"},
            {"label": "Speaker 2", "notes": "engineer"},
        ],
    }
    SEGMENTS = [
        {"start": 0.0, "end": 5.0, "text": "Hello everyone"},
        {"start": 6.0, "end": 10.0, "text": "Hi there"},
    ]

    def test_contains_transcript_heading(self):
        md = render_transcript_markdown(
            self.ANALYSIS, "Hello everyone Hi there", self.SEGMENTS,
            None, 2, 45, 80, None,
        )
        assert "# Transcript" in md

    def test_contains_speaker_labels(self):
        md = render_transcript_markdown(
            self.ANALYSIS, "Hello everyone Hi there", self.SEGMENTS,
            None, 2, 45, 80, None,
        )
        assert "Speaker 1" in md
        assert "Speaker 2" in md

    def test_disclaimer(self):
        md = render_transcript_markdown(
            self.ANALYSIS, "Hello", self.SEGMENTS,
            None, 2, 45, 80, "DRAFT",
        )
        assert "DRAFT" in md

    def test_empty_analysis_falls_back_to_speaker(self):
        md = render_transcript_markdown(
            {}, "Hello everyone Hi there", self.SEGMENTS,
            None, 2, 45, 80, None,
        )
        assert "Speaker:" in md

    def test_no_summary_or_decisions_sections(self):
        md = render_transcript_markdown(
            self.ANALYSIS, "Hello", self.SEGMENTS,
            None, 2, 45, 80, None,
        )
        assert "## Summary" not in md
        assert "## Decisions" not in md
        assert "## Action Items" not in md
        assert "## Metadata" not in md
        assert "## Key Points" not in md
