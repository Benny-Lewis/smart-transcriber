"""Tests for smart_transcriber.utils."""

import pytest

from smart_transcriber.utils import (
    DEFAULT_AUDIO_EXTS,
    DEFAULT_CHUNK_SECONDS,
    MAX_UPLOAD_BYTES,
    format_annotation_timestamp,
    format_time_of_day,
    format_timestamp,
    normalize_text_ascii,
    parse_time_of_day,
)


class TestFormatTimestamp:
    def test_none_ms_precision(self):
        assert format_timestamp(None) == "00:00:00.000"

    def test_none_s_precision(self):
        assert format_timestamp(None, precision="s") == "00:00:00"

    def test_zero(self):
        assert format_timestamp(0.0) == "00:00:00.000"

    def test_seconds_only(self):
        assert format_timestamp(5.0, precision="s") == "00:00:05"

    def test_minutes_and_seconds(self):
        assert format_timestamp(125.0, precision="s") == "00:02:05"

    def test_hours(self):
        assert format_timestamp(3661.5, precision="s") == "01:01:02"

    def test_ms_precision(self):
        assert format_timestamp(1.234) == "00:00:01.234"

    def test_ms_rounding(self):
        # Python uses banker's rounding: round(1234.5) == 1234 (round half to even)
        assert format_timestamp(1.2345) == "00:00:01.234"

    def test_large_value(self):
        assert format_timestamp(86400.0, precision="s") == "24:00:00"


class TestNormalizeTextAscii:
    def test_no_replacement(self):
        assert normalize_text_ascii("hello world") == "hello world"

    def test_ellipsis_replacement(self):
        # Verify the replacement dict works by testing a known mapping.
        assert normalize_text_ascii("\u00c2") == ""

    def test_passthrough_clean_text(self):
        assert normalize_text_ascii("Hello, world!") == "Hello, world!"

    def test_empty_string(self):
        assert normalize_text_ascii("") == ""


class TestParseTimeOfDay:
    def test_hh_mm(self):
        assert parse_time_of_day("09:30") == 9 * 3600 + 30 * 60

    def test_hh_mm_ss(self):
        assert parse_time_of_day("09:30:15") == 9 * 3600 + 30 * 60 + 15

    def test_midnight(self):
        assert parse_time_of_day("00:00") == 0

    def test_end_of_day(self):
        assert parse_time_of_day("23:59:59") == 23 * 3600 + 59 * 60 + 59

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="HH:MM"):
            parse_time_of_day("9")

    def test_invalid_hour(self):
        with pytest.raises(ValueError, match="valid 24h"):
            parse_time_of_day("25:00")

    def test_invalid_minute(self):
        with pytest.raises(ValueError, match="valid 24h"):
            parse_time_of_day("12:60")


class TestFormatTimeOfDay:
    def test_none_offset(self):
        assert format_time_of_day(None, 0) == "--:--:--"

    def test_zero_offset_zero_start(self):
        assert format_time_of_day(0.0, 0) == "00:00:00"

    def test_offset_with_start(self):
        # start at 09:30:00 (34200s), offset 90s => 09:31:30
        assert format_time_of_day(90.0, 34200) == "09:31:30"

    def test_day_rollover(self):
        # start at 23:00:00, offset 7200s (2h) => 01:00:00 (+1d)
        result = format_time_of_day(7200.0, 23 * 3600)
        assert "+1d" in result
        assert "01:00:00" in result


class TestFormatAnnotationTimestamp:
    def test_numeric_without_start_time(self):
        result = format_annotation_timestamp(90.0, None)
        assert result == "00:01:30"

    def test_numeric_with_start_time(self):
        result = format_annotation_timestamp(90.0, 34200)
        assert result == "09:31:30"

    def test_string_numeric(self):
        result = format_annotation_timestamp("90.0", None)
        assert result == "00:01:30"

    def test_string_non_numeric(self):
        result = format_annotation_timestamp("early in meeting", None)
        assert result == "early in meeting"

    def test_integer(self):
        result = format_annotation_timestamp(60, None)
        assert result == "00:01:00"


class TestConstants:
    def test_max_upload_bytes(self):
        assert MAX_UPLOAD_BYTES == 26_214_400

    def test_default_chunk_seconds(self):
        assert DEFAULT_CHUNK_SECONDS == 600

    def test_default_audio_exts_contains_common(self):
        assert ".mp3" in DEFAULT_AUDIO_EXTS
        assert ".wav" in DEFAULT_AUDIO_EXTS
        assert ".m4a" in DEFAULT_AUDIO_EXTS
