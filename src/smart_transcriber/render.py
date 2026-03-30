"""Markdown rendering for report, outline, and transcript styles.

render_markdown(), render_outline_markdown(), and render_transcript_markdown()
are independent entry points. All share build_annotated_transcript_lines().
Report and outline also share format_list() and append_outline_sections().
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from smart_transcriber.transcribe import merge_display_segments
from smart_transcriber.utils import (
    format_annotation_timestamp,
    format_time_of_day,
    format_timestamp,
    normalize_text_ascii,
)


def filter_participants(participants: List[Any], transcript_text: str) -> List[str]:
    names = [str(p).strip() for p in participants if str(p).strip()]
    if not transcript_text or not names:
        return names
    filtered: List[str] = []
    for name in names:
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        count = len(pattern.findall(transcript_text))
        if count >= 2:
            filtered.append(name)
    return filtered or names


def format_list(items: List[Any]) -> List[str]:
    lines: List[str] = []
    for item in items:
        if isinstance(item, dict):
            value = item.get("item") or item.get("note") or item.get("text")
            owner = item.get("owner")
            due = item.get("due")
            parts = [p for p in [value, owner, due] if p]
            if parts:
                lines.append(normalize_text_ascii(" - ".join(str(p) for p in parts)))
        else:
            lines.append(normalize_text_ascii(str(item)))
    return lines


def build_annotated_transcript_lines(
    analysis: Dict[str, Any],
    transcript_text: str,
    segments: List[Dict[str, Any]],
    start_time_seconds: int | None,
    merge_gap_seconds: int,
    max_merge_seconds: int,
    max_merge_words: int,
) -> List[str]:
    lines: List[str] = []
    if segments:
        speaker_map: Dict[int, str] = {}
        segment_speakers = analysis.get("segment_speakers") or []
        for item in segment_speakers:
            if isinstance(item, dict):
                idx = item.get("segment_index")
                speaker = item.get("speaker")
                if isinstance(idx, int) and speaker:
                    speaker_map[idx] = speaker
        merged_segments = merge_display_segments(
            segments,
            speaker_map,
            merge_gap_seconds,
            max_merge_seconds,
            max_merge_words,
        )
        for seg in merged_segments:
            if start_time_seconds is not None:
                start = format_time_of_day(seg.get("start"), start_time_seconds)
                end = format_time_of_day(seg.get("end"), start_time_seconds)
            else:
                start = format_timestamp(seg.get("start"), precision="s")
                end = format_timestamp(seg.get("end"), precision="s")
            speaker = seg.get("speaker") or "Speaker"
            text = (seg.get("text") or "").strip()
            if text:
                lines.append(f"- [{start} - {end}] {speaker}: {text}")
                lines.append("")
    elif transcript_text:
        lines.append(f"- [--:--:-- - --:--:--] Speaker: {transcript_text.strip()}")
        lines.append("")
    return lines


def append_outline_sections(lines: List[str], analysis: Dict[str, Any]) -> None:
    sections = analysis.get("sections") or []
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = section.get("heading") or section.get("title")
        if not heading:
            continue
        lines.append(f"#### {normalize_text_ascii(heading)}")
        intro = normalize_text_ascii((section.get("intro") or "").strip())
        if intro:
            lines.append(intro)
            lines.append("")
        bullets = section.get("bullets") or []
        for bullet in bullets:
            if bullet:
                lines.append(f"- {normalize_text_ascii(str(bullet))}")
        lines.append("")

    qa = analysis.get("qa") or []
    if qa:
        lines.append("#### Q&A")
        lines.append("")
        for item in qa:
            if not isinstance(item, dict):
                continue
            question = normalize_text_ascii((item.get("question") or "").strip())
            answers = item.get("answers") or []
            if question:
                lines.append(f"**{question}**")
            for answer in answers:
                if answer:
                    lines.append(f"- {normalize_text_ascii(str(answer))}")
            lines.append("")


def render_outline_markdown(
    analysis: Dict[str, Any],
    transcript_text: str,
    segments: List[Dict[str, Any]],
    meta: Dict[str, Any],
    start_time_seconds: int | None,
    merge_gap_seconds: int,
    max_merge_seconds: int,
    max_merge_words: int,
    include_metadata: bool,
    disclaimer: str | None,
) -> str:
    lines: List[str] = []

    meta_block = analysis.get("meta") or {}
    title = meta_block.get("title")
    date = meta_block.get("date")
    if title and date:
        lines.append(f"# {normalize_text_ascii(title)} — {normalize_text_ascii(date)}")
    elif title:
        lines.append(f"# {normalize_text_ascii(title)}")
    elif date:
        lines.append(f"# Meeting Notes — {normalize_text_ascii(date)}")
    else:
        lines.append("# Meeting Notes")
    lines.append("")

    if disclaimer:
        lines.append(f"> {normalize_text_ascii(disclaimer)}")
        lines.append("")

    if include_metadata:
        lines.append("#### Metadata")
        lines.append(f"Audio file: {meta['audio_file']}")
        lines.append(f"Duration: {format_timestamp(meta['duration_seconds'])}")
        lines.append(f"Transcription model: {meta['transcribe_model']}")
        lines.append(f"Analysis model: {meta['analysis_model']}")
        if start_time_seconds is not None:
            lines.append(f"Start time: {format_time_of_day(0.0, start_time_seconds)}")
        if meta_block.get("participants"):
            lines.append("Participants:")
            participants = filter_participants(meta_block["participants"], transcript_text)
            for participant in participants:
                lines.append(f"- {normalize_text_ascii(participant)}")
        if meta_block.get("topics"):
            lines.append("Topics:")
            for topic in meta_block["topics"]:
                lines.append(f"- {normalize_text_ascii(topic)}")
        lines.append("")

    append_outline_sections(lines, analysis)

    action_items = analysis.get("action_items") or []
    if action_items:
        lines.append("#### Action Items")
        for item in format_list(action_items):
            lines.append(f"- {item}")
        lines.append("")

    decisions = analysis.get("decisions") or []
    if decisions:
        lines.append("#### Decisions")
        for item in format_list(decisions):
            lines.append(f"- {item}")
        lines.append("")

    lines.append("#### Annotated Transcript")
    lines.extend(
        build_annotated_transcript_lines(
            analysis,
            transcript_text,
            segments,
            start_time_seconds,
            merge_gap_seconds,
            max_merge_seconds,
            max_merge_words,
        )
    )
    lines.append("")

    return "\n".join(lines)


def render_markdown(
    analysis: Dict[str, Any],
    transcript_text: str,
    segments: List[Dict[str, Any]],
    meta: Dict[str, Any],
    start_time_seconds: int | None,
    merge_gap_seconds: int,
    max_merge_seconds: int,
    max_merge_words: int,
    disclaimer: str | None,
) -> str:
    lines: List[str] = []
    lines.append("# Meeting Notes")
    lines.append("")

    if disclaimer:
        lines.append(f"> {normalize_text_ascii(disclaimer)}")
        lines.append("")

    lines.append("## Summary")
    summary = (analysis.get("summary") or "").strip()
    summary = normalize_text_ascii(summary)
    lines.append(summary if summary else "_No summary produced._")
    lines.append("")

    lines.append("## Metadata")
    lines.append(f"Audio file: {meta['audio_file']}")
    lines.append(f"Duration: {format_timestamp(meta['duration_seconds'])}")
    lines.append(f"Transcription model: {meta['transcribe_model']}")
    lines.append(f"Analysis model: {meta['analysis_model']}")
    if start_time_seconds is not None:
        lines.append(f"Start time: {format_time_of_day(0.0, start_time_seconds)}")
    meta_block = analysis.get("meta") or {}
    if meta_block.get("title"):
        lines.append(f"Title: {normalize_text_ascii(meta_block['title'])}")
    if meta_block.get("date"):
        lines.append(f"Date: {normalize_text_ascii(meta_block['date'])}")
    if meta_block.get("participants"):
        lines.append("Participants:")
        participants = filter_participants(meta_block["participants"], transcript_text)
        for participant in participants:
            lines.append(f"- {normalize_text_ascii(participant)}")
    if meta_block.get("topics"):
        lines.append("Topics:")
        for topic in meta_block["topics"]:
            lines.append(f"- {normalize_text_ascii(topic)}")
    lines.append("")

    lines.append("## Decisions")
    decisions = analysis.get("decisions") or []
    if decisions:
        for item in format_list(decisions):
            lines.append(f"- {item}")
    else:
        lines.append("- _None noted._")
    lines.append("")

    lines.append("## Action Items")
    action_items = analysis.get("action_items") or []
    if action_items:
        for item in format_list(action_items):
            lines.append(f"- {item}")
    else:
        lines.append("- _None noted._")
    lines.append("")

    annotations = analysis.get("annotations") or []
    lines.append("## Key Points")
    if annotations:
        for note in annotations:
            if isinstance(note, dict):
                timestamp = note.get("timestamp")
                speaker = note.get("speaker")
                text = normalize_text_ascii(note.get("note") or "")
                prefix = ""
                if timestamp:
                    prefix += f"[{format_annotation_timestamp(timestamp, start_time_seconds)}] "
                if speaker:
                    prefix += f"{normalize_text_ascii(speaker)}: "
                lines.append(f"- {prefix}{text}")
            else:
                lines.append(f"- {normalize_text_ascii(str(note))}")
    else:
        lines.append("- _None noted._")
    lines.append("")

    if analysis.get("sections") or analysis.get("qa"):
        lines.append("## Notes")
        append_outline_sections(lines, analysis)

    lines.append("## Annotated Transcript")
    lines.extend(
        build_annotated_transcript_lines(
            analysis,
            transcript_text,
            segments,
            start_time_seconds,
            merge_gap_seconds,
            max_merge_seconds,
            max_merge_words,
        )
    )
    lines.append("")

    return "\n".join(lines)


def render_transcript_markdown(
    analysis: Dict[str, Any],
    transcript_text: str,
    segments: List[Dict[str, Any]],
    start_time_seconds: int | None,
    merge_gap_seconds: int,
    max_merge_seconds: int,
    max_merge_words: int,
    disclaimer: str | None,
) -> str:
    lines: List[str] = []
    lines.append("# Transcript")
    lines.append("")

    if disclaimer:
        lines.append(f"> {normalize_text_ascii(disclaimer)}")
        lines.append("")

    lines.extend(
        build_annotated_transcript_lines(
            analysis,
            transcript_text,
            segments,
            start_time_seconds,
            merge_gap_seconds,
            max_merge_seconds,
            max_merge_words,
        )
    )
    lines.append("")

    return "\n".join(lines)
