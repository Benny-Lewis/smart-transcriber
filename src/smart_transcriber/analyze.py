"""Transcript analysis via OpenAI API — prompt construction, schema contract."""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List, TypedDict

from openai import OpenAI

ANALYSIS_PROMPT_VERSION = 3


class AnalysisResult(TypedDict, total=False):
    """Expected shape of the LLM's JSON analysis response."""

    summary: str
    meta: Dict[str, Any]
    sections: List[Dict[str, Any]]
    qa: List[Dict[str, Any]]
    speakers: List[Dict[str, Any]]
    decisions: List[str]
    action_items: List[Dict[str, Any]]
    annotations: List[Dict[str, Any]]
    segment_speakers: List[Dict[str, Any]]


def build_analysis_prompt(
    payload: Dict[str, Any],
    transcript_only: bool = False,
) -> str:
    if transcript_only:
        return (
            "You are a careful meeting analyst. Use only the provided transcript data. "
            "Do not invent facts or speakers. If unsure, use null or empty lists. "
            "Return a JSON object with these keys:\n"
            "speakers: list of {label, notes}\n"
            "segment_speakers: list of {segment_index, speaker}\n\n"
            "Keep speaker notes short. "
            "Preserve input speaker labels exactly. You receive text, not audio: "
            "never infer voice identity or invent assignments. If a segment has no "
            "input speaker, leave its speaker unknown.\n\n"
            "For segment_speakers, use the segment_index values from input segments and "
            "only assign speakers; do not rewrite or paraphrase the text.\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=True)}"
        )
    return (
        "You are a careful meeting analyst. Use only the provided transcript data. "
        "Do not invent facts, speakers, or decisions. If unsure, use null or empty lists. "
        "Return a JSON object with these keys:\n"
        "summary: string\n"
        "meta: {title, date, participants, topics}\n"
        "sections: list of {heading, intro, bullets}\n"
        "qa: list of {question, answers}\n"
        "speakers: list of {label, notes}\n"
        "decisions: list of strings\n"
        "action_items: list of {item, owner, due}\n"
        "annotations: list of {timestamp, speaker, note}\n"
        "segment_speakers: list of {segment_index, speaker}\n\n"
        "Keep speaker notes short. "
        "Preserve input speaker labels exactly. You receive text, not audio: "
        "never infer voice identity or invent assignments. If a segment has no "
        "input speaker, leave its speaker unknown. Unresolved wording differences "
        "are uncertainty, not established facts; do not silently choose their alternatives.\n\n"
        "For sections, create high-level headings (e.g., 'Lift and Shift Updates', "
        "'Plans for a Restart', 'Severance', 'Closing') with concise bullets. "
        "Use intro for short lead-in sentences if needed.\n"
        "For qa, list key questions with short bullet answers; avoid nesting.\n\n"
        "For segment_speakers, use the segment_index values from input segments and "
        "only assign speakers; do not rewrite or paraphrase the text.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def validate_analysis(result: Any) -> None:
    """Validate fresh, cached and imported data before rendering or caching it."""
    if not isinstance(result, dict):
        raise ValueError("Analysis must be a JSON object.")

    def optional_string(obj: dict, field: str) -> None:
        if obj.get(field) is not None and not isinstance(obj[field], str):
            raise ValueError(f"Analysis {field} must be a string or null.")

    def optional_list(obj: dict, field: str) -> list:
        value = obj.get(field)
        if value is not None and not isinstance(value, list):
            raise ValueError(f"Analysis {field} must be a list or null.")
        return value or []

    optional_string(result, "summary")
    meta = result.get("meta")
    if meta is not None and not isinstance(meta, dict):
        raise ValueError("Analysis meta must be an object or null.")
    for key in ("title", "date"):
        optional_string(meta or {}, key)
    optional_list(meta or {}, "participants")
    if any(not isinstance(t, str) for t in optional_list(meta or {}, "topics")):
        raise ValueError("Analysis topics must contain strings.")
    for field in ("sections", "qa", "speakers", "segment_speakers"):
        for item in optional_list(result, field):
            if not isinstance(item, dict):
                raise ValueError(f"Analysis {field} must contain objects.")
            for key in ("heading", "title", "intro", "question", "label", "notes", "speaker"):
                optional_string(item, key)
            for key in ("bullets", "answers"):
                optional_list(item, key)
    for field in ("decisions", "action_items", "annotations"):
        optional_list(result, field)
    for item in result.get("annotations") or []:
        if isinstance(item, dict):
            optional_string(item, "speaker")
            optional_string(item, "note")
            value = item.get("timestamp")
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("Analysis timestamp must be finite.")
            if isinstance(value, str):
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    raise ValueError("Analysis timestamp must be finite.")


def preserve_analysis_speakers(result: dict, payload: dict) -> AnalysisResult:
    """A text analysis may describe input speakers, never establish their identities."""
    assignments = [{"segment_index": s["index"], "speaker": s["speaker"]}
                   for s in payload.get("segments", []) if s.get("speaker")]
    result["segment_speakers"] = assignments
    labels = {a["speaker"] for a in assignments}
    result["speakers"] = [s for s in (result.get("speakers") or []) if s.get("label") in labels]
    for annotation in result.get("annotations") or []:
        if isinstance(annotation, dict) and annotation.get("speaker") not in labels:
            annotation["speaker"] = None
    return result


def analyze_transcript(
    client: OpenAI,
    model: str,
    payload: Dict[str, Any],
    transcript_only: bool = False,
    save_response: Callable[[dict], None] | None = None,
) -> AnalysisResult:
    prompt = build_analysis_prompt(payload, transcript_only=transcript_only)
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    if save_response is not None:
        save_response(response.model_dump())
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("Analysis returned no content; any original response has been retained.")
    content = response.choices[0].message.content
    result = json.loads(content)
    validate_analysis(result)
    return preserve_analysis_speakers(result, payload)
