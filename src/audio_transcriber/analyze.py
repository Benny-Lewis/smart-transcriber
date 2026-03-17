"""Transcript analysis via OpenAI API — prompt construction, schema contract."""

from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

from openai import OpenAI


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


def build_analysis_prompt(payload: Dict[str, Any]) -> str:
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
        "Prefer generic speaker labels like 'Speaker 1' unless the transcript "
        "explicitly identifies names. Keep notes short.\n"
        "If there are many speakers (e.g., town hall), assign unique labels "
        "such as 'Speaker 1', 'Speaker 2', or 'Audience 1', 'Audience 2', etc. "
        "Reuse labels consistently for the same voice across segments.\n\n"
        "For sections, create high-level headings (e.g., 'Lift and Shift Updates', "
        "'Plans for a Restart', 'Severance', 'Closing') with concise bullets. "
        "Use intro for short lead-in sentences if needed.\n"
        "For qa, list key questions with short bullet answers; avoid nesting.\n\n"
        "For segment_speakers, use the segment_index values from input segments and "
        "only assign speakers; do not rewrite or paraphrase the text.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def analyze_transcript(
    client: OpenAI,
    model: str,
    payload: Dict[str, Any],
) -> AnalysisResult:
    prompt = build_analysis_prompt(payload)
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)
