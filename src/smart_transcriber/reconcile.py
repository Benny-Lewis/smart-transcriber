"""Conservative, bounded wording reconciliation; never infers speakers or timing."""

from __future__ import annotations

import copy
import re
from collections import Counter
from difflib import SequenceMatcher

VERSION = 1
MAX_TOKENS = 20_000
MAX_MATCH_PAIRS = 2_000_000
NEGATIONS = {"no", "not", "never", "neither", "nor", "without", "cannot", "can't", "won't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "couldn't", "wouldn't", "shouldn't", "hasn't", "haven't", "hadn't", "mustn't", "needn't", "shan't", "ain't", "none", "nobody", "nothing", "nowhere"}
NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion trillion first second third fourth fifth sixth seventh eighth ninth tenth half quarter twice once".split())


def _tokens(text: str) -> list[tuple[str, int, int]]:
    # Whitespace tokenization retains identifiers, punctuation, case and Unicode.
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _negation(words: list[str]) -> bool:
    return any(w.lower().replace("’", "'").strip(".,;:!?()[]\"") in NEGATIONS for w in words)


def _numeric(text: str) -> bool:
    return any(c.isdigit() for c in text) or bool(NUMBER_WORDS.intersection(re.findall(r"[a-z]+", text.lower())))


def reconcile(diarized: dict, wording: dict) -> dict:
    result = copy.deepcopy(diarized)
    segments = result.get("segments") or []
    original = "\n".join(s.get("text", "") for s in segments)
    companion = wording.get("text", "") or ""
    source_tokens = []
    owner = []
    local_spans = []
    global_offset = 0
    for i, segment in enumerate(segments):
        text = segment.get("text", "")
        for word, start, end in _tokens(text):
            source_tokens.append((word, start + global_offset, end + global_offset))
            local_spans.append((start, end))
            owner.append(i)
        global_offset += len(text) + 1
    target_tokens = _tokens(companion)
    left = [t[0] for t in source_tokens]
    right = [t[0] for t in target_tokens]
    changes: list[dict] = []
    report = {"version": VERSION, "status": "complete", "accepted_count": 0, "review_count": 0, "changes": changes}
    result["reconciliation"] = report
    result["wording_text"] = companion

    def fallback(reason: str) -> dict:
        report.update(status="review_required", review_count=1)
        changes.append({"disposition": "review", "reason": reason,
                        "before": original or result.get("text", ""), "after": companion, "segment_ids": []})
        return result

    if not segments:
        return fallback("missing_diarized_segments") if companion or result.get("text") else result
    if not right and left:
        return fallback("empty_wording_response")
    if max(len(left), len(right)) > MAX_TOKENS:
        return fallback("alignment_token_limit")
    counts_left, counts_right = Counter(left), Counter(right)
    if sum(n * counts_right[word] for word, n in counts_left.items()) > MAX_MATCH_PAIRS:
        return fallback("alignment_work_limit")
    operations = SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
    ngram_counts: dict[int, tuple[Counter, Counter]] = {}

    def unique_anchor(start: int, stop: int, target_start: int, target_stop: int, from_end: bool, segment_index: int):
        for length in range(3, min(8, stop - start, target_stop - target_start) + 1):
            a = stop - length if from_end else start
            b = target_stop - length if from_end else target_start
            if any(v != segment_index for v in owner[a:a + length]):
                continue
            anchor = tuple(left[a:a + length])
            if anchor != tuple(right[b:b + length]):
                continue
            if length not in ngram_counts:
                ngram_counts[length] = (
                    Counter(tuple(left[k:k + length]) for k in range(len(left) - length + 1)),
                    Counter(tuple(right[k:k + length]) for k in range(len(right) - length + 1)),
                )
            c1, c2 = ngram_counts[length]
            if c1[anchor] == 1 and c2[anchor] == 1:
                return {"source_tokens": [a, a + length], "wording_tokens": [b, b + length], "text": " ".join(anchor)}
        return None

    replacements: dict[int, list[tuple[int, int, str]]] = {}
    for k, (tag, a1, a2, b1, b2) in enumerate(operations):
        if tag == "equal":
            continue
        ids = sorted(set(owner[a1:a2]))
        before = original[source_tokens[a1][1]:source_tokens[a2 - 1][2]] if a2 > a1 else ""
        after = companion[target_tokens[b1][1]:target_tokens[b2 - 1][2]] if b2 > b1 else ""
        entry = {"id": len(changes) + 1, "disposition": "review", "reason": "",
                 "segment_ids": [segments[i].get("id", str(i)) for i in ids],
                 "source_tokens": [a1, a2], "wording_tokens": [b1, b2],
                 "source_characters": [source_tokens[a1][1], source_tokens[a2 - 1][2]] if a2 > a1 else None,
                 "wording_characters": [target_tokens[b1][1], target_tokens[b2 - 1][2]] if b2 > b1 else None,
                 "before": before, "after": after}
        reason = ""
        if tag != "replace":
            reason = "insertion" if tag == "insert" else "deletion"
        elif len(ids) != 1:
            reason = "crosses_segment"
        elif not (1 <= a2 - a1 <= 4 and 1 <= b2 - b1 <= 4):
            reason = "large_rewrite"
        elif _numeric(before + " " + after):
            reason = "numeric_or_version_change"
        elif _negation(left[a1:a2]) or _negation(right[b1:b2]):
            reason = "negation_change"
        elif k == 0 or k + 1 == len(operations):
            reason = "missing_anchor"
        else:
            segment_index = ids[0]
            segment = segments[segment_index]
            if not segment.get("speaker"):
                reason = "unknown_speaker"
            else:
                start, end = segment.get("start"), segment.get("end")
                overlap = start is not None and end is not None and any(
                    j != segment_index and s.get("start") is not None and s.get("end") is not None
                    and start < s["end"] and s["start"] < end for j, s in enumerate(segments)
                )
                if overlap:
                    reason = "overlapping_audio"
            prev, following = operations[k - 1], operations[k + 1]
            if not reason and (prev[0] != "equal" or following[0] != "equal"):
                reason = "missing_anchor"
            if not reason:
                anchor_left = unique_anchor(*prev[1:], True, segment_index)
                anchor_right = unique_anchor(*following[1:], False, segment_index)
                if not anchor_left or not anchor_right:
                    reason = "missing_or_nonunique_within_segment_anchor"
                else:
                    entry.update(disposition="accepted", anchors=[anchor_left, anchor_right])
                    begin, finish = local_spans[a1][0], local_spans[a2 - 1][1]
                    replacements.setdefault(segment_index, []).append((begin, finish, after))
                    entry["segment_characters"] = [begin, finish]
        entry["reason"] = reason or "small_anchored_substitution"
        changes.append(entry)

    for index, edits in replacements.items():
        segment = segments[index]
        segment["original_text"] = segment["text"]
        for begin, finish, replacement in reversed(edits):
            segment["text"] = segment["text"][:begin] + replacement + segment["text"][finish:]
        segment["wording_source"] = "reconciled"
    result["text"] = "\n".join(s.get("text", "") for s in segments)
    report["accepted_count"] = sum(c["disposition"] == "accepted" for c in changes)
    report["review_count"] = len(changes) - report["accepted_count"]
    return result


def render_review(transcript: dict) -> str:
    report = transcript.get("reconciliation", {})
    lines = ["# Transcript wording review", "", "Both original API responses are retained in the job's raw directory.",
             "Speaker labels and segment timing come from the audio diarization pass. Accepted wording changes are hypotheses, not independently verified corrections.", "",
             f"Accepted substitutions: {report.get('accepted_count', 0)}. Differences requiring review: {report.get('review_count', 0)}.", ""]
    for i, change in enumerate(report.get("changes", []), 1):
        lines.extend([f"## {i}. {change['disposition']} — {change['reason']}", "",
                      "Segments: " + ", ".join(str(s) for s in change.get("segment_ids", [])), ""])
        for label, field in [("Diarized wording", "before"), ("Wording pass", "after")]:
            excerpt = change.get(field, "")[:600].replace("`", "\\`")
            lines.extend([f"{label}:", "", "> " + excerpt.replace("\n", "\n> "), ""])
    return "\n".join(lines) + "\n"
