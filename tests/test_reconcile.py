import copy

import pytest

from smart_transcriber.reconcile import reconcile, render_review


def transcript(text, speaker="A", index="one", start=0, end=30):
    return {"text": text, "segments": [{"id": index, "speaker": speaker, "start": start, "end": end, "text": text}]}


def test_internal_technical_correction_preserves_raw_speaker_and_time():
    original = transcript("We discussed the cube control system with Jane yesterday.")
    before = copy.deepcopy(original)
    wording = {"text": "We discussed the kubectl system with Jane yesterday."}
    result = reconcile(original, wording)
    assert result["segments"][0]["text"] == wording["text"]
    assert result["segments"][0]["original_text"] == original["text"]
    assert result["segments"][0]["speaker"] == "A"
    assert result["segments"][0]["start"] == 0
    assert result["segments"][0]["end"] == 30
    assert result["reconciliation"]["accepted_count"] == 1
    assert original == before
    assert "hypotheses" in render_review(result)


@pytest.mark.parametrize("before,after,reason", [
    ("42", "43", "numeric_or_version_change"),
    ("v1.2", "v1.3", "numeric_or_version_change"),
    ("two", "three", "numeric_or_version_change"),
    ("twenty-four", "twenty-five", "numeric_or_version_change"),
    ("not", "now", "negation_change"),
    ("isn't", "is", "negation_change"),
    ("haven’t", "have", "negation_change"),
    ("alpha beta gamma delta epsilon", "omega", "large_rewrite"),
])
def test_consequential_or_large_edits_require_review(before, after, reason):
    source = transcript(f"We carefully discussed {before} during the planning meeting.")
    result = reconcile(source, {"text": f"We carefully discussed {after} during the planning meeting."})
    assert result["segments"] == source["segments"]
    assert result["reconciliation"]["changes"][0]["reason"] == reason


@pytest.mark.parametrize("before,after", [("C", "C++"), ("net", ".NET"), ("sea sharp", "C#"), ("Kubernetes", "Kübernetes")])
def test_technical_punctuation_and_unicode_are_retained(before, after):
    source = transcript(f"Today we discussed {before} in the engineering meeting.")
    result = reconcile(source, {"text": f"Today we discussed {after} in the engineering meeting."})
    assert after in result["segments"][0]["text"]
    assert result["reconciliation"]["accepted_count"] == 1


def test_cannot_borrow_anchor_from_next_speaker():
    source = transcript("Today we discussed cube control")
    source["segments"] += transcript("in the engineering meeting.", "B", "two", 30, 40)["segments"]
    result = reconcile(source, {"text": "Today we discussed kubectl in the engineering meeting."})
    assert result["segments"] == source["segments"]
    assert result["reconciliation"]["review_count"] == 1


def test_repeated_anchors_cannot_authorize_substitution():
    source = transcript("a b c d e f g h wrong i j k l m n o p")
    source["segments"] += transcript("a b c d e f g h correct i j k l m n o p", "B", "two", 31, 60)["segments"]
    result = reconcile(source, {"text": "a b c d e f g h right i j k l m n o p a b c d e f g h correct i j k l m n o p"})
    assert result["reconciliation"]["accepted_count"] == 0


@pytest.mark.parametrize("wording", ["begin alpha beta gamma delta", "alpha beta gamma delta end", "alpha gamma delta", ""])
def test_insertions_deletions_and_boundaries_are_reviewed(wording):
    source = transcript("alpha beta gamma delta")
    result = reconcile(source, {"text": wording})
    assert result["segments"] == source["segments"]
    assert result["reconciliation"]["review_count"] > 0


def test_overlapping_audio_is_not_rewritten():
    source = transcript("We discussed the cube control system with Jane yesterday.")
    source["segments"] += transcript("Yes that makes sense.", "B", "two", 20, 35)["segments"]
    result = reconcile(source, {"text": "We discussed the kubectl system with Jane yesterday. Yes that makes sense."})
    assert result["reconciliation"]["accepted_count"] == 0
    assert result["reconciliation"]["changes"][0]["reason"] == "overlapping_audio"


def test_two_adjacent_changes_do_not_shift_character_offsets():
    text = "First we discussed alhpa with the first team then covered beeta with the second team."
    target = text.replace("alhpa", "alpha").replace("beeta", "beta")
    result = reconcile(transcript(text), {"text": target})
    assert result["text"] == target
    assert result["reconciliation"]["accepted_count"] == 2


def test_repetitive_input_has_a_bounded_fallback():
    source = transcript("filler " * 2000)
    result = reconcile(source, {"text": "filler " * 2000})
    assert result["reconciliation"]["changes"][0]["reason"] == "alignment_work_limit"
    assert result["segments"] == source["segments"]


def test_missing_utterance_is_not_silently_attributed():
    source = transcript("We finished the discussion.")
    result = reconcile(source, {"text": "We finished the discussion. Another person answered yes."})
    assert result["reconciliation"]["changes"][0]["reason"] == "insertion"
    assert result["segments"] == source["segments"]
