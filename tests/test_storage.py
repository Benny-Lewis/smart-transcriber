"""Checkpoint integrity under an interrupted local write."""

import json

import pytest

from smart_transcriber import storage


def test_failed_atomic_replace_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "transcript.json"
    storage.atomic_json(path, {"text": "completed"})
    monkeypatch.setattr(storage.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("simulated interruption")))
    with pytest.raises(OSError):
        storage.atomic_json(path, {"text": "partial"})
    assert json.loads(path.read_text()) == {"text": "completed"}
    assert list(tmp_path.iterdir()) == [path]


def test_index_failure_still_leaves_paid_raw_response(tmp_path, monkeypatch):
    real_atomic = storage.atomic_json

    def fail_index(path, value):
        if path.parent.name == "cache":
            raise OSError("simulated index failure")
        real_atomic(path, value)

    monkeypatch.setattr(storage, "atomic_json", fail_index)
    with pytest.raises(OSError):
        storage.StageStore(tmp_path).save("wording", {"model": "test"}, {"text": "Paid result"})
    paths = list((tmp_path / "raw").glob("*.json"))
    assert len(paths) == 1
    assert json.loads(paths[0].read_text()) == {"text": "Paid result"}
    recovered = storage.StageStore(tmp_path).load("wording", {"model": "test"})
    assert recovered[0] == {"text": "Paid result"}
    assert storage.StageStore(tmp_path).load("wording", {"model": "different"}) is None
