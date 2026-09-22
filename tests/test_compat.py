import json

from smart_transcriber import compat


def test_legacy_dryrun_uses_valid_api_format_without_writes(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.wav"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(compat, "probe_audio", lambda path: {"duration": 30, "channels": 2})
    output = tmp_path / "output.json"
    assert compat.main([str(source), "--model", "gpt-4o-transcribe-diarize", "--response-format", "diarized_json", "--out", str(output), "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["response_format"] == "diarized_json"
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [source]


def test_legacy_text_export_still_requests_json(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.wav"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(compat, "probe_audio", lambda path: {"duration": 30, "channels": 2})
    assert compat.main([str(source), "--response-format", "text", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["response_format"] == "json"
    assert result["export_format"] == "text"


def test_legacy_diarization_cannot_silently_drop_prompt(tmp_path):
    assert compat.main([str(tmp_path / "input.wav"), "--model", "gpt-4o-transcribe-diarize", "--prompt", "context"]) == 1
