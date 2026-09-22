import copy
import json
from types import SimpleNamespace

import pytest

from smart_transcriber import cli, pipeline
from smart_transcriber.audio import AudioChunk
from smart_transcriber.pipeline import TranscriptionOptions, run_transcription


SOURCE = "We discussed the cube control system with Jane yesterday."
WORDING = "We discussed the kubectl system with Jane yesterday."


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.fail_wording = False
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append({k: v for k, v in kwargs.items() if k != "file"})
        if kwargs["model"] == pipeline.DIARIZATION_MODEL:
            return {"text": SOURCE, "duration": 60, "segments": [
                {"id": "s1", "start": 0, "end": 30, "speaker": "A", "text": SOURCE}], "usage": {"seconds": 60}}
        if self.fail_wording:
            raise RuntimeError("simulated interrupted request")
        return {"text": WORDING, "languages": [{"code": "en"}], "usage": {"seconds": 60}}


@pytest.fixture
def job(tmp_path, monkeypatch):
    audio = tmp_path / "recording.wav"
    audio.write_bytes(b"synthetic fixture; media preparation mocked")
    monkeypatch.setattr(pipeline, "prepare_audio", lambda path, *args: (
        [AudioChunk(path, 0.0, 60.0)], {"duration": 60.0, "channels": 2}))
    return audio, tmp_path / "work", FakeAPI()


def run(job, **kwargs):
    audio, work, client = job
    return run_transcription(client, audio, work, TranscriptionOptions(**kwargs), progress=lambda _: None)


def test_two_pass_payload_and_completed_reuse(job):
    result = run(job, glossary=["kubectl"], prompt="Engineering interview")
    client = job[2]
    assert len(client.calls) == 2
    assert client.calls[0]["response_format"] == "diarized_json"
    assert client.calls[0]["chunking_strategy"] == "auto"
    assert "prompt" not in client.calls[0]
    assert client.calls[1]["extra_body"] == {"keywords": ["kubectl"], "languages": ["en"]}
    assert "language" not in client.calls[1]
    assert result["segments"][0]["speaker"] == "A"
    assert result["text"] == WORDING
    for paths in result["meta"]["raw_responses"].values():
        assert len(paths) == 1
    raw = json.loads(__import__("pathlib").Path(result["meta"]["raw_responses"]["diarization"][0]).read_text())
    assert raw["text"] == SOURCE
    assert "original_text" not in raw["segments"][0]
    run(job, glossary=["kubectl"], prompt="Engineering interview")
    assert len(client.calls) == 2


def test_glossary_change_reruns_only_wording(job):
    run(job)
    run(job, glossary=["kubectl"])
    assert [c["model"] for c in job[2].calls] == [pipeline.DIARIZATION_MODEL, pipeline.WORDING_MODEL, pipeline.WORDING_MODEL]
    assert len(list(job[1].rglob("derived/raw/reconciliation-*.json"))) == 2


def test_reference_change_reruns_only_diarization(job):
    refs = [{"name": "Jane", "sha256": "first", "data_url": "data:audio/wav;base64,AA=="}]
    run(job, known_speakers=refs)
    refs[0]["sha256"] = "second"
    run(job, known_speakers=refs)
    assert job[2].calls[-1]["model"] == pipeline.DIARIZATION_MODEL
    assert len(job[2].calls) == 3
    for path in job[1].rglob("cache/*.json"):
        assert "base64" not in path.read_text()


def test_failed_wording_resumes_without_repeating_diarization(job):
    job[2].fail_wording = True
    with pytest.raises(RuntimeError):
        run(job)
    assert len(list(job[1].rglob("raw/*.json"))) == 1
    assert len(list(job[1].rglob("diarization.json"))) == 1
    job[2].fail_wording = False
    result = run(job)
    assert result["text"] == WORDING
    assert len(job[2].calls) == 3


def test_force_retains_prior_raw_responses(job):
    run(job)
    run(job, force=True)
    assert len(job[2].calls) == 4
    assert len(list(job[1].rglob("raw/diarization-*.json"))) == 2
    assert len(list(job[1].rglob("raw/wording-*.json"))) == 2


def test_source_change_cannot_reuse_paid_results(job):
    first = run(job)
    job[0].write_bytes(b"a different recording")
    second = run(job)
    assert first["meta"]["source_sha256"] != second["meta"]["source_sha256"]
    assert len(job[2].calls) == 4


def test_corrupt_cache_reruns_only_damaged_stage(job):
    result = run(job)
    from pathlib import Path
    Path(result["meta"]["raw_responses"]["wording"][0]).write_text("broken")
    run(job)
    assert len(job[2].calls) == 3
    assert job[2].calls[-1]["model"] == pipeline.WORDING_MODEL


def test_explicit_chunk_offsets_and_scoped_unknown_speakers(job, monkeypatch):
    monkeypatch.setattr(pipeline, "prepare_audio", lambda path, *args: (
        [AudioChunk(path, 0.0, 600), AudioChunk(path, 600.0, 60)], {"duration": 660, "channels": 2}))
    result = run(job, mode="diarized")
    assert result["segments"][1]["start"] == 600
    assert [s["speaker"] for s in result["segments"]] == ["chunk_1:A", "chunk_2:A"]


@pytest.mark.parametrize("kwargs", [{"mode": "diarized", "glossary": ["test"]},
    {"mode": "diarized", "prompt": "context"}, {"chunk_seconds": 0},
    {"mode": "text", "known_speakers": [{"name": "Jane"}]}, {"transcribe_model": "unknown-model"}])
def test_bad_capabilities_fail_before_api(job, kwargs):
    with pytest.raises(ValueError):
        run(job, **kwargs)
    assert job[2].calls == []


def test_cli_transcript_without_analysis_and_cached_without_key(job, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(cli, "OpenAI", lambda **kwargs: job[2])
    output = tmp_path / "transcript.md"
    argv = [str(job[0]), "--out", str(output), "--work-dir", str(job[1]), "--style", "transcript", "--no-analysis"]
    assert cli.main(argv) == 0
    assert "A:" in output.read_text()
    monkeypatch.delenv("OPENAI_API_KEY")
    assert cli.main(argv) == 0
    assert len(job[2].calls) == 2


def test_analysis_failure_keeps_transcript_and_resume_reuses_audio(job, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(cli, "OpenAI", lambda **kwargs: job[2])
    monkeypatch.setattr(cli, "analyze_transcript", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("analysis failed")))
    output = tmp_path / "notes.md"
    argv = [str(job[0]), "--out", str(output), "--work-dir", str(job[1])]
    assert cli.main(argv) == 1
    assert output.with_suffix(".transcript.json").exists()
    assert output.with_suffix(".review.md").exists()
    monkeypatch.setattr(cli, "analyze_transcript", lambda *args, **kwargs: {"summary": "Review summary"})
    assert cli.main(argv) == 0
    assert len(job[2].calls) == 2
    assert "Review summary" in output.read_text()


def test_render_only_preserves_imported_provenance_without_api(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = tmp_path / "saved.json"
    source.write_text(json.dumps({"meta": {"transcribe_model": "original-model"}, "transcript": {
        "text": "Hello", "segments": [{"start": 1, "end": 2, "speaker": "B", "text": "Hello"}]}}))
    output = tmp_path / "rendered.md"
    assert cli.main(["--render-only", "--transcript-input", str(source), "--out", str(output)]) == 0
    text = output.read_text()
    assert "original-model" in text and "B:" in text


def test_output_collision_prevents_any_api_call(job):
    assert cli.main([str(job[0]), "--out", str(job[0])]) == 1
    assert job[2].calls == []


def test_changed_analysis_model_and_invalid_cache_never_retranscribe(job, monkeypatch, tmp_path):
    from smart_transcriber.storage import StageStore
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(cli, "OpenAI", lambda **kwargs: job[2])
    analyses = []

    def analyze(*args, **kwargs):
        analyses.append(args[1])
        return {"summary": "Saved summary"}

    monkeypatch.setattr(cli, "analyze_transcript", analyze)
    output, combined = tmp_path / "notes.md", tmp_path / "combined.json"
    argv = [str(job[0]), "--out", str(output), "--work-dir", str(job[1]), "--json-out", str(combined)]
    assert cli.main(argv) == 0
    assert cli.load_transcript(combined)[0]["meta"]["analysis_model"] == pipeline.ANALYSIS_MODEL
    assert cli.main(argv) == 0
    assert len(analyses) == 1
    index = next((job[1] / "analysis" / "cache").glob("analysis-*.json"))
    settings = json.loads(index.read_text())["settings"]
    StageStore(job[1] / "analysis").save("analysis", settings, {"sections": [{"heading": 7}]})
    assert cli.main(argv) == 0
    assert len(analyses) == 2
    assert cli.main(argv + ["--analysis-model", "configured-alternative"]) == 0
    assert analyses[-1] == "configured-alternative"
    assert len(job[2].calls) == 2


def test_reconciliation_failure_keeps_both_paid_results(job, monkeypatch):
    monkeypatch.setattr(pipeline, "reconcile", lambda *args: (_ for _ in ()).throw(RuntimeError("failure")))
    result = run(job)
    assert result["text"] == SOURCE
    assert result["wording_text"] == WORDING
    assert result["reconciliation"]["status"] == "failed"
    assert len(list(job[1].rglob("raw/diarization-*.json"))) == 1
    assert len(list(job[1].rglob("raw/wording-*.json"))) == 1
    run(job)
    assert len(job[2].calls) == 2


def test_uncached_run_without_key_fails_without_api(job, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli.main([str(job[0]), "--work-dir", str(job[1]), "--out", str(job[1] / "notes.md"), "--force"]) == 1
    assert job[2].calls == []
    assert "without --force" in capsys.readouterr().err


def test_legacy_speaker_import_is_explicitly_marked(tmp_path):
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps({"transcript": {"text": "Hi", "segments": [{"text": "Hi", "start": 0, "end": 1}]},
        "analysis": {"segment_speakers": [{"segment_index": 0, "speaker": "Speaker 1"}]}}))
    result, _ = cli.load_transcript(source)
    assert result["segments"][0]["speaker_source"] == "legacy_inferred"
    assert "not verified" in result["meta"]["speaker_warning"]
