"""Real media preparation checks on generated audio; no recordings or API calls."""

import json
import random
import shutil
import struct
import subprocess
import wave

import pytest

from smart_transcriber import audio
from smart_transcriber.storage import file_hash

pytestmark = pytest.mark.skipif(not all(shutil.which(c) for c in ("ffmpeg", "ffprobe")), reason="ffmpeg and ffprobe required")


def make_wave(path, seconds=4):
    rng = random.Random(42)
    # Distinct stereo signals, followed by one full second of silence.
    frames = b"".join(struct.pack("<hh", rng.randint(-30000, 30000), rng.randint(-10000, 10000))
                      for _ in range(8000 * (seconds - 1))) + b"\x00" * (8000 * 4)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(frames)
    return frames


def test_small_file_uploads_whole_without_preparation(tmp_path):
    source = tmp_path / "stereo.wav"
    make_wave(source)
    directory = tmp_path / "prepared"
    chunks, info = audio.prepare_audio(source, directory, 1, file_hash(source))
    assert chunks == [audio.AudioChunk(source, 0, 4)]
    assert info["channels"] == 2
    assert not directory.exists()


def test_split_preserves_exact_samples_channels_offsets_and_silent_tail(tmp_path, monkeypatch):
    source = tmp_path / "stereo.wav"
    original = make_wave(source)
    monkeypatch.setattr(audio, "UPLOAD_LIMIT", 30000)
    chunks, info = audio.prepare_audio(source, tmp_path / "prepared", 2, file_hash(source))
    assert len(chunks) > 1
    assert sum(c.duration for c in chunks) == pytest.approx(4)
    assert chunks[-1].offset + chunks[-1].duration == pytest.approx(4)
    decoded = []
    for index, chunk in enumerate(chunks):
        assert chunk.path.stat().st_size <= audio.UPLOAD_LIMIT
        assert audio.probe_audio(chunk.path)["channels"] == 2
        assert audio.probe_audio(chunk.path)["duration"] == pytest.approx(chunk.duration)
        assert chunk.offset == pytest.approx(sum(c.duration for c in chunks[:index]))
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(chunk.path), "-f", "s16le", "-"], capture_output=True, check=True)
        decoded.append(result.stdout)
    assert b"".join(decoded) == original
    monkeypatch.setattr(audio, "_run", lambda *args: (_ for _ in ()).throw(AssertionError("must reuse prepared media")))
    monkeypatch.setattr(audio, "probe_audio", lambda path: info)
    assert audio.prepare_audio(source, tmp_path / "prepared", 2, file_hash(source))[0] == chunks


def test_unsupported_container_is_converted_whole_losslessly(tmp_path):
    source = tmp_path / "stereo.data"
    make_wave(source)
    chunks, _ = audio.prepare_audio(source, tmp_path / "prepared", 1, file_hash(source))
    assert len(chunks) == 1
    assert chunks[0].path.suffix == ".flac"
    assert audio.probe_audio(chunks[0].path)["channels"] == 2


def test_known_speaker_duration_and_content_hash(tmp_path):
    source = tmp_path / "reference.wav"
    make_wave(source, 2)
    refs = audio.known_speaker_references([f"Jane={source}"])
    assert refs[0]["name"] == "Jane"
    assert refs[0]["sha256"] == file_hash(source)
    assert refs[0]["data_url"].startswith("data:audio/wav;base64,")
    with pytest.raises(ValueError, match="distinct"):
        audio.known_speaker_references([f"Jane={source}", f"Jane={source}"])
    make_wave(source, 1)
    with pytest.raises(ValueError, match="2 to 10"):
        audio.known_speaker_references([f"Jane={source}"])
    with pytest.raises(ValueError, match="at most four"):
        audio.known_speaker_references([f"Jane={source}"] * 5)


def test_probe_rejects_multiple_tracks_and_failed_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "_run", lambda *args: json.dumps({"streams": [
        {"codec_type": "audio", "duration": "10"}, {"codec_type": "audio", "duration": "10"}]}))
    with pytest.raises(ValueError, match="one audio track"):
        audio.probe_audio(tmp_path / "fake.wav")
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="ffprobe failed"):
        audio.probe_audio(tmp_path / "missing.wav")
