"""CLI smoke tests for smart_transcriber."""

import os
import subprocess
import sys


def run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    run_env = env if env is not None else os.environ.copy()
    return subprocess.run(
        [sys.executable, "-m", "smart_transcriber.cli", *args],
        capture_output=True,
        text=True,
        env=run_env,
    )


class TestCliSmoke:
    def test_help_exits_zero(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "audio" in result.stdout.lower() or "transcribe" in result.stdout.lower()

    def test_version_exits_zero(self):
        result = run_cli("--version")
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_missing_api_key_exits_one(self):
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        result = run_cli("nonexistent.mp3", env=env)
        assert result.returncode == 1
        assert "OPENAI_API_KEY" in result.stderr

    def test_style_transcript_with_no_analysis_exits_one(self):
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "dummy"
        result = run_cli("nonexistent.mp3", "--style", "transcript", "--no-analysis", env=env)
        assert result.returncode == 1
        assert "cannot combine" in result.stderr.lower()
