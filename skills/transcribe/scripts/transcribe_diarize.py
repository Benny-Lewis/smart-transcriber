#!/usr/bin/env python3
"""Compatibility launcher; the smart-transcriber package owns all API behavior."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    configured = os.getenv("SMART_TRANSCRIBER_PYTHON")
    project = Path.home() / "dev" / "audio-transcriber"
    local = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    interpreter = configured or (str(local) if local.is_file() else sys.executable)
    try:
        return subprocess.call([interpreter, "-m", "smart_transcriber.compat", *sys.argv[1:]])
    except OSError as exc:
        print(f"Cannot launch smart-transcriber: {exc}. Set SMART_TRANSCRIBER_PYTHON to its Python interpreter.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
