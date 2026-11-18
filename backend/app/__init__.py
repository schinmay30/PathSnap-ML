"""Expose the repository's src-layout ML package to backend-only uvicorn launches."""

from pathlib import Path
import sys

_src = str(Path(__file__).resolve().parents[2] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
