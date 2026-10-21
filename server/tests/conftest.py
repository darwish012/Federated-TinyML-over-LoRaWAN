"""pytest setup for the v3 tests (the v2 tests insert the same paths themselves)."""
from __future__ import annotations

from pathlib import Path
import sys

SERVER = Path(__file__).resolve().parents[1]
for path in (SERVER, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
