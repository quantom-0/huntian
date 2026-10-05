"""Test package bootstrap.

Adds the in-repo ``python/`` and project root to ``sys.path`` so the suite runs
from a clean checkout with no installation step:

    python -m unittest discover -t . -s tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
PLUGIN_DIR = ROOT / "plugins"
DECKS_DIR = ROOT / "decks"
DECK_PATH = DECKS_DIR / "demo_hemt.json"
DECK_SP1D_PATH = DECKS_DIR / "demo_hemt_sp1d.json"

for _entry in (str(PYTHON_DIR), str(ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

__all__ = [
    "ROOT",
    "PYTHON_DIR",
    "PLUGIN_DIR",
    "DECKS_DIR",
    "DECK_PATH",
    "DECK_SP1D_PATH",
]
