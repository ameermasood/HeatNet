"""Shared bootstrap helpers for top-level task scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"


def add_src_to_path():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    return ROOT
