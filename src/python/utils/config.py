"""
config.py
=========
Central configuration loader.  Reads config/settings.yaml and exposes
a singleton `cfg` object used throughout the package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Resolve paths relative to project root
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[3]          # texas-crime-geospatial-analysis/
CONFIG_PATH  = PROJECT_ROOT / "config" / "settings.yaml"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


class Config:
    """Thin wrapper around the YAML settings dict."""

    def __init__(self, data: dict) -> None:
        self._data = data

    # Allow attribute-style access: cfg.paths.raw_data
    def __getattr__(self, key: str) -> Any:
        val = self._data.get(key)
        if isinstance(val, dict):
            return Config(val)
        return val

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __repr__(self) -> str:
        return f"Config({list(self._data.keys())})"


# ---------------------------------------------------------------------------
# Paths helpers
# ---------------------------------------------------------------------------
def get_data_dir(subdir: str = "raw") -> Path:
    p = PROJECT_ROOT / "data" / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_output_dir(subdir: str = "maps") -> Path:
    p = PROJECT_ROOT / "outputs" / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


# Singleton
cfg = Config(_load_yaml(CONFIG_PATH))
