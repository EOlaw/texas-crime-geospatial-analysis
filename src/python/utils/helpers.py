"""
helpers.py
==========
Utility functions used across the package.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s – %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------
def timed(func: Callable) -> Callable:
    """Log the execution time of a function."""
    log = get_logger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0     = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info("%s completed in %.2f s", func.__qualname__, elapsed)
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------
TEXAS_BBOX = {
    "min_lon": -106.65,
    "max_lon":  -93.51,
    "min_lat":   25.84,
    "max_lat":   36.50,
}


def is_in_texas(lon: float, lat: float) -> bool:
    """Return True if (lon, lat) falls within the Texas bounding box."""
    bb = TEXAS_BBOX
    return (bb["min_lon"] <= lon <= bb["max_lon"] and
            bb["min_lat"] <= lat <= bb["max_lat"])


def filter_texas_coords(df: pd.DataFrame,
                        lon_col: str = "longitude",
                        lat_col: str = "latitude") -> pd.DataFrame:
    """Drop rows whose coordinates fall outside Texas."""
    bb = TEXAS_BBOX
    mask = (
        df[lon_col].between(bb["min_lon"], bb["max_lon"]) &
        df[lat_col].between(bb["min_lat"], bb["max_lat"])
    )
    return df[mask].copy()


def haversine_vectorised(lon1: np.ndarray,
                         lat1: np.ndarray,
                         lon2: float,
                         lat2: float) -> np.ndarray:
    """Vectorised Haversine distance (km) from an array of points to a single point."""
    R    = 6371.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a    = (np.sin(dlat / 2) ** 2 +
            np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
            np.sin(dlon / 2) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with error context."""
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def normalise_crime_type(series: pd.Series) -> pd.Series:
    """Standardise crime type strings: lowercase, strip, consolidate."""
    return (
        series.astype(str)
              .str.lower()
              .str.strip()
              .str.replace(r"\s+", " ", regex=True)
    )


# ---------------------------------------------------------------------------
# Risk score helper
# ---------------------------------------------------------------------------
def min_max_scale(arr: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)
