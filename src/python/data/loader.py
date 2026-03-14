"""
loader.py
=========
Load raw crime and geographic data into pandas DataFrames / GeoPandas
GeoDataFrames ready for preprocessing and analysis.

Supports
--------
- Texas DPS UCR CSV (from Socrata)
- FBI Crime Data Explorer JSON
- TIGER/Line shapefiles (counties, places)
- Generic lon/lat CSV (user-supplied incident-level data)
- Synthetic demo dataset (no external files required)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from ..utils import get_data_dir, get_logger, safe_read_csv

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column name aliases – map various raw column names to canonical names
# ---------------------------------------------------------------------------
_LON_ALIASES = ["longitude", "lon", "x", "lng", "long"]
_LAT_ALIASES = ["latitude",  "lat", "y"]

_CANONICAL = {
    "offense_type":   ["offense_type", "offense", "crime_type", "offense_desc",
                       "nibrs_offense_name", "ucr_offense"],
    "year":           ["year", "report_year", "incident_year", "data_year"],
    "county":         ["county", "county_name", "cnty_nm"],
    "city":           ["city", "city_name", "place_name", "agency_city"],
    "offense_count":  ["offense_count", "count", "num_offenses", "incident_count",
                       "value"],
}


def _resolve_col(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    """Return first column name from aliases that exists in df."""
    lower_cols = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower_cols:
            return lower_cols[alias.lower()]
    return None


def _resolve_lon_lat(df: pd.DataFrame) -> tuple[str, str]:
    lon_col = _resolve_col(df, _LON_ALIASES)
    lat_col = _resolve_col(df, _LAT_ALIASES)
    if lon_col is None or lat_col is None:
        raise ValueError(
            f"Could not find longitude/latitude columns. "
            f"Available columns: {list(df.columns)}"
        )
    return lon_col, lat_col


# ---------------------------------------------------------------------------
# UCR county-level CSV
# ---------------------------------------------------------------------------
def load_ucr_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Texas DPS UCR county-level crime CSV.

    Returns a DataFrame with canonical column names:
        year, county, offense_type, offense_count
    """
    path = path or get_data_dir("raw") / "texas_ucr_socrata.csv"
    log.info("Loading UCR CSV from %s", path)
    df = safe_read_csv(path, low_memory=False)
    log.info("  Raw shape: %s", df.shape)

    # Rename to canonical
    rename = {}
    for canonical, aliases in _CANONICAL.items():
        col = _resolve_col(df, aliases)
        if col and col != canonical:
            rename[col] = canonical
    df = df.rename(columns=rename)

    # Coerce types
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "offense_count" in df.columns:
        df["offense_count"] = pd.to_numeric(df["offense_count"], errors="coerce").fillna(0)

    log.info("  Loaded %d rows, %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# FBI CDE JSON
# ---------------------------------------------------------------------------
def load_fbi_json(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load FBI Crime Data Explorer JSON response into a flat DataFrame.
    """
    path = path or get_data_dir("raw") / "fbi_cde_TX_2018_2022.json"
    log.info("Loading FBI CDE JSON from %s", path)

    with open(path, "r") as fh:
        raw = json.load(fh)

    # CDE response schema: {"data": [...], "pagination": {...}}
    records = raw.get("data", raw) if isinstance(raw, dict) else raw

    if not records:
        log.warning("FBI JSON appears empty")
        return pd.DataFrame()

    df = pd.json_normalize(records)

    # Rename canonical
    rename = {}
    for canonical, aliases in _CANONICAL.items():
        col = _resolve_col(df, aliases)
        if col and col != canonical:
            rename[col] = canonical
    df = df.rename(columns=rename)

    log.info("  Loaded %d rows from FBI CDE JSON", len(df))
    return df


# ---------------------------------------------------------------------------
# Generic incident-level CSV (lon/lat rows)
# ---------------------------------------------------------------------------
def load_incident_csv(path: Path,
                      lon_col: Optional[str] = None,
                      lat_col: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Load a CSV of individual crime incidents that includes coordinates.

    Returns a GeoDataFrame (EPSG:4326) with a 'geometry' Point column.
    """
    log.info("Loading incident CSV from %s", path)
    df = safe_read_csv(path, low_memory=False)

    # Auto-detect coordinate columns if not specified
    _lon_col = lon_col or _resolve_col(df, _LON_ALIASES)
    _lat_col = lat_col or _resolve_col(df, _LAT_ALIASES)
    if _lon_col is None or _lat_col is None:
        raise ValueError("Could not detect lon/lat columns; pass them explicitly.")

    df[_lon_col] = pd.to_numeric(df[_lon_col], errors="coerce")
    df[_lat_col] = pd.to_numeric(df[_lat_col], errors="coerce")
    df = df.dropna(subset=[_lon_col, _lat_col])

    geometry = [Point(xy) for xy in zip(df[_lon_col], df[_lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    log.info("  Loaded %d incident points", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# TIGER/Line shapefiles
# ---------------------------------------------------------------------------
def load_county_shapefile(path: Optional[Path] = None,
                          texas_only: bool = True) -> gpd.GeoDataFrame:
    """
    Load US county TIGER shapefile and optionally filter to Texas (FIPS=48).
    """
    path = path or (get_data_dir("shapefiles") /
                    "tl_2022_us_county" / "tl_2022_us_county.shp")

    log.info("Loading county shapefile from %s", path)
    gdf = gpd.read_file(str(path))
    gdf = gdf.to_crs("EPSG:4326")

    if texas_only:
        gdf = gdf[gdf["STATEFP"] == "48"].copy()
        log.info("  Filtered to Texas: %d counties", len(gdf))

    gdf["county_name"] = gdf["NAME"].str.upper()
    return gdf


def load_places_shapefile(path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Load Texas incorporated places (cities) shapefile."""
    path = path or (get_data_dir("shapefiles") /
                    "tl_2022_48_place" / "tl_2022_48_place.shp")

    log.info("Loading places shapefile from %s", path)
    gdf = gpd.read_file(str(path))
    gdf = gdf.to_crs("EPSG:4326")
    log.info("  Loaded %d places", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Synthetic demo dataset  (no downloads required)
# ---------------------------------------------------------------------------
_TEXAS_CITIES = {
    "Houston":      (-95.37,  29.76),
    "San Antonio":  (-98.49,  29.42),
    "Dallas":       (-96.80,  32.78),
    "Austin":       (-97.74,  30.27),
    "Fort Worth":   (-97.33,  32.75),
    "El Paso":      (-106.49, 31.76),
    "Arlington":    (-97.11,  32.74),
    "Corpus Christi":(-97.40, 27.80),
    "Lubbock":      (-101.85, 33.58),
    "Laredo":       (-99.51,  27.51),
}

_CRIME_TYPES = [
    "Burglary", "Larceny-Theft", "Motor Vehicle Theft",
    "Aggravated Assault", "Robbery", "Rape", "Murder",
    "Simple Assault", "Drug Offense", "Vandalism",
]


def generate_synthetic_dataset(n_incidents: int = 5000,
                               seed: int = 42) -> gpd.GeoDataFrame:
    """
    Generate a synthetic crime incident dataset for Texas (demo / testing).

    Each incident has:
        longitude, latitude, offense_type, year, city, severity
    """
    rng = np.random.default_rng(seed)

    cities       = list(_TEXAS_CITIES.keys())
    city_centers = list(_TEXAS_CITIES.values())
    city_weights = np.array([20, 15, 18, 14, 10, 6, 5, 4, 4, 4], dtype=float)
    city_weights /= city_weights.sum()

    chosen_idx  = rng.choice(len(cities), size=n_incidents, p=city_weights)
    chosen_city = [cities[i] for i in chosen_idx]

    # Scatter points around city centres (Gaussian, ~0.15° std ≈ 15 km)
    lons = np.array([city_centers[i][0] for i in chosen_idx])
    lats = np.array([city_centers[i][1] for i in chosen_idx])
    lons += rng.normal(0, 0.15, n_incidents)
    lats += rng.normal(0, 0.10, n_incidents)

    crime_weights = np.array([12, 30, 10, 15, 8, 4, 1, 10, 6, 4], dtype=float)
    crime_weights /= crime_weights.sum()
    offense_types = rng.choice(_CRIME_TYPES, size=n_incidents, p=crime_weights)

    severity_map = {
        "Murder": 10, "Rape": 9, "Robbery": 8, "Aggravated Assault": 7,
        "Burglary": 6, "Motor Vehicle Theft": 5, "Drug Offense": 5,
        "Simple Assault": 4, "Larceny-Theft": 3, "Vandalism": 2,
    }
    severities = np.array([severity_map[ot] for ot in offense_types])

    years = rng.integers(2018, 2023, size=n_incidents)

    df = pd.DataFrame({
        "longitude":    lons,
        "latitude":     lats,
        "offense_type": offense_types,
        "severity":     severities,
        "year":         years,
        "city":         chosen_city,
        "incident_id":  np.arange(n_incidents),
    })

    geometry = [Point(x, y) for x, y in zip(df.longitude, df.latitude)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    log.info("Generated synthetic dataset with %d incidents", n_incidents)
    return gdf
