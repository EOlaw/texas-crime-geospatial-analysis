"""
preprocessor.py
===============
Data cleaning, validation, feature engineering, and spatial join operations
to prepare raw crime data for analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from ..utils import (
    filter_texas_coords, get_logger, normalise_crime_type, get_data_dir
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Clean incident GeoDataFrame
# ---------------------------------------------------------------------------
def clean_incident_gdf(
    gdf:       gpd.GeoDataFrame,
    lon_col:   str = "longitude",
    lat_col:   str = "latitude",
    year_min:  int = 2010,
    year_max:  int = 2024,
) -> gpd.GeoDataFrame:
    """
    Core cleaning pipeline for incident-level GeoDataFrames.

    Steps
    -----
    1. Drop rows with null geometry.
    2. Clip to Texas bounding box.
    3. Drop duplicate incidents (same coords + offense_type + year).
    4. Normalise offense_type strings.
    5. Filter year range.
    6. Reset index.
    """
    n0 = len(gdf)

    # 1. Drop null geometry
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    # 2. Clip to Texas bbox
    if lon_col in gdf.columns and lat_col in gdf.columns:
        gdf = filter_texas_coords(gdf, lon_col, lat_col)

    # 3. Drop duplicates
    subset = [c for c in [lon_col, lat_col, "offense_type", "year"] if c in gdf.columns]
    if subset:
        gdf = gdf.drop_duplicates(subset=subset)

    # 4. Normalise crime type
    if "offense_type" in gdf.columns:
        gdf["offense_type"] = normalise_crime_type(gdf["offense_type"])

    # 5. Filter year range
    if "year" in gdf.columns:
        gdf["year"] = pd.to_numeric(gdf["year"], errors="coerce")
        gdf = gdf[gdf["year"].between(year_min, year_max, inclusive="both")]

    gdf = gdf.reset_index(drop=True)
    log.info("Cleaned incidents: %d → %d rows", n0, len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Spatial join: attach county/city to each incident
# ---------------------------------------------------------------------------
def attach_county(
    incidents: gpd.GeoDataFrame,
    counties:  gpd.GeoDataFrame,
    county_name_col: str = "county_name",
) -> gpd.GeoDataFrame:
    """
    Spatial join to attach county name (and GEOID) to each incident point.
    Both GeoDataFrames must share the same CRS.
    """
    if incidents.crs != counties.crs:
        counties = counties.to_crs(incidents.crs)

    joined = gpd.sjoin(
        incidents,
        counties[[county_name_col, "GEOID", "geometry"]],
        how="left",
        predicate="within",
    )

    # sjoin adds '_left'/'_right' suffixes for duplicate columns
    joined = joined.rename(columns={county_name_col: "county"})
    joined = joined.drop(columns=["index_right"], errors="ignore")
    log.info("Attached county to %d / %d incidents", joined["county"].notna().sum(), len(joined))
    return joined


def attach_city(
    incidents: gpd.GeoDataFrame,
    places:    gpd.GeoDataFrame,
    place_name_col: str = "NAME",
) -> gpd.GeoDataFrame:
    """Spatial join to attach city/place name to incident points."""
    if incidents.crs != places.crs:
        places = places.to_crs(incidents.crs)

    joined = gpd.sjoin(
        incidents,
        places[[place_name_col, "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.rename(columns={place_name_col: "place_name"})
    joined = joined.drop(columns=["index_right"], errors="ignore")
    return joined


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
_SEVERITY_MAP = {
    "murder":                10,
    "rape":                   9,
    "robbery":                8,
    "aggravated assault":     7,
    "burglary":               6,
    "motor vehicle theft":    5,
    "drug offense":           5,
    "simple assault":         4,
    "larceny-theft":          3,
    "vandalism":              2,
    "arson":                  7,
    "human trafficking":      9,
    "kidnapping/abduction":   8,
    "fraud":                  4,
    "embezzlement":           3,
}

_VIOLENT_CRIMES = {
    "murder", "rape", "robbery", "aggravated assault",
    "simple assault", "kidnapping/abduction", "human trafficking",
}

_PROPERTY_CRIMES = {
    "burglary", "larceny-theft", "motor vehicle theft",
    "arson", "vandalism", "fraud", "embezzlement",
}


def add_severity(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map offense_type to numeric severity [1–10]."""
    if "offense_type" not in gdf.columns:
        raise ValueError("offense_type column required")

    gdf = gdf.copy()
    gdf["severity"] = (
        gdf["offense_type"]
        .map(_SEVERITY_MAP)
        .fillna(3)                         # default medium-low
        .astype(int)
    )
    return gdf


def add_crime_category(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Classify each offense as 'violent', 'property', or 'other'."""
    gdf = gdf.copy()

    def categorise(ot: str) -> str:
        if ot in _VIOLENT_CRIMES:
            return "violent"
        if ot in _PROPERTY_CRIMES:
            return "property"
        return "other"

    if "offense_type" in gdf.columns:
        gdf["crime_category"] = gdf["offense_type"].apply(categorise)
    return gdf


def add_temporal_features(gdf: gpd.GeoDataFrame,
                          date_col: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Extract temporal features (month, day_of_week, hour, season).
    If a date_col exists it is parsed; otherwise only 'year' is used.
    """
    gdf = gdf.copy()

    if date_col and date_col in gdf.columns:
        dt = pd.to_datetime(gdf[date_col], errors="coerce", infer_datetime_format=True)
        gdf["month"]       = dt.dt.month
        gdf["day_of_week"] = dt.dt.dayofweek
        gdf["hour"]        = dt.dt.hour
        gdf["season"]      = dt.dt.month.map(
            lambda m: ("Winter" if m in [12, 1, 2] else
                       "Spring" if m in [3, 4, 5] else
                       "Summer" if m in [6, 7, 8] else "Fall")
        )
    return gdf


def aggregate_by_county_year(
    gdf:          gpd.GeoDataFrame,
    county_col:   str = "county",
    year_col:     str = "year",
    count_col:    str = "offense_count",
) -> pd.DataFrame:
    """
    Aggregate incident counts to county × year level.
    Useful when working with point data that needs to be joined back to polygons.
    """
    if county_col not in gdf.columns:
        raise ValueError(f"Column '{county_col}' not found")

    agg = (
        gdf.groupby([county_col, year_col])
           .size()
           .reset_index(name=count_col)
    )
    return agg


# ---------------------------------------------------------------------------
# Save / load processed data
# ---------------------------------------------------------------------------
def save_processed(gdf: gpd.GeoDataFrame, name: str = "incidents") -> Path:
    """Save a GeoDataFrame to the processed data directory as GeoParquet."""
    out_dir  = get_data_dir("processed")
    out_path = out_dir / f"{name}.parquet"
    gdf.to_parquet(str(out_path))
    log.info("Saved processed data → %s", out_path)
    return out_path


def load_processed(name: str = "incidents") -> gpd.GeoDataFrame:
    """Load a previously saved processed GeoDataFrame."""
    path = get_data_dir("processed") / f"{name}.parquet"
    gdf  = gpd.read_parquet(str(path))
    log.info("Loaded processed data from %s (%d rows)", path, len(gdf))
    return gdf
