"""
fetcher.py
==========
Downloads Texas crime data from public APIs and stores raw files locally.

Primary sources
---------------
1. **data.texas.gov (Socrata API)**
   Texas DPS UCR Offense data – county-level annual crime counts.
   Dataset: https://data.texas.gov/Public-Safety/Texas-DPS-Uniform-Crime-Reporting-UCR-Program-Coun/xvp6-c9hn

2. **FBI Crime Data Explorer API** (requires free API key)
   City/agency-level offense data.

3. **US Census TIGER/Line shapefiles**
   Texas county and place boundaries.

All functions return the local ``Path`` where the file was saved.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from ..utils import get_data_dir, get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Socrata (data.texas.gov)
# ---------------------------------------------------------------------------
SOCRATA_BASE    = "https://data.texas.gov/resource/"
UCR_DATASET_ID  = "xvp6-c9hn"          # Texas DPS UCR county-level
MAX_SOCRATA_ROWS = 50_000


def fetch_texas_ucr_socrata(
    limit: int = MAX_SOCRATA_ROWS,
    app_token: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Download Texas DPS UCR county-level crime data from data.texas.gov.

    Parameters
    ----------
    limit     : max rows to retrieve (Socrata default is 1000).
    app_token : optional Socrata app token to raise rate limits.
    out_dir   : directory for the output CSV (defaults to data/raw/).

    Returns
    -------
    Path to the saved CSV file.
    """
    out_dir = out_dir or get_data_dir("raw")
    out_path = out_dir / "texas_ucr_socrata.csv"

    url     = f"{SOCRATA_BASE}{UCR_DATASET_ID}.csv"
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    params = {"$limit": limit}

    log.info("Fetching Texas UCR data from data.texas.gov (limit=%d)…", limit)
    resp = requests.get(url, headers=headers, params=params, timeout=60)
    resp.raise_for_status()

    out_path.write_bytes(resp.content)
    log.info("Saved UCR data → %s  (%d bytes)", out_path, len(resp.content))
    return out_path


# ---------------------------------------------------------------------------
# FBI Crime Data Explorer
# ---------------------------------------------------------------------------
FBI_CDE_BASE = "https://api.usa.gov/crime/fbi/cde/"


def fetch_fbi_state_data(
    state_abbr: str = "TX",
    year_from:  int = 2018,
    year_to:    int = 2022,
    api_key:    str = "DEMO_KEY",
    out_dir:    Optional[Path] = None,
) -> Path:
    """
    Fetch state-level offense summary from the FBI Crime Data Explorer API.

    A free API key can be obtained at https://api.data.gov/signup/
    The special 'DEMO_KEY' works with strict rate limits (40 req/hr).
    """
    out_dir  = out_dir or get_data_dir("raw")
    out_path = out_dir / f"fbi_cde_{state_abbr}_{year_from}_{year_to}.json"

    url = (
        f"{FBI_CDE_BASE}summarized/state/{state_abbr}/all-offenses"
        f"/{year_from}/{year_to}?API_KEY={api_key}"
    )

    log.info("Fetching FBI CDE data for %s (%d–%d)…", state_abbr, year_from, year_to)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    out_path.write_bytes(resp.content)
    log.info("Saved FBI CDE data → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# US Census TIGER/Line shapefiles
# ---------------------------------------------------------------------------
CENSUS_TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2022/"
TEXAS_FIPS        = "48"


def fetch_texas_counties_shapefile(out_dir: Optional[Path] = None) -> Path:
    """
    Download the Texas county boundaries TIGER/Line shapefile (2022).
    Saves the unzipped shapefile components to data/shapefiles/.
    """
    shp_dir  = out_dir or get_data_dir("shapefiles")
    out_path = shp_dir / "tl_2022_us_county" / "tl_2022_us_county.shp"

    if out_path.exists():
        log.info("County shapefile already present: %s", out_path)
        return out_path

    zip_url = f"{CENSUS_TIGER_BASE}COUNTY/tl_2022_us_county.zip"
    log.info("Downloading county shapefile from Census TIGER…")
    resp = requests.get(zip_url, timeout=120, stream=True)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        extract_to = shp_dir / "tl_2022_us_county"
        extract_to.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_to)

    log.info("Extracted county shapefile → %s", extract_to)
    return out_path


def fetch_texas_places_shapefile(out_dir: Optional[Path] = None) -> Path:
    """
    Download Texas incorporated places (cities) TIGER/Line shapefile.
    """
    shp_dir  = out_dir or get_data_dir("shapefiles")
    out_path = shp_dir / "tl_2022_48_place" / "tl_2022_48_place.shp"

    if out_path.exists():
        log.info("Places shapefile already present: %s", out_path)
        return out_path

    zip_url = f"{CENSUS_TIGER_BASE}PLACE/tl_2022_48_place.zip"
    log.info("Downloading Texas places shapefile…")
    resp = requests.get(zip_url, timeout=120, stream=True)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        extract_to = shp_dir / "tl_2022_48_place"
        extract_to.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_to)

    log.info("Extracted places shapefile → %s", extract_to)
    return out_path


# ---------------------------------------------------------------------------
# Convenience: fetch everything
# ---------------------------------------------------------------------------
def fetch_all(app_token: Optional[str] = None,
              fbi_api_key: str = "DEMO_KEY") -> dict[str, Path]:
    """Download all data sources and return a dict of name → Path."""
    results: dict[str, Path] = {}

    try:
        results["ucr_socrata"]       = fetch_texas_ucr_socrata(app_token=app_token)
    except Exception as exc:
        log.warning("UCR Socrata fetch failed: %s", exc)

    try:
        results["fbi_cde"]           = fetch_fbi_state_data(api_key=fbi_api_key)
    except Exception as exc:
        log.warning("FBI CDE fetch failed: %s", exc)

    try:
        results["county_shapefile"]  = fetch_texas_counties_shapefile()
    except Exception as exc:
        log.warning("County shapefile fetch failed: %s", exc)

    try:
        results["places_shapefile"]  = fetch_texas_places_shapefile()
    except Exception as exc:
        log.warning("Places shapefile fetch failed: %s", exc)

    return results
