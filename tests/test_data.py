"""
tests/test_data.py
==================
Unit tests for the data loading and preprocessing layer.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_gdf():
    """Small synthetic incident GeoDataFrame."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.python.data.loader import generate_synthetic_dataset
    return generate_synthetic_dataset(n_incidents=500, seed=0)


# ── loader tests ──────────────────────────────────────────────────────────────
class TestSyntheticDataset:
    def test_returns_geodataframe(self, synthetic_gdf):
        assert isinstance(synthetic_gdf, gpd.GeoDataFrame)

    def test_row_count(self, synthetic_gdf):
        assert len(synthetic_gdf) == 500

    def test_required_columns(self, synthetic_gdf):
        for col in ["longitude", "latitude", "offense_type", "year", "city"]:
            assert col in synthetic_gdf.columns, f"Missing column: {col}"

    def test_geometry_column(self, synthetic_gdf):
        assert "geometry" in synthetic_gdf.columns
        assert synthetic_gdf.geometry.notna().all()

    def test_crs_is_4326(self, synthetic_gdf):
        assert synthetic_gdf.crs is not None
        assert synthetic_gdf.crs.to_epsg() == 4326

    def test_years_in_range(self, synthetic_gdf):
        assert synthetic_gdf["year"].between(2018, 2022).all()

    def test_longitude_in_texas(self, synthetic_gdf):
        lons = synthetic_gdf["longitude"]
        assert (lons >= -110).all() and (lons <= -90).all()

    def test_latitude_in_texas(self, synthetic_gdf):
        lats = synthetic_gdf["latitude"]
        assert (lats >= 24).all() and (lats <= 37).all()


# ── preprocessor tests ────────────────────────────────────────────────────────
class TestCleanIncidentGdf:
    def test_removes_out_of_range_years(self, synthetic_gdf):
        from src.python.data.preprocessor import clean_incident_gdf
        dirty = synthetic_gdf.copy()
        dirty.loc[0, "year"] = 1900
        dirty.loc[1, "year"] = 2099
        cleaned = clean_incident_gdf(dirty, year_min=2018, year_max=2023)
        assert (cleaned["year"] >= 2018).all()
        assert (cleaned["year"] <= 2023).all()

    def test_normalises_offense_type(self, synthetic_gdf):
        from src.python.data.preprocessor import clean_incident_gdf
        dirty = synthetic_gdf.copy()
        dirty["offense_type"] = "  Murder  "
        cleaned = clean_incident_gdf(dirty)
        assert (cleaned["offense_type"] == "murder").all()

    def test_row_count_non_negative(self, synthetic_gdf):
        from src.python.data.preprocessor import clean_incident_gdf
        cleaned = clean_incident_gdf(synthetic_gdf)
        assert len(cleaned) >= 0


class TestAddSeverity:
    def test_severity_column_added(self, synthetic_gdf):
        from src.python.data.preprocessor import add_severity, clean_incident_gdf
        gdf = clean_incident_gdf(synthetic_gdf)
        gdf = add_severity(gdf)
        assert "severity" in gdf.columns

    def test_severity_range(self, synthetic_gdf):
        from src.python.data.preprocessor import add_severity, clean_incident_gdf
        gdf = clean_incident_gdf(synthetic_gdf)
        gdf = add_severity(gdf)
        assert gdf["severity"].between(1, 10).all()


class TestAddCrimeCategory:
    def test_category_values(self, synthetic_gdf):
        from src.python.data.preprocessor import add_crime_category, clean_incident_gdf
        gdf = clean_incident_gdf(synthetic_gdf)
        gdf = add_crime_category(gdf)
        assert gdf["crime_category"].isin(["violent", "property", "other"]).all()


# ── helpers tests ─────────────────────────────────────────────────────────────
class TestHelpers:
    def test_filter_texas_coords(self, synthetic_gdf):
        from src.python.utils.helpers import filter_texas_coords
        df = pd.DataFrame({"longitude": [-99.5, 0.0, -98.0],
                           "latitude":  [30.0,  50.0,  29.0]})
        filtered = filter_texas_coords(df)
        assert len(filtered) == 2

    def test_min_max_scale(self):
        from src.python.utils.helpers import min_max_scale
        arr = np.array([0, 5, 10], dtype=float)
        scaled = min_max_scale(arr)
        assert scaled[0] == pytest.approx(0.0)
        assert scaled[-1] == pytest.approx(1.0)

    def test_min_max_scale_constant(self):
        from src.python.utils.helpers import min_max_scale
        arr = np.array([5, 5, 5], dtype=float)
        scaled = min_max_scale(arr)
        assert (scaled == 0).all()

    def test_haversine_vectorised(self):
        from src.python.utils.helpers import haversine_vectorised
        lons = np.array([-95.37])
        lats = np.array([29.76])
        dist = haversine_vectorised(lons, lats, -96.80, 32.78)
        assert 350 < dist[0] < 430, f"Expected ~392 km, got {dist[0]:.1f}"
