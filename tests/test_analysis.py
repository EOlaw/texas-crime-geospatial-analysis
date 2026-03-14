"""
tests/test_analysis.py
======================
Unit tests for the analysis layer (hotspot, clustering, stats, model).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.python.data.loader import generate_synthetic_dataset
from src.python.data.preprocessor import add_severity, clean_incident_gdf


@pytest.fixture(scope="module")
def gdf():
    raw = generate_synthetic_dataset(n_incidents=1000, seed=7)
    raw = clean_incident_gdf(raw)
    return add_severity(raw)


# ── KDE ───────────────────────────────────────────────────────────────────────
class TestKDE:
    def test_kde_returns_result(self, gdf):
        from src.python.analysis.hotspot_detection import compute_kde
        result = compute_kde(gdf, grid_size=50)
        assert result is not None

    def test_density_shape(self, gdf):
        from src.python.analysis.hotspot_detection import compute_kde
        result = compute_kde(gdf, grid_size=50)
        assert result.density.shape == (50, 50)

    def test_density_non_negative(self, gdf):
        from src.python.analysis.hotspot_detection import compute_kde
        result = compute_kde(gdf, grid_size=50)
        assert (result.density >= 0).all()

    def test_hotspot_mask_fraction(self, gdf):
        from src.python.analysis.hotspot_detection import compute_kde
        result = compute_kde(gdf, grid_size=50, hotspot_pct=90.0)
        fraction = result.hotspot_mask.mean()
        assert 0.05 < fraction < 0.20, f"Unexpected hotspot fraction: {fraction:.2f}"


# ── Quadrat analysis ──────────────────────────────────────────────────────────
class TestQuadrat:
    def test_quadrat_returns_result(self, gdf):
        from src.python.analysis.hotspot_detection import compute_quadrat_analysis
        result = compute_quadrat_analysis(gdf, n_cols=10, n_rows=10)
        assert result is not None

    def test_quadrat_grid_size(self, gdf):
        from src.python.analysis.hotspot_detection import compute_quadrat_analysis
        result = compute_quadrat_analysis(gdf, n_cols=10, n_rows=10)
        assert len(result.grid_gdf) == 100

    def test_vmr_positive(self, gdf):
        from src.python.analysis.hotspot_detection import compute_quadrat_analysis
        result = compute_quadrat_analysis(gdf, n_cols=10, n_rows=10)
        assert result.vmr >= 0


# ── DBSCAN ────────────────────────────────────────────────────────────────────
class TestDBSCAN:
    def test_dbscan_finds_clusters(self, gdf):
        from src.python.analysis.spatial_clustering import run_dbscan
        result = run_dbscan(gdf, eps_deg=0.15, min_pts=5)
        assert result.n_clusters > 0

    def test_dbscan_labels_length(self, gdf):
        from src.python.analysis.spatial_clustering import run_dbscan
        result = run_dbscan(gdf, eps_deg=0.15, min_pts=5)
        assert len(result.gdf) == len(gdf)

    def test_dbscan_label_column(self, gdf):
        from src.python.analysis.spatial_clustering import run_dbscan
        result = run_dbscan(gdf, eps_deg=0.15, min_pts=5)
        assert "cluster_id" in result.gdf.columns

    def test_cluster_gdf_has_geometry(self, gdf):
        from src.python.analysis.spatial_clustering import run_dbscan
        result = run_dbscan(gdf, eps_deg=0.15, min_pts=5)
        if len(result.cluster_gdf) > 0:
            assert "geometry" in result.cluster_gdf.columns


# ── K-Means ───────────────────────────────────────────────────────────────────
class TestKMeans:
    def test_kmeans_k_clusters(self, gdf):
        from src.python.analysis.spatial_clustering import run_kmeans
        result = run_kmeans(gdf, k=5)
        assert result.k == 5

    def test_kmeans_centre_count(self, gdf):
        from src.python.analysis.spatial_clustering import run_kmeans
        result = run_kmeans(gdf, k=5)
        assert len(result.centres) == 5

    def test_kmeans_inertia_positive(self, gdf):
        from src.python.analysis.spatial_clustering import run_kmeans
        result = run_kmeans(gdf, k=5)
        assert result.inertia > 0


# ── Hexbins ───────────────────────────────────────────────────────────────────
class TestHexbins:
    def test_hexbins_returns_gdf(self, gdf):
        from src.python.analysis.spatial_clustering import compute_hexbins
        result = compute_hexbins(gdf, cell_size=0.5)
        assert len(result.hex_gdf) > 0

    def test_hexbins_counts_positive(self, gdf):
        from src.python.analysis.spatial_clustering import compute_hexbins
        result = compute_hexbins(gdf, cell_size=0.5)
        assert (result.hex_gdf["count"] > 0).all()


# ── Ripley's K ────────────────────────────────────────────────────────────────
class TestRipleys:
    def test_ripleys_columns(self, gdf):
        from src.python.analysis.spatial_clustering import ripleys_k
        df = ripleys_k(gdf, r_values=np.linspace(0.01, 0.5, 10))
        assert set(["r", "K", "L", "L_minus_r"]).issubset(df.columns)

    def test_ripleys_positive_k(self, gdf):
        from src.python.analysis.spatial_clustering import ripleys_k
        df = ripleys_k(gdf, r_values=np.linspace(0.01, 0.5, 10))
        assert (df["K"] >= 0).all()


# ── Statistical analysis ──────────────────────────────────────────────────────
class TestStatistics:
    def test_crime_summary(self, gdf):
        from src.python.analysis.statistical_analysis import crime_summary_stats
        summary = crime_summary_stats(gdf, "severity")
        assert "offense_type" in summary.columns
        assert len(summary) > 0

    def test_temporal_trend(self, gdf):
        from src.python.analysis.statistical_analysis import temporal_trend
        df = gdf.drop(columns=["geometry"], errors="ignore")
        df["offense_count"] = 1
        result = temporal_trend(df)
        assert "year" in result.df.columns
        assert "count" in result.df.columns
        assert result.trend_direction in ("increasing", "decreasing", "no trend")


# ── Predictive model ──────────────────────────────────────────────────────────
class TestPredictiveModel:
    def test_feature_matrix_shape(self, gdf):
        import geopandas as gpd
        from shapely.geometry import box
        from src.python.analysis.predictive_model import build_feature_matrix
        from src.python.analysis.spatial_clustering import compute_hexbins

        hex_result = compute_hexbins(gdf, cell_size=0.5)
        hex_gdf    = hex_result.hex_gdf
        hex_gdf["offense_count"] = hex_gdf["count"]

        X, y, names = build_feature_matrix(hex_gdf, "offense_count",
                                            include_spatial_lag=False)
        assert X.shape[0] == len(hex_gdf)
        assert len(names) == X.shape[1]

    def test_random_forest_trains(self, gdf):
        from src.python.analysis.predictive_model import (
            build_feature_matrix, train_random_forest
        )
        from src.python.analysis.spatial_clustering import compute_hexbins

        hex_gdf = compute_hexbins(gdf, cell_size=0.5).hex_gdf
        hex_gdf["offense_count"] = hex_gdf["count"]

        X, y, names = build_feature_matrix(hex_gdf, "offense_count",
                                            include_spatial_lag=False)
        result = train_random_forest(X, y, names, n_estimators=20, cv_folds=2)

        assert result.cv_rmse_mean >= 0
        assert result.test_metrics["r2"] <= 1.0
        assert len(result.feature_importances) == len(names)
