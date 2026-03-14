"""
tests/test_visualization.py
============================
Unit tests for the visualization layer (maps, static figures).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend for CI
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.python.data.loader      import generate_synthetic_dataset
from src.python.data.preprocessor import add_severity, clean_incident_gdf


@pytest.fixture(scope="module")
def gdf():
    raw = generate_synthetic_dataset(n_incidents=500, seed=3)
    raw = clean_incident_gdf(raw)
    return add_severity(raw)


# ── Static figures ────────────────────────────────────────────────────────────
class TestStaticFigures:
    def test_crime_type_bar(self, gdf):
        from src.python.visualization.heatmap import plot_crime_type_bar
        fig = plot_crime_type_bar(gdf, save=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_temporal_trend_plot(self, gdf):
        import pandas as pd
        from src.python.visualization.heatmap import plot_temporal_trend
        df = gdf.drop(columns=["geometry"], errors="ignore")
        df["offense_count"] = 1
        from src.python.analysis.statistical_analysis import temporal_trend
        trend = temporal_trend(df)
        fig = plot_temporal_trend(trend.df, save=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cluster_scatter_plot(self, gdf):
        from src.python.analysis.spatial_clustering import run_dbscan
        from src.python.visualization.heatmap import plot_cluster_scatter
        db  = run_dbscan(gdf, eps_deg=0.15, min_pts=5)
        fig = plot_cluster_scatter(db.gdf, save=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_kde_surface_plot(self, gdf):
        from src.python.analysis.hotspot_detection import compute_kde
        from src.python.visualization.heatmap import plot_kde_surface
        kde = compute_kde(gdf, grid_size=40)
        fig = plot_kde_surface(kde, county_gdf=None, save=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ripleys_l_plot(self, gdf):
        import numpy as np
        from src.python.analysis.spatial_clustering import ripleys_k
        from src.python.visualization.heatmap import plot_ripleys_l
        rk  = ripleys_k(gdf, r_values=np.linspace(0.01, 0.5, 10))
        fig = plot_ripleys_l(rk, save=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ── Folium maps ───────────────────────────────────────────────────────────────
class TestFoliumMaps:
    def test_base_map_returns_folium_map(self):
        import folium
        from src.python.visualization.map_generator import base_map
        m = base_map()
        assert isinstance(m, folium.Map)

    def test_kde_heatmap_returns_map(self, gdf):
        import folium
        from src.python.visualization.map_generator import kde_heatmap_map
        m = kde_heatmap_map(gdf, out_name="_test_kde_heatmap")
        assert isinstance(m, folium.Map)

    def test_incident_map_returns_map(self, gdf):
        import folium
        from src.python.visualization.map_generator import incident_point_map
        m = incident_point_map(gdf, max_points=100, out_name="_test_incident_map")
        assert isinstance(m, folium.Map)

    def test_cluster_map_returns_map(self, gdf):
        import folium
        from src.python.analysis.spatial_clustering import run_dbscan
        from src.python.visualization.map_generator import cluster_map
        db = run_dbscan(gdf, eps_deg=0.15, min_pts=5)
        m  = cluster_map(db.gdf, db.cluster_gdf, out_name="_test_cluster_map")
        assert isinstance(m, folium.Map)
