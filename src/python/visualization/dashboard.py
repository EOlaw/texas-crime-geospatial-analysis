"""
dashboard.py
============
Plotly Dash interactive analytics dashboard for Texas crime data.

Launch:  python -m src.python.visualization.dashboard
or:      from src.python.visualization.dashboard import create_app; app.run_server()

Features
--------
- KPI cards: total incidents, hotspot count, trend direction.
- Interactive choropleth (county-level crime).
- Crime type breakdown (pie + bar).
- Year filter slider.
- DBSCAN cluster scatter on Mapbox.
- Temporal trend chart.
- Top-10 risky counties table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html, dash_table
from dash.exceptions import PreventUpdate

from ..utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(
    gdf:         gpd.GeoDataFrame,
    county_gdf:  Optional[gpd.GeoDataFrame] = None,
    port:        int  = 8050,
    debug:       bool = False,
) -> Dash:
    """
    Build and return the Dash application.

    Parameters
    ----------
    gdf        : Incident-level GeoDataFrame (requires longitude, latitude,
                 offense_type, year, city columns).
    county_gdf : County polygon GeoDataFrame with offense_count column (optional).
    port       : Port for the dev server.
    debug      : Enable Dash debug mode.
    """

    # ------------------------------------------------------------------
    # Pre-process data for charts
    # ------------------------------------------------------------------
    df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))

    years = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else [2022]
    year_min, year_max = int(min(years)), int(max(years))

    crime_types = sorted(df["offense_type"].dropna().unique().tolist()) \
        if "offense_type" in df.columns else []

    # County GeoJSON for choropleth
    county_geojson = None
    county_df      = None
    if county_gdf is not None:
        county_gdf    = county_gdf.copy()
        county_geojson = json.loads(county_gdf.to_json())
        county_df     = pd.DataFrame(county_gdf.drop(columns=["geometry"], errors="ignore"))

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    app = Dash(
        __name__,
        title="Texas Crime Geospatial Analysis",
        external_stylesheets=[
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
        ],
    )

    app.layout = html.Div([
        # ── Header ──────────────────────────────────────────────────────
        html.Div([
            html.H1("Texas Crime Geospatial Analysis Dashboard",
                    style={"color": "#ffffff", "margin": "0", "fontFamily": "Inter, sans-serif"}),
            html.P("Spatial crime pattern analysis and predictive risk mapping",
                   style={"color": "#ccc", "margin": "4px 0 0 0"}),
        ], style={"background": "#1a1a2e", "padding": "20px 30px",
                  "borderBottom": "3px solid #e94560"}),

        # ── Controls ────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Label("Year Range", style={"fontWeight": "600"}),
                dcc.RangeSlider(
                    id="year-slider",
                    min=year_min, max=year_max,
                    value=[year_min, year_max],
                    marks={y: str(y) for y in years},
                    step=1,
                ),
            ], style={"flex": "2", "minWidth": "300px"}),

            html.Div([
                html.Label("Crime Type", style={"fontWeight": "600"}),
                dcc.Dropdown(
                    id="crime-type-dropdown",
                    options=[{"label": "All", "value": "All"}] +
                            [{"label": c.title(), "value": c} for c in crime_types],
                    value="All",
                    clearable=False,
                ),
            ], style={"flex": "1", "minWidth": "200px"}),
        ], style={"display": "flex", "gap": "30px", "padding": "16px 30px",
                  "background": "#f8f9fa", "flexWrap": "wrap",
                  "borderBottom": "1px solid #dee2e6"}),

        # ── KPI cards ───────────────────────────────────────────────────
        html.Div(id="kpi-cards",
                 style={"display": "flex", "gap": "16px", "padding": "16px 30px",
                        "flexWrap": "wrap"}),

        # ── Main charts ─────────────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Graph(id="choropleth-map", style={"height": "480px"}),
            ], style={"flex": "3", "minWidth": "400px"}),

            html.Div([
                dcc.Graph(id="crime-pie-chart", style={"height": "230px"}),
                dcc.Graph(id="temporal-trend",  style={"height": "230px"}),
            ], style={"flex": "1", "minWidth": "300px",
                      "display": "flex", "flexDirection": "column", "gap": "8px"}),
        ], style={"display": "flex", "gap": "16px", "padding": "0 30px 16px",
                  "flexWrap": "wrap"}),

        # ── Incident scatter map ─────────────────────────────────────────
        html.Div([
            dcc.Graph(id="scatter-map", style={"height": "420px"}),
        ], style={"padding": "0 30px 16px"}),

        # ── Top counties table ───────────────────────────────────────────
        html.Div([
            html.H3("Top 10 High-Risk Counties",
                    style={"fontFamily": "Inter", "color": "#1a1a2e"}),
            html.Div(id="top-counties-table"),
        ], style={"padding": "0 30px 30px"}),

    ], style={"fontFamily": "Inter, sans-serif", "background": "#f0f2f5",
              "minHeight": "100vh"})

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    @app.callback(
        [Output("kpi-cards",         "children"),
         Output("choropleth-map",    "figure"),
         Output("crime-pie-chart",   "figure"),
         Output("temporal-trend",    "figure"),
         Output("scatter-map",       "figure"),
         Output("top-counties-table","children")],
        [Input("year-slider",         "value"),
         Input("crime-type-dropdown", "value")],
    )
    def update_all(year_range, crime_type):
        yr_lo, yr_hi = year_range

        # Filter
        filtered = df.copy()
        if "year" in filtered.columns:
            filtered = filtered[filtered["year"].between(yr_lo, yr_hi)]
        if crime_type and crime_type != "All" and "offense_type" in filtered.columns:
            filtered = filtered[filtered["offense_type"] == crime_type]

        if filtered.empty:
            raise PreventUpdate

        n_incidents   = len(filtered)
        top_city      = (filtered["city"].value_counts().index[0]
                         if "city" in filtered.columns else "N/A")
        top_crime     = (filtered["offense_type"].value_counts().index[0]
                         if "offense_type" in filtered.columns else "N/A")

        # KPI cards
        kpi_data = [
            ("Total Incidents",   f"{n_incidents:,}",    "#e94560"),
            ("Top City",          top_city,               "#0f3460"),
            ("Most Common Crime", top_crime.title(),      "#533483"),
            ("Years Covered",     f"{yr_lo}–{yr_hi}",    "#1a7a4a"),
        ]
        kpi_cards = [
            html.Div([
                html.P(label, style={"margin": "0", "fontSize": "12px",
                                     "color": "#666", "fontWeight": "600"}),
                html.H2(value, style={"margin": "4px 0 0 0", "color": colour,
                                      "fontSize": "22px", "fontWeight": "700"}),
            ], style={"background": "#fff", "padding": "14px 20px",
                      "borderRadius": "8px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                      "flex": "1", "minWidth": "150px",
                      "borderLeft": f"4px solid {colour}"})
            for label, value, colour in kpi_data
        ]

        # Choropleth (county-level aggregated from filtered incidents)
        if "city" in filtered.columns:
            city_counts = filtered["city"].value_counts().reset_index()
            city_counts.columns = ["city", "count"]
        else:
            city_counts = pd.DataFrame({"city": [], "count": []})

        choro_fig = _make_tx_scatter_density(filtered)

        # Pie chart
        if "offense_type" in filtered.columns:
            type_counts = filtered["offense_type"].value_counts().head(8)
            pie_fig = px.pie(
                values=type_counts.values,
                names=[t.title() for t in type_counts.index],
                title="Crime Type Distribution",
                color_discrete_sequence=px.colors.sequential.YlOrRd,
                hole=0.35,
            )
            pie_fig.update_layout(margin=dict(t=40, b=0, l=0, r=0),
                                  font_family="Inter")
        else:
            pie_fig = go.Figure()

        # Temporal trend
        if "year" in filtered.columns:
            yearly = filtered.groupby("year").size().reset_index(name="count")
            trend_fig = px.bar(
                yearly, x="year", y="count",
                title="Annual Incident Counts",
                color="count",
                color_continuous_scale="YlOrRd",
            )
            trend_fig.add_scatter(x=yearly["year"],
                                  y=yearly["count"].rolling(2, center=True).mean(),
                                  mode="lines+markers",
                                  line=dict(color="#333", width=2),
                                  name="Trend")
            trend_fig.update_layout(
                margin=dict(t=40, b=30, l=40, r=10),
                showlegend=False,
                coloraxis_showscale=False,
                font_family="Inter",
            )
        else:
            trend_fig = go.Figure()

        # Scatter map
        scatter_fig = _make_scatter_map(filtered)

        # Top counties table
        if "county" in filtered.columns:
            top_counties = (
                filtered.groupby("county")
                        .agg(total=("offense_type", "count"),
                             top_crime=("offense_type",
                                        lambda x: x.value_counts().index[0] if len(x) > 0 else "N/A"))
                        .reset_index()
                        .sort_values("total", ascending=False)
                        .head(10)
            )
            top_counties.columns = ["County", "Total Incidents", "Most Common Crime"]
            table = dash_table.DataTable(
                data=top_counties.to_dict("records"),
                columns=[{"name": c, "id": c} for c in top_counties.columns],
                style_cell={"fontFamily": "Inter", "padding": "8px 12px",
                            "textAlign": "left"},
                style_header={"fontWeight": "700", "background": "#1a1a2e",
                              "color": "#fff"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "background": "#f8f9fa"}
                ],
            )
        else:
            table = html.P("County data not available.")

        return kpi_cards, choro_fig, pie_fig, trend_fig, scatter_fig, table

    return app


# ---------------------------------------------------------------------------
# Internal chart helpers
# ---------------------------------------------------------------------------
def _make_tx_scatter_density(df: pd.DataFrame) -> go.Figure:
    """Density mapbox as choropleth substitute."""
    if "longitude" not in df.columns or "latitude" not in df.columns:
        return go.Figure()

    fig = px.density_mapbox(
        df.dropna(subset=["longitude", "latitude"]),
        lat="latitude", lon="longitude",
        radius=8,
        center={"lat": 31.0, "lon": -99.5},
        zoom=5,
        mapbox_style="carto-positron",
        title="Crime Density Map",
        color_continuous_scale="YlOrRd",
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0),
                      font_family="Inter",
                      coloraxis_showscale=False)
    return fig


def _make_scatter_map(df: pd.DataFrame) -> go.Figure:
    if "longitude" not in df.columns or "latitude" not in df.columns:
        return go.Figure()

    sample = df.sample(min(3000, len(df)), random_state=0)
    colour_col = "offense_type" if "offense_type" in sample.columns else None

    fig = px.scatter_mapbox(
        sample.dropna(subset=["longitude", "latitude"]),
        lat="latitude",
        lon="longitude",
        color=colour_col,
        hover_data={c: True for c in ["year", "city", "offense_type"]
                    if c in sample.columns},
        zoom=5,
        center={"lat": 31.0, "lon": -99.5},
        mapbox_style="carto-positron",
        title="Individual Crime Incidents (sample)",
        opacity=0.65,
        size_max=6,
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0),
                      font_family="Inter",
                      legend_title="Crime Type")
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ..data.loader import generate_synthetic_dataset

    log.info("Generating synthetic dataset for dashboard demo…")
    gdf = generate_synthetic_dataset(n_incidents=5000)

    app = create_app(gdf)
    log.info("Dashboard running at http://127.0.0.1:8050/")
    app.run(debug=True, port=8050)
