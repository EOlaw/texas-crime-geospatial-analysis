"""
heatmap.py
==========
Static matplotlib / seaborn visualisations for crime analysis.
All functions return a matplotlib Figure and optionally save to /outputs/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from ..analysis.hotspot_detection import KDEResult
from ..utils import get_logger, get_output_dir

log = get_logger(__name__)
sns.set_theme(style="whitegrid", palette="muted")

_CMAP_CRIME = LinearSegmentedColormap.from_list(
    "crime",
    ["#FFFFFF", "#FFF3E0", "#FFCC80", "#FF7043", "#B71C1C"],
)


# ---------------------------------------------------------------------------
# KDE density surface plot
# ---------------------------------------------------------------------------
def plot_kde_surface(
    result:      KDEResult,
    county_gdf:  Optional[gpd.GeoDataFrame] = None,
    title:       str = "Crime Density Surface – Texas",
    figsize:     Tuple[int, int] = (12, 9),
    save:        bool = True,
    fname:       str  = "kde_surface",
) -> plt.Figure:
    """
    Plot the KDE density surface as a filled contour with county boundaries.
    """
    fig, ax = plt.subplots(figsize=figsize)

    cf = ax.contourf(
        result.lon_grid, result.lat_grid, result.density,
        levels=20, cmap=_CMAP_CRIME, alpha=0.85,
    )
    plt.colorbar(cf, ax=ax, label="Crime Density (relative)")

    if county_gdf is not None:
        county_gdf.boundary.plot(ax=ax, linewidth=0.5, color="#555555", alpha=0.6)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    _decorate_texas_axes(ax)
    plt.tight_layout()
    _save_fig(fig, fname, save)
    return fig


# ---------------------------------------------------------------------------
# Crime count bar chart by type
# ---------------------------------------------------------------------------
def plot_crime_type_bar(
    gdf:       gpd.GeoDataFrame,
    type_col:  str   = "offense_type",
    top_n:     int   = 10,
    title:     str   = "Top Crime Types in Texas",
    figsize:   Tuple = (12, 6),
    save:      bool  = True,
    fname:     str   = "crime_type_bar",
) -> plt.Figure:
    """Horizontal bar chart of the most frequent crime types."""
    counts = gdf[type_col].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    colours = sns.color_palette("YlOrRd", len(counts))[::-1]

    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colours)

    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + counts.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Incidents")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    _save_fig(fig, fname, save)
    return fig


# ---------------------------------------------------------------------------
# Temporal trend line
# ---------------------------------------------------------------------------
def plot_temporal_trend(
    trend_df:  pd.DataFrame,
    title:     str   = "Annual Crime Trend – Texas",
    figsize:   Tuple = (11, 5),
    save:      bool  = True,
    fname:     str   = "temporal_trend",
) -> plt.Figure:
    """Line plot of crime counts over years with rolling average."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(trend_df["year"], trend_df["count"],
           color="#FFCC80", alpha=0.7, label="Annual Count")

    if "rolling_avg" in trend_df.columns:
        ax.plot(trend_df["year"], trend_df["rolling_avg"],
                color="#B71C1C", linewidth=2.5, label="Rolling Average", marker="o")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Incidents")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    _save_fig(fig, fname, save)
    return fig


# ---------------------------------------------------------------------------
# Choropleth static map
# ---------------------------------------------------------------------------
def plot_county_choropleth(
    county_gdf:  gpd.GeoDataFrame,
    value_col:   str   = "offense_count",
    title:       str   = "Crime Count by County",
    cmap:        str   = "YlOrRd",
    figsize:     Tuple = (14, 10),
    save:        bool  = True,
    fname:       str   = "county_choropleth",
) -> plt.Figure:
    """Static choropleth using GeoPandas plot method."""
    fig, ax = plt.subplots(figsize=figsize)

    county_gdf.plot(
        column=value_col,
        cmap=cmap,
        linewidth=0.5,
        edgecolor="#444444",
        legend=True,
        legend_kwds={"label": value_col.replace("_", " ").title(),
                     "orientation": "vertical"},
        missing_kwds={"color": "#CCCCCC", "label": "No data"},
        ax=ax,
    )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    _save_fig(fig, fname, save)
    return fig


# ---------------------------------------------------------------------------
# DBSCAN cluster scatter
# ---------------------------------------------------------------------------
def plot_cluster_scatter(
    gdf:      gpd.GeoDataFrame,
    lon_col:  str = "longitude",
    lat_col:  str = "latitude",
    label_col: str = "cluster_id",
    title:    str = "DBSCAN Spatial Clusters",
    figsize:  Tuple = (12, 9),
    save:     bool = True,
    fname:    str  = "cluster_scatter",
) -> plt.Figure:
    """2-D scatter plot coloured by cluster label."""
    fig, ax = plt.subplots(figsize=figsize)

    labels    = gdf[label_col].values
    unique_l  = sorted(set(labels))
    cmap      = plt.get_cmap("tab20")

    for cid in unique_l:
        mask   = labels == cid
        colour = "#AAAAAA" if cid == -1 else cmap(cid % 20)
        label  = "Noise" if cid == -1 else f"Cluster {cid}"
        ax.scatter(
            gdf.loc[mask, lon_col], gdf.loc[mask, lat_col],
            s=5, c=[colour], alpha=0.6, label=label,
        )

    if len(unique_l) <= 15:
        ax.legend(markerscale=3, loc="upper right", fontsize=8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    _save_fig(fig, fname, save)
    return fig


# ---------------------------------------------------------------------------
# Feature importance bar chart
# ---------------------------------------------------------------------------
def plot_feature_importance(
    importances: pd.Series,
    title:       str   = "Random Forest Feature Importances",
    figsize:     Tuple = (9, 5),
    save:        bool  = True,
    fname:       str   = "feature_importance",
) -> plt.Figure:
    """Horizontal bar chart of model feature importances."""
    fig, ax = plt.subplots(figsize=figsize)
    imp = importances.sort_values(ascending=True)

    colours = sns.color_palette("Blues_d", len(imp))
    ax.barh(imp.index, imp.values, color=colours)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    _save_fig(fig, fname, save)
    return fig


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(
    df:      pd.DataFrame,
    cols:    Optional[List[str]] = None,
    title:   str   = "Crime Variable Correlation Matrix",
    figsize: Tuple = (10, 8),
    save:    bool  = True,
    fname:   str   = "correlation_heatmap",
) -> plt.Figure:
    """Seaborn heatmap of the Pearson correlation matrix."""
    if cols:
        df = df[cols]

    corr = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, fname, save)
    return fig


# ---------------------------------------------------------------------------
# Ripley's K / L function plot
# ---------------------------------------------------------------------------
def plot_ripleys_l(
    ripleys_df: pd.DataFrame,
    title:      str   = "Ripley's L Function – Texas Crime Patterns",
    figsize:    Tuple = (9, 5),
    save:       bool  = True,
    fname:      str   = "ripleys_l",
) -> plt.Figure:
    """Plot L(r) − r to assess clustering vs. regularity."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(ripleys_df["r"], ripleys_df["L_minus_r"],
            color="#B71C1C", linewidth=2, label="L(r) − r (observed)")
    ax.axhline(0, color="#555", linestyle="--", linewidth=1, label="CSR (expected)")
    ax.fill_between(ripleys_df["r"], ripleys_df["L_minus_r"], 0,
                    where=ripleys_df["L_minus_r"] > 0,
                    alpha=0.2, color="#FF7043", label="Clustering region")
    ax.fill_between(ripleys_df["r"], ripleys_df["L_minus_r"], 0,
                    where=ripleys_df["L_minus_r"] < 0,
                    alpha=0.2, color="#42A5F5", label="Regularity region")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Distance r (degrees)")
    ax.set_ylabel("L(r) − r")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, fname, save)
    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _decorate_texas_axes(ax: plt.Axes) -> None:
    ax.set_xlim(-106.8, -93.3)
    ax.set_ylim(25.6, 36.6)
    ax.set_aspect("equal")


def _save_fig(fig: plt.Figure, fname: str, save: bool) -> Optional[Path]:
    if not save:
        return None
    out_dir  = get_output_dir("figures")
    out_path = out_dir / f"{fname}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    log.info("Figure saved → %s", out_path)
    return out_path
