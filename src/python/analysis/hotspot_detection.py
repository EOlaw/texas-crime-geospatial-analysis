"""
hotspot_detection.py
====================
Spatial hotspot detection methods:
  1. Kernel Density Estimation (KDE) – continuous crime density surface.
  2. Getis-Ord Gi* statistic      – local spatial autocorrelation hotspots.
  3. Local Moran's I (LISA)       – local indicators of spatial association.
  4. Quadrat analysis             – grid-based density counts.

All functions return a result object or GeoDataFrame that can be passed
directly to the visualization layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
from esda.getisord import G_Local
from esda.moran import Moran, Moran_Local
from scipy.stats import gaussian_kde
from shapely.geometry import Point, box

from ..utils import get_logger, timed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# KDE Hotspot Surface
# ---------------------------------------------------------------------------
@dataclass
class KDEResult:
    """Output of kernel density estimation."""
    lon_grid:     np.ndarray   # 2-D grid of longitude values
    lat_grid:     np.ndarray   # 2-D grid of latitude values
    density:      np.ndarray   # 2-D density surface
    bandwidth:    float        # KDE bandwidth used
    n_points:     int          # number of input points
    hotspot_threshold: float   # density percentile used for hotspot mask
    hotspot_mask: np.ndarray   # boolean mask: True = hotspot cell


@timed
def compute_kde(
    gdf:          gpd.GeoDataFrame,
    lon_col:      str   = "longitude",
    lat_col:      str   = "latitude",
    grid_size:    int   = 300,
    bandwidth:    Optional[float] = None,
    hotspot_pct:  float = 90.0,
) -> KDEResult:
    """
    Compute a Kernel Density Estimation surface over crime incident locations.

    Parameters
    ----------
    gdf         : GeoDataFrame with crime incidents.
    lon_col     : longitude column name.
    lat_col     : latitude column name.
    grid_size   : number of grid cells per axis.
    bandwidth   : KDE bandwidth (Scott's rule if None).
    hotspot_pct : percentile above which a cell is classified as a hotspot.

    Returns
    -------
    KDEResult with the density surface and hotspot mask.
    """
    lons = gdf[lon_col].values.astype(float)
    lats = gdf[lat_col].values.astype(float)

    # Bounding box
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    # Add 5% margin
    lon_pad = (lon_max - lon_min) * 0.05
    lat_pad = (lat_max - lat_min) * 0.05
    lon_min -= lon_pad; lon_max += lon_pad
    lat_min -= lat_pad; lat_max += lat_pad

    lon_grid, lat_grid = np.meshgrid(
        np.linspace(lon_min, lon_max, grid_size),
        np.linspace(lat_min, lat_max, grid_size),
    )

    xy_grid  = np.vstack([lon_grid.ravel(), lat_grid.ravel()])
    xy_data  = np.vstack([lons, lats])

    kde = gaussian_kde(xy_data, bw_method=bandwidth)
    density = kde(xy_grid).reshape(grid_size, grid_size)

    threshold     = np.percentile(density, hotspot_pct)
    hotspot_mask  = density >= threshold

    log.info("KDE computed on %d points | bandwidth=%.4f | hotspots: %d cells",
             len(lons), kde.factor, hotspot_mask.sum())

    return KDEResult(
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        density=density,
        bandwidth=float(kde.factor),
        n_points=len(lons),
        hotspot_threshold=float(threshold),
        hotspot_mask=hotspot_mask,
    )


def kde_to_geodataframe(result: KDEResult) -> gpd.GeoDataFrame:
    """Convert a KDEResult grid to a point GeoDataFrame for mapping."""
    lons = result.lon_grid.ravel()
    lats = result.lat_grid.ravel()
    dens = result.density.ravel()
    mask = result.hotspot_mask.ravel()

    gdf = gpd.GeoDataFrame({
        "density":    dens,
        "is_hotspot": mask,
        "geometry":   [Point(x, y) for x, y in zip(lons, lats)],
    }, crs="EPSG:4326")
    return gdf


# ---------------------------------------------------------------------------
# Getis-Ord Gi* statistic
# ---------------------------------------------------------------------------
@dataclass
class GetisOrdResult:
    """Output of Getis-Ord Gi* analysis on polygon/raster units."""
    gdf:            gpd.GeoDataFrame   # input polygons augmented with Gi* stats
    hotspot_gdf:    gpd.GeoDataFrame   # significant hot spots  (z > 1.96)
    coldspot_gdf:   gpd.GeoDataFrame   # significant cold spots (z < -1.96)
    global_summary: dict


@timed
def compute_getis_ord(
    polygon_gdf:     gpd.GeoDataFrame,
    count_col:       str   = "offense_count",
    significance:    float = 0.05,
    weights_type:    str   = "queen",
) -> GetisOrdResult:
    """
    Compute the Getis-Ord Gi* local statistic for spatial hotspot detection
    on polygon units (counties / hexbins / grid cells).

    Parameters
    ----------
    polygon_gdf  : GeoDataFrame of polygons with a crime count column.
    count_col    : column with crime counts.
    significance : p-value threshold for significance.
    weights_type : 'queen' or 'rook' contiguity weights.

    Returns
    -------
    GetisOrdResult with enriched GeoDataFrame.
    """
    gdf = polygon_gdf.copy()

    if count_col not in gdf.columns:
        raise ValueError(f"Column '{count_col}' not found in GeoDataFrame")

    # Build spatial weights
    if weights_type == "queen":
        w = libpysal.weights.Queen.from_dataframe(gdf, silence_warnings=True, use_index=False)
    else:
        w = libpysal.weights.Rook.from_dataframe(gdf, silence_warnings=True, use_index=False)

    # Fill diagonal with 1 before row-standardising so Gi* has a defined self-weight
    libpysal.weights.fill_diagonal(w, 1)
    w.transform = "r"   # row-standardise

    gi_star = G_Local(gdf[count_col].values, w, transform="R", permutations=999, star=None)

    gdf["gi_z_score"] = gi_star.Zs
    gdf["gi_p_value"] = gi_star.p_sim
    gdf["hotspot_90"] = (gi_star.Zs > 1.645)  & (gi_star.p_sim < 0.10)
    gdf["hotspot_95"] = (gi_star.Zs > 1.960)  & (gi_star.p_sim < 0.05)
    gdf["hotspot_99"] = (gi_star.Zs > 2.576)  & (gi_star.p_sim < 0.01)
    gdf["coldspot_95"] = (gi_star.Zs < -1.960) & (gi_star.p_sim < 0.05)

    hotspot_gdf  = gdf[gdf["hotspot_95"]].copy()
    coldspot_gdf = gdf[gdf["coldspot_95"]].copy()

    global_summary = {
        "n_polygons":     len(gdf),
        "n_hotspots_95":  int(gdf["hotspot_95"].sum()),
        "n_coldspots_95": int(gdf["coldspot_95"].sum()),
        "max_z":          float(gi_star.Zs.max()),
        "min_z":          float(gi_star.Zs.min()),
    }

    log.info("Getis-Ord Gi*: %d hotspots, %d coldspots (p<0.05)",
             global_summary["n_hotspots_95"], global_summary["n_coldspots_95"])

    return GetisOrdResult(
        gdf=gdf,
        hotspot_gdf=hotspot_gdf,
        coldspot_gdf=coldspot_gdf,
        global_summary=global_summary,
    )


# ---------------------------------------------------------------------------
# Local Moran's I (LISA)
# ---------------------------------------------------------------------------
@dataclass
class LISAResult:
    gdf:          gpd.GeoDataFrame
    global_moran: float
    global_p:     float


@timed
def compute_lisa(
    polygon_gdf: gpd.GeoDataFrame,
    count_col:   str = "offense_count",
) -> LISAResult:
    """
    Compute Local Moran's I (LISA) for spatial autocorrelation.

    Cluster types (quadrant):
        HH = High-High (hotspot)
        LL = Low-Low  (coldspot)
        HL = High surrounded by Low (spatial outlier)
        LH = Low surrounded by High (spatial outlier)
    """
    gdf = polygon_gdf.copy()
    w   = libpysal.weights.Queen.from_dataframe(gdf, silence_warnings=True, use_index=False)
    w.transform = "r"

    mi = Moran(gdf[count_col].values, w)
    li = Moran_Local(gdf[count_col].values, w, permutations=999)

    gdf["lisa_I"]       = li.Is
    gdf["lisa_p"]       = li.p_sim
    gdf["lisa_sig"]     = li.p_sim < 0.05
    gdf["lisa_quadrant"] = np.where(
        li.p_sim >= 0.05, "Not Significant",
        np.where(li.q == 1, "HH",
        np.where(li.q == 2, "LH",
        np.where(li.q == 3, "LL", "HL")))
    )

    log.info("Global Moran's I = %.4f  (p=%.4f)", mi.I, mi.p_sim)
    return LISAResult(gdf=gdf, global_moran=float(mi.I), global_p=float(mi.p_sim))


# ---------------------------------------------------------------------------
# Quadrat analysis
# ---------------------------------------------------------------------------
@dataclass
class QuadratResult:
    grid_gdf:   gpd.GeoDataFrame   # grid cells with counts
    chi2_stat:  float
    chi2_p:     float
    vmr:        float               # variance-to-mean ratio (>1 = clustered)


@timed
def compute_quadrat_analysis(
    gdf:        gpd.GeoDataFrame,
    lon_col:    str = "longitude",
    lat_col:    str = "latitude",
    n_cols:     int = 20,
    n_rows:     int = 20,
) -> QuadratResult:
    """
    Partition the study area into a regular grid and count incidents per cell.
    Computes chi-squared test against CSR (complete spatial randomness).
    """
    from scipy.stats import chisquare

    lons = gdf[lon_col].values.astype(float)
    lats = gdf[lat_col].values.astype(float)

    lon_edges = np.linspace(lons.min(), lons.max(), n_cols + 1)
    lat_edges = np.linspace(lats.min(), lats.max(), n_rows + 1)

    counts, _, _ = np.histogram2d(lons, lats,
                                  bins=[lon_edges, lat_edges])

    # Build grid polygons
    cells = []
    for i in range(n_cols):
        for j in range(n_rows):
            cell_box = box(lon_edges[i], lat_edges[j],
                           lon_edges[i+1], lat_edges[j+1])
            cells.append({
                "geometry":  cell_box,
                "col":       i,
                "row":       j,
                "count":     int(counts[i, j]),
            })

    grid_gdf = gpd.GeoDataFrame(cells, crs="EPSG:4326")

    # Chi-squared test for CSR
    observed = counts.ravel()
    expected = np.full_like(observed, observed.mean(), dtype=float)
    chi2_stat, chi2_p = chisquare(observed + 1e-9, f_exp=expected + 1e-9)

    vmr = (observed.var() / observed.mean()) if observed.mean() > 0 else 0.0

    log.info("Quadrat analysis: χ²=%.2f  p=%.4f  VMR=%.2f", chi2_stat, chi2_p, vmr)
    return QuadratResult(grid_gdf=grid_gdf,
                         chi2_stat=float(chi2_stat),
                         chi2_p=float(chi2_p),
                         vmr=float(vmr))
