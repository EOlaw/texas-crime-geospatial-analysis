"""
spatial_clustering.py
=====================
Spatial clustering algorithms applied to crime incidents:
  1. DBSCAN  – density-based clustering (Python sklearn + C++ extension).
  2. K-Means  – centroid-based clustering (geographic crime centres).
  3. Hex-bin aggregation – hexagonal spatial binning.
  4. Ripley's K / L function – spatial point pattern statistics.

The C++ `texas_crime_spatial` extension is used for DBSCAN when available;
falls back to scikit-learn otherwise.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, MultiPoint
from sklearn.cluster import DBSCAN as SklearnDBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from ..utils import get_logger, timed

log = get_logger(__name__)

# Try to import C++ extension
try:
    import texas_crime_spatial as _tcs
    _CPP_AVAILABLE = True
    log.info("C++ texas_crime_spatial extension loaded")
except ImportError:
    _tcs = None
    _CPP_AVAILABLE = False
    log.info("C++ extension not found – using scikit-learn DBSCAN fallback")


# ---------------------------------------------------------------------------
# DBSCAN clustering
# ---------------------------------------------------------------------------
@dataclass
class DBSCANResult:
    gdf:            gpd.GeoDataFrame   # input gdf with cluster labels
    n_clusters:     int
    n_noise:        int
    cluster_gdf:    gpd.GeoDataFrame   # cluster polygons (convex hulls)
    used_cpp:       bool


@timed
def run_dbscan(
    gdf:        gpd.GeoDataFrame,
    lon_col:    str   = "longitude",
    lat_col:    str   = "latitude",
    eps_deg:    float = 0.05,          # ~5.5 km at Texas latitudes
    min_pts:    int   = 10,
    use_cpp:    bool  = True,
) -> DBSCANResult:
    """
    DBSCAN clustering of crime incident points.

    Parameters
    ----------
    gdf      : GeoDataFrame of incidents.
    lon_col  : longitude column.
    lat_col  : latitude column.
    eps_deg  : neighbourhood radius in decimal degrees.
    min_pts  : minimum cluster membership.
    use_cpp  : prefer C++ implementation when available.

    Returns
    -------
    DBSCANResult with cluster labels and convex hull polygons.
    """
    lons = gdf[lon_col].values.astype(float)
    lats = gdf[lat_col].values.astype(float)
    coords = np.column_stack([lons, lats])

    if use_cpp and _CPP_AVAILABLE and _tcs is not None:
        # Build Point2D list and call C++ DBSCAN
        pts    = [_tcs.Point2D(x=lons[i], y=lats[i], id=i) for i in range(len(lons))]
        db     = _tcs.DBSCAN(eps=eps_deg, min_pts=min_pts)
        labels = np.array(db.fit(pts))
        n_clusters = db.num_clusters
        n_noise    = db.num_noise_points
        used_cpp   = True
        log.info("C++ DBSCAN: %d clusters, %d noise", n_clusters, n_noise)
    else:
        # scikit-learn fallback
        db     = SklearnDBSCAN(eps=eps_deg, min_samples=min_pts,
                               metric="euclidean", n_jobs=-1)
        labels = db.fit_predict(coords)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = int((labels == -1).sum())
        used_cpp   = False
        log.info("sklearn DBSCAN: %d clusters, %d noise", n_clusters, n_noise)

    result_gdf = gdf.copy()
    result_gdf["cluster_id"] = labels

    # Build convex hull polygons per cluster
    hulls = []
    for cid in range(n_clusters):
        mask    = labels == cid
        pts_c   = coords[mask]
        if len(pts_c) >= 3:
            hull = MultiPoint([Point(p) for p in pts_c]).convex_hull
        else:
            hull = Point(pts_c[0]).buffer(eps_deg)
        hulls.append({
            "cluster_id": cid,
            "size":       int(mask.sum()),
            "centroid_lon": pts_c[:, 0].mean(),
            "centroid_lat": pts_c[:, 1].mean(),
            "geometry":   hull,
        })

    cluster_gdf = gpd.GeoDataFrame(hulls, crs="EPSG:4326") if hulls else gpd.GeoDataFrame()

    return DBSCANResult(
        gdf=result_gdf,
        n_clusters=n_clusters,
        n_noise=n_noise,
        cluster_gdf=cluster_gdf,
        used_cpp=used_cpp,
    )


# ---------------------------------------------------------------------------
# K-Means clustering
# ---------------------------------------------------------------------------
@dataclass
class KMeansResult:
    gdf:         gpd.GeoDataFrame
    centres:     np.ndarray    # (k, 2) centroid coordinates
    inertia:     float
    k:           int
    centre_gdf:  gpd.GeoDataFrame


@timed
def run_kmeans(
    gdf:     gpd.GeoDataFrame,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    k:       int = 8,
    seed:    int = 42,
) -> KMeansResult:
    """
    K-Means clustering to identify k major crime concentration centres.
    """
    lons = gdf[lon_col].values.astype(float)
    lats = gdf[lat_col].values.astype(float)
    coords = np.column_stack([lons, lats])

    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    km.fit(coords)

    result_gdf = gdf.copy()
    result_gdf["kmeans_cluster"] = km.labels_

    centres = km.cluster_centers_
    centre_gdf = gpd.GeoDataFrame(
        {"cluster_id": range(k),
         "lon": centres[:, 0],
         "lat": centres[:, 1],
         "geometry": [Point(c[0], c[1]) for c in centres]},
        crs="EPSG:4326",
    )

    log.info("K-Means: k=%d  inertia=%.2f", k, km.inertia_)
    return KMeansResult(
        gdf=result_gdf,
        centres=centres,
        inertia=float(km.inertia_),
        k=k,
        centre_gdf=centre_gdf,
    )


# ---------------------------------------------------------------------------
# Optimal k selection (elbow method)
# ---------------------------------------------------------------------------
def elbow_analysis(
    gdf:      gpd.GeoDataFrame,
    lon_col:  str = "longitude",
    lat_col:  str = "latitude",
    k_range:  range = range(2, 15),
) -> pd.DataFrame:
    """
    Compute K-Means inertia for a range of k values to aid cluster number selection.
    Returns a DataFrame with columns [k, inertia].
    """
    lons   = gdf[lon_col].values.astype(float)
    lats   = gdf[lat_col].values.astype(float)
    coords = np.column_stack([lons, lats])

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(coords)
        rows.append({"k": k, "inertia": km.inertia_})
        log.info("  k=%2d  inertia=%.1f", k, km.inertia_)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hexagonal binning
# ---------------------------------------------------------------------------
@dataclass
class HexBinResult:
    hex_gdf:   gpd.GeoDataFrame   # hexagonal cells with counts
    cell_size: float


@timed
def compute_hexbins(
    gdf:       gpd.GeoDataFrame,
    lon_col:   str   = "longitude",
    lat_col:   str   = "latitude",
    cell_size: float = 0.25,          # degrees (~28 km)
) -> HexBinResult:
    """
    Aggregate incidents into hexagonal bins using H3-style offset grids.
    (Pure NumPy implementation; no H3 library required.)

    Returns a GeoDataFrame of hexagonal polygons with incident counts.
    """
    from shapely.geometry import Polygon

    lons = gdf[lon_col].values.astype(float)
    lats = gdf[lat_col].values.astype(float)

    # Hex geometry helpers
    w  = cell_size
    h  = cell_size * np.sqrt(3) / 2

    def hex_vertices(cx, cy):
        """Return the 6 vertices of a flat-topped hexagon."""
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        xs = cx + (w / 2) * np.cos(angles)
        ys = cy + (h / np.sqrt(3)) * np.sin(angles)
        return list(zip(xs, ys))

    # Assign each point to a hex centre
    col = np.round(lons / w).astype(int)
    row = np.round(lats / h).astype(int)
    hex_ids = list(zip(col, row))

    from collections import Counter
    counts = Counter(hex_ids)

    cells = []
    for (ci, ri), cnt in counts.items():
        cx = ci * w + (0.5 * w if ri % 2 else 0)
        cy = ri * h
        verts = hex_vertices(cx, cy)
        cells.append({
            "hex_col": ci,
            "hex_row": ri,
            "count":   cnt,
            "geometry": Polygon(verts),
        })

    hex_gdf = gpd.GeoDataFrame(cells, crs="EPSG:4326")
    log.info("Hexbin: %d cells  cell_size=%.3f°", len(hex_gdf), cell_size)
    return HexBinResult(hex_gdf=hex_gdf, cell_size=cell_size)


# ---------------------------------------------------------------------------
# Ripley's K function
# ---------------------------------------------------------------------------
def ripleys_k(
    gdf:       gpd.GeoDataFrame,
    lon_col:   str          = "longitude",
    lat_col:   str          = "latitude",
    r_values:  np.ndarray  = None,
) -> pd.DataFrame:
    """
    Empirical Ripley's K function to characterise spatial point pattern.

    Returns a DataFrame with columns [r, K, L, L_minus_r].
    L > r  → clustering at scale r
    L < r  → regularity at scale r
    """
    lons = gdf[lon_col].values.astype(float)
    lats = gdf[lat_col].values.astype(float)
    n    = len(lons)

    if r_values is None:
        r_values = np.linspace(0.01, 2.0, 50)

    # Study area estimate
    area = ((lons.max() - lons.min()) * (lats.max() - lats.min()))
    lam  = n / area   # intensity (points per unit area)

    # Compute pairwise distances (vectorised for moderate n)
    dx = lons[:, None] - lons[None, :]
    dy = lats[:, None] - lats[None, :]
    D  = np.sqrt(dx ** 2 + dy ** 2)

    rows = []
    for r in r_values:
        # K(r) = (1 / λ) * Σ_{i≠j} I(d_ij ≤ r) / n
        K  = float(((D <= r) & (D > 0)).sum()) / (lam * n)
        L  = np.sqrt(K / np.pi)
        rows.append({"r": r, "K": K, "L": L, "L_minus_r": L - r})

    df = pd.DataFrame(rows)
    log.info("Ripley's K computed for %d distance values", len(r_values))
    return df
