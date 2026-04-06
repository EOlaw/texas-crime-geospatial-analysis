"""
map_generator.py
================
Interactive Folium map builders for Texas crime geospatial analysis.

Maps produced
-------------
1. Incident point map        – raw crime incidents coloured by type.
2. Choropleth map            – county/city-level crime counts or rates.
3. KDE hotspot overlay       – kernel density surface over base map.
4. DBSCAN cluster map        – cluster hulls + noise points.
5. Getis-Ord Gi* map         – hot/cold spot significance choropleth.
6. Risk prediction map       – model-predicted risk surface.
7. Composite dashboard map   – layered map with toggle controls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

import branca.colormap as cm
import folium
import folium.plugins as fplugins
import geopandas as gpd
import numpy as np
import pandas as pd

from ..utils import get_logger, get_output_dir

log = get_logger(__name__)

# Texas approximate centre
_TX_CENTER = [31.0, -99.5]
_TX_ZOOM   = 6

# Colour palettes
_CRIME_COLOURS = {
    "murder":              "#8B0000",
    "rape":                "#DC143C",
    "robbery":             "#FF4500",
    "aggravated assault":  "#FF6347",
    "burglary":            "#FFA500",
    "motor vehicle theft": "#FFD700",
    "drug offense":        "#9370DB",
    "simple assault":      "#FF8C00",
    "larceny-theft":       "#6495ED",
    "vandalism":           "#20B2AA",
}
_DEFAULT_COLOUR = "#607D8B"


def _crime_colour(offense_type: str) -> str:
    return _CRIME_COLOURS.get(str(offense_type).lower(), _DEFAULT_COLOUR)


# ---------------------------------------------------------------------------
# Base map factory
# ---------------------------------------------------------------------------
def base_map(center: List[float] = None,
             zoom: int = _TX_ZOOM,
             tiles: str = "CartoDB positron") -> folium.Map:
    """Create a base Folium map centred on Texas."""
    return folium.Map(location=center or _TX_CENTER,
                      zoom_start=zoom,
                      tiles=tiles)


# ---------------------------------------------------------------------------
# 1. Incident point map
# ---------------------------------------------------------------------------
def incident_point_map(
    gdf:        gpd.GeoDataFrame,
    lon_col:    str = "longitude",
    lat_col:    str = "latitude",
    type_col:   str = "offense_type",
    max_points: int = 10_000,
    out_name:   str = "incident_map",
) -> folium.Map:
    """
    Plot individual crime incidents as coloured circle markers.
    Large datasets are sub-sampled to `max_points` for browser performance.
    """
    if len(gdf) > max_points:
        log.info("Sub-sampling from %d to %d points for incident map", len(gdf), max_points)
        gdf = gdf.sample(max_points, random_state=42)

    fmap = base_map()
    fg   = folium.FeatureGroup(name="Crime Incidents", show=True)

    for _, row in gdf.iterrows():
        colour = _crime_colour(row.get(type_col, ""))
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{row.get(type_col, 'Unknown')}</b><br>"
                f"Year: {row.get('year', 'N/A')}<br>"
                f"City: {row.get('city', 'N/A')}",
                max_width=200,
            ),
            tooltip=str(row.get(type_col, "")),
        ).add_to(fg)

    fg.add_to(fmap)
    folium.LayerControl().add_to(fmap)

    _save_map(fmap, out_name)
    return fmap


# ---------------------------------------------------------------------------
# 2. Choropleth map
# ---------------------------------------------------------------------------
def choropleth_map(
    polygon_gdf: gpd.GeoDataFrame,
    value_col:   str,
    id_col:      str,
    legend_name: str = "Crime Count",
    title:       str = "Texas Crime Choropleth",
    cmap:        str = "YlOrRd",
    out_name:    str = "choropleth_map",
) -> folium.Map:
    """
    County-level choropleth map of crime counts or rates.
    """
    fmap = base_map()

    geojson = polygon_gdf[[id_col, value_col, "geometry"]].to_json()

    folium.Choropleth(
        geo_data=geojson,
        data=polygon_gdf,
        columns=[id_col, value_col],
        key_on=f"feature.properties.{id_col}",
        fill_color=cmap,
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name=legend_name,
        name=title,
    ).add_to(fmap)

    # Tooltip overlay
    style_fn  = lambda x: {"fillColor": "transparent", "color": "#333",
                            "weight": 0.5, "fillOpacity": 0}
    highlight = lambda x: {"fillColor": "#ffaf00", "color": "#000",
                            "weight": 2, "fillOpacity": 0.5}
    tooltip   = folium.GeoJsonTooltip(fields=[id_col, value_col],
                                      aliases=["Area:", f"{legend_name}:"],
                                      localize=True)

    folium.GeoJson(geojson,
                   style_function=style_fn,
                   highlight_function=highlight,
                   tooltip=tooltip,
                   name="Tooltip").add_to(fmap)

    folium.LayerControl().add_to(fmap)
    _save_map(fmap, out_name)
    return fmap


# ---------------------------------------------------------------------------
# 3. KDE heatmap overlay
# ---------------------------------------------------------------------------
def kde_heatmap_map(
    gdf:        gpd.GeoDataFrame,
    lon_col:    str   = "longitude",
    lat_col:    str   = "latitude",
    weight_col: Optional[str] = "severity",
    radius:     int   = 25,
    blur:       int   = 15,
    out_name:   str   = "kde_heatmap",
) -> folium.Map:
    """
    Browser-rendered heatmap using Folium HeatMap plugin.
    """
    fmap = base_map()

    weights = None
    if weight_col and weight_col in gdf.columns:
        weights = gdf[weight_col].fillna(1).tolist()

    heat_data = list(zip(gdf[lat_col].tolist(), gdf[lon_col].tolist(),
                         weights or [1] * len(gdf)))

    fplugins.HeatMap(
        heat_data,
        name="Crime Density Heatmap",
        min_opacity=0.3,
        max_zoom=18,
        radius=radius,
        blur=blur,
        gradient={0.2: "#0000FF", 0.4: "#00FFFF",
                  0.6: "#00FF00", 0.8: "#FFFF00", 1.0: "#FF0000"},
    ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    _save_map(fmap, out_name)
    return fmap


# ---------------------------------------------------------------------------
# 4. DBSCAN cluster map
# ---------------------------------------------------------------------------
def cluster_map(
    incidents_gdf: gpd.GeoDataFrame,
    cluster_gdf:   gpd.GeoDataFrame,
    lon_col:       str = "longitude",
    lat_col:       str = "latitude",
    label_col:     str = "cluster_id",
    out_name:      str = "cluster_map",
) -> folium.Map:
    """
    Map DBSCAN clusters: convex hull polygons + coloured incident points.
    """
    fmap = base_map()

    # Cluster hulls
    if len(cluster_gdf) > 0:
        cluster_colours = _generate_colours(len(cluster_gdf))
        hulls_fg = folium.FeatureGroup(name="Cluster Hulls", show=True)
        for idx, (_, row) in enumerate(cluster_gdf.iterrows()):
            colour = cluster_colours[idx % len(cluster_colours)]
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda x, c=colour: {
                    "fillColor": c, "color": c,
                    "weight": 2, "fillOpacity": 0.2,
                },
                tooltip=f"Cluster {row.get('cluster_id', idx)} "
                        f"(n={row.get('size', '?')})",
            ).add_to(hulls_fg)
        hulls_fg.add_to(fmap)

    # Incident points coloured by cluster
    points_fg = folium.FeatureGroup(name="Clustered Incidents", show=True)
    noise_fg  = folium.FeatureGroup(name="Noise Points",        show=False)

    n_clusters = incidents_gdf[label_col].max() + 1 if label_col in incidents_gdf.columns else 0
    colours    = _generate_colours(int(n_clusters))

    for _, row in incidents_gdf.iterrows():
        cid = int(row.get(label_col, -1))
        col = "#999999" if cid == -1 else colours[cid % len(colours)]
        marker = folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=3,
            color=col, fill=True, fill_color=col, fill_opacity=0.8,
        )
        if cid == -1:
            marker.add_to(noise_fg)
        else:
            marker.add_to(points_fg)

    points_fg.add_to(fmap)
    noise_fg.add_to(fmap)
    folium.LayerControl().add_to(fmap)
    _save_map(fmap, out_name)
    return fmap


# ---------------------------------------------------------------------------
# 5. Getis-Ord hot/cold spot map
# ---------------------------------------------------------------------------
def hotspot_map(
    polygon_gdf: gpd.GeoDataFrame,
    id_col:      str = "county_name",
    out_name:    str = "hotspot_map",
) -> folium.Map:
    """
    Visualise Getis-Ord Gi* results as a 5-class hot/cold spot map.
    """
    fmap = base_map()

    def classify(row):
        if row.get("hotspot_99", False):  return "Hot Spot 99%"
        if row.get("hotspot_95", False):  return "Hot Spot 95%"
        if row.get("hotspot_90", False):  return "Hot Spot 90%"
        if row.get("coldspot_95", False): return "Cold Spot 95%"
        return "Not Significant"

    _SPOT_COLOURS = {
        "Hot Spot 99%":    "#B2182B",
        "Hot Spot 95%":    "#EF8A62",
        "Hot Spot 90%":    "#FDDBC7",
        "Not Significant": "#F7F7F7",
        "Cold Spot 95%":   "#67A9CF",
    }

    polygon_gdf = polygon_gdf.copy()
    polygon_gdf["spot_class"] = polygon_gdf.apply(classify, axis=1)

    tooltip_fields = [f for f in [id_col, "spot_class", "gi_z_score"] if f in polygon_gdf.columns]
    tooltip_aliases = {"spot_class": "Classification:", "gi_z_score": "Z-Score:", id_col: "Area:"}

    for _, row in polygon_gdf.iterrows():
        colour = _SPOT_COLOURS.get(row["spot_class"], "#F7F7F7")
        props = {f: (float(row[f]) if isinstance(row[f], (int, float)) else str(row[f]))
                 for f in tooltip_fields}
        feature = {
            "type": "Feature",
            "geometry": row["geometry"].__geo_interface__,
            "properties": props,
        }
        folium.GeoJson(
            feature,
            style_function=lambda x, c=colour: {
                "fillColor": c, "color": "#555", "weight": 0.5, "fillOpacity": 0.75,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=[tooltip_aliases.get(f, f + ":") for f in tooltip_fields],
            ) if tooltip_fields else None,
        ).add_to(fmap)

    _add_legend(fmap, "Getis-Ord Gi*", _SPOT_COLOURS)
    folium.LayerControl().add_to(fmap)
    _save_map(fmap, out_name)
    return fmap


# ---------------------------------------------------------------------------
# 6. Risk prediction grid map
# ---------------------------------------------------------------------------
def risk_prediction_map(
    grid_gdf:    gpd.GeoDataFrame,
    value_col:   str = "predicted_count",
    out_name:    str = "risk_prediction_map",
) -> folium.Map:
    """
    Visualise continuous risk prediction as a heatmap overlay.
    """
    fmap = base_map()

    max_val = float(grid_gdf[value_col].max())
    heat_data = [
        [row["latitude"], row["longitude"],
         float(row[value_col]) / (max_val + 1e-9)]
        for _, row in grid_gdf.iterrows()
        if pd.notna(row[value_col])
    ]

    fplugins.HeatMap(
        heat_data,
        name="Predicted Crime Risk",
        min_opacity=0.2,
        radius=30,
        blur=20,
        gradient={0.2: "#00FF00", 0.5: "#FFFF00",
                  0.7: "#FF8C00", 1.0: "#FF0000"},
    ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    _save_map(fmap, out_name)
    return fmap


# ---------------------------------------------------------------------------
# 7. Composite multi-layer map
# ---------------------------------------------------------------------------
def composite_map(
    incidents_gdf:  gpd.GeoDataFrame,
    county_gdf:     gpd.GeoDataFrame,
    cluster_gdf:    Optional[gpd.GeoDataFrame] = None,
    lon_col:        str = "longitude",
    lat_col:        str = "latitude",
    count_col:      str = "offense_count",
    id_col:         str = "county_name",
    out_name:       str = "composite_map",
) -> folium.Map:
    """
    Full-featured multi-layer map combining choropleth, heatmap, clusters,
    and individual incidents with toggle controls.
    """
    fmap = base_map()

    # Layer 1: county choropleth (base)
    if count_col in county_gdf.columns:
        folium.Choropleth(
            geo_data=county_gdf[[id_col, count_col, "geometry"]].to_json(),
            data=county_gdf,
            columns=[id_col, count_col],
            key_on=f"feature.properties.{id_col}",
            fill_color="YlOrRd",
            fill_opacity=0.5,
            line_opacity=0.3,
            legend_name="Crime Count per County",
            name="County Crime Counts",
        ).add_to(fmap)

    # Layer 2: heatmap
    heat_fg = folium.FeatureGroup(name="Density Heatmap", show=True)
    weights = (incidents_gdf["severity"].fillna(1).tolist()
               if "severity" in incidents_gdf.columns
               else [1] * len(incidents_gdf))
    heat_data = list(zip(incidents_gdf[lat_col].tolist(),
                         incidents_gdf[lon_col].tolist(),
                         weights))
    fplugins.HeatMap(heat_data, radius=20, blur=12,
                     min_opacity=0.2).add_to(heat_fg)
    heat_fg.add_to(fmap)

    # Layer 3: cluster hulls
    if cluster_gdf is not None and len(cluster_gdf) > 0:
        cluster_fg = folium.FeatureGroup(name="Crime Clusters", show=True)
        for _, row in cluster_gdf.iterrows():
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda x: {
                    "fillColor": "#FF6347", "color": "#CC0000",
                    "weight": 2, "fillOpacity": 0.25,
                },
            ).add_to(cluster_fg)
        cluster_fg.add_to(fmap)

    # Layer 4: incident markers (sample)
    sample = incidents_gdf.sample(min(2000, len(incidents_gdf)), random_state=1)
    pts_fg = folium.FeatureGroup(name="Incident Points", show=False)
    for _, row in sample.iterrows():
        col = _crime_colour(row.get("offense_type", ""))
        folium.CircleMarker(
            [row[lat_col], row[lon_col]], radius=3,
            color=col, fill=True, fill_color=col, fill_opacity=0.7,
            tooltip=str(row.get("offense_type", "")),
        ).add_to(pts_fg)
    pts_fg.add_to(fmap)

    # Layer 5: city labels
    _add_city_markers(fmap)

    fplugins.MiniMap().add_to(fmap)
    fplugins.Fullscreen().add_to(fmap)
    fplugins.MousePosition().add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    _save_map(fmap, out_name)
    return fmap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_MAJOR_CITIES = {
    "Houston":       (29.760, -95.370),
    "San Antonio":   (29.424, -98.493),
    "Dallas":        (32.776, -96.797),
    "Austin":        (30.267, -97.743),
    "Fort Worth":    (32.753, -97.333),
    "El Paso":       (31.762, -106.487),
}


def _add_city_markers(fmap: folium.Map) -> None:
    fg = folium.FeatureGroup(name="Major Cities", show=True)
    for city, (lat, lon) in _MAJOR_CITIES.items():
        folium.Marker(
            [lat, lon],
            icon=folium.Icon(icon="building", prefix="fa", color="darkblue"),
            tooltip=city,
            popup=city,
        ).add_to(fg)
    fg.add_to(fmap)


def _generate_colours(n: int) -> List[str]:
    """Generate n visually distinct colours."""
    import colorsys
    colours = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.85)
        colours.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colours


def _add_legend(fmap: folium.Map, title: str, colour_map: dict) -> None:
    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:white; padding:10px; border:2px solid #ccc;
                border-radius:5px; font-size:13px;">
      <b>{title}</b><br>
    """
    for label, colour in colour_map.items():
        legend_html += (
            f'<i style="background:{colour};width:12px;height:12px;'
            f'display:inline-block;margin-right:6px;"></i>{label}<br>'
        )
    legend_html += "</div>"
    fmap.get_root().html.add_child(folium.Element(legend_html))


def _save_map(fmap: folium.Map, name: str) -> Path:
    out_dir  = get_output_dir("maps")
    out_path = out_dir / f"{name}.html"
    fmap.save(str(out_path))
    log.info("Map saved → %s", out_path)
    return out_path
