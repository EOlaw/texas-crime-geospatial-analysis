# Python API Reference

## `src.python.data`

### `loader`

#### `generate_synthetic_dataset(n_incidents, seed) → GeoDataFrame`
Generate a synthetic Texas crime incident dataset (no download required).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_incidents` | int | 5000 | Number of incidents to generate |
| `seed` | int | 42 | RNG seed for reproducibility |

Returns a `GeoDataFrame` (EPSG:4326) with columns: `longitude`, `latitude`,
`offense_type`, `severity`, `year`, `city`, `incident_id`, `geometry`.

---

#### `load_incident_csv(path, lon_col, lat_col) → GeoDataFrame`
Load a CSV file of incident locations into a GeoDataFrame.

Auto-detects `longitude`/`latitude` columns if not specified.

---

#### `load_county_shapefile(path, texas_only) → GeoDataFrame`
Load TIGER/Line county boundaries. Filters to Texas (FIPS=48) by default.

---

### `preprocessor`

#### `clean_incident_gdf(gdf, lon_col, lat_col, year_min, year_max) → GeoDataFrame`
Full cleaning pipeline:
1. Drop null / empty geometries.
2. Clip to Texas bounding box.
3. Drop duplicates.
4. Normalise `offense_type` strings.
5. Filter year range.

---

#### `add_severity(gdf) → GeoDataFrame`
Map `offense_type` to a numeric severity score [1–10].

| Crime Type | Severity |
|---|---|
| Murder | 10 |
| Rape | 9 |
| Robbery | 8 |
| Aggravated Assault | 7 |
| Burglary | 6 |
| Motor Vehicle Theft / Drug | 5 |
| Simple Assault | 4 |
| Larceny-Theft / Fraud | 3 |
| Vandalism | 2 |

---

#### `add_crime_category(gdf) → GeoDataFrame`
Add `crime_category` column: `'violent'`, `'property'`, or `'other'`.

---

#### `attach_county(incidents, counties) → GeoDataFrame`
Spatial join to attach county name and GEOID to each incident point.

---

### `fetcher`

#### `fetch_all(app_token, fbi_api_key) → dict[str, Path]`
Download all data sources and return file paths. Failures are logged as
warnings (not exceptions) so partial downloads still succeed.

---

## `src.python.analysis`

### `hotspot_detection`

#### `compute_kde(gdf, lon_col, lat_col, grid_size, bandwidth, hotspot_pct) → KDEResult`

| Field | Type | Description |
|---|---|---|
| `lon_grid` | ndarray (grid_size, grid_size) | Longitude grid |
| `lat_grid` | ndarray (grid_size, grid_size) | Latitude grid |
| `density` | ndarray (grid_size, grid_size) | KDE values |
| `bandwidth` | float | Kernel bandwidth used |
| `hotspot_mask` | ndarray[bool] | True = hotspot cell |

---

#### `compute_getis_ord(polygon_gdf, count_col, significance, weights_type) → GetisOrdResult`

| Field | Type | Description |
|---|---|---|
| `gdf` | GeoDataFrame | Input polygons + Gi* columns |
| `hotspot_gdf` | GeoDataFrame | Polygons with p < 0.05 hot spots |
| `coldspot_gdf` | GeoDataFrame | Polygons with p < 0.05 cold spots |
| `global_summary` | dict | Counts and z-score range |

Added columns: `gi_z_score`, `gi_p_value`, `hotspot_90/95/99`, `coldspot_95`.

---

#### `compute_lisa(polygon_gdf, count_col) → LISAResult`

Added columns: `lisa_I`, `lisa_p`, `lisa_sig`, `lisa_quadrant` (HH/LL/HL/LH).

---

### `spatial_clustering`

#### `run_dbscan(gdf, eps_deg, min_pts, use_cpp) → DBSCANResult`

| Field | Type | Description |
|---|---|---|
| `gdf` | GeoDataFrame | Input + `cluster_id` column |
| `n_clusters` | int | Number of clusters found |
| `n_noise` | int | Noise point count |
| `cluster_gdf` | GeoDataFrame | Convex hull polygons per cluster |
| `used_cpp` | bool | True if C++ backend was used |

---

#### `run_kmeans(gdf, k, seed) → KMeansResult`

| Field | Type | Description |
|---|---|---|
| `centres` | ndarray (k, 2) | Cluster centroids [lon, lat] |
| `inertia` | float | Within-cluster sum of squares |
| `centre_gdf` | GeoDataFrame | Centroid point GeoDataFrame |

---

#### `ripleys_k(gdf, r_values) → DataFrame`
Returns columns: `r`, `K`, `L`, `L_minus_r`.

---

### `statistical_analysis`

#### `global_morans_i(polygon_gdf, count_col) → MoranResult`

| Field | Type | Description |
|---|---|---|
| `I` | float | Moran's I statistic |
| `z_score` | float | Z-score under normality |
| `p_value` | float | Two-tailed p-value |
| `is_clustered` | bool | True if p < 0.05 and I > E[I] |

---

#### `temporal_trend(df, year_col, count_col) → TrendResult`

| Field | Type | Description |
|---|---|---|
| `df` | DataFrame | year, count, pct_change, rolling_avg |
| `trend_direction` | str | 'increasing' / 'decreasing' / 'no trend' |
| `mk_p_value` | float | Mann-Kendall test p-value |

---

### `predictive_model`

#### `build_feature_matrix(polygon_gdf, count_col, extra_features, include_spatial_lag) → (X, y, feature_names)`

#### `train_random_forest(X, y, feature_names, n_estimators, cv_folds) → ModelResult`

#### `train_gradient_boosting(X, y, feature_names, n_estimators, learning_rate) → ModelResult`

| Field | Type | Description |
|---|---|---|
| `cv_rmse_mean` | float | Mean cross-validation RMSE |
| `cv_r2_mean` | float | Mean cross-validation R² |
| `test_metrics` | dict | rmse, mae, r2 on training data |
| `feature_importances` | Series | Sorted feature importance scores |

#### `predict_risk_grid(model_result, study_area, grid_size) → GeoDataFrame`

---

## `src.python.visualization`

### `map_generator`

| Function | Output file | Description |
|---|---|---|
| `incident_point_map(gdf)` | `incident_map.html` | Coloured circle markers per incident |
| `choropleth_map(gdf, val, id)` | `choropleth_map.html` | Polygon choropleth |
| `kde_heatmap_map(gdf)` | `kde_heatmap.html` | Browser-rendered heatmap |
| `cluster_map(incidents, clusters)` | `cluster_map.html` | DBSCAN hulls + points |
| `hotspot_map(polygon_gdf)` | `hotspot_map.html` | Gi* hot/cold spot colours |
| `risk_prediction_map(grid_gdf)` | `risk_prediction_map.html` | Predicted risk heatmap |
| `composite_map(incidents, counties)` | `composite_map.html` | All layers combined |

All maps are saved to `outputs/maps/` and also returned as `folium.Map` objects.

### `heatmap` (static figures)

All static figure functions accept `save=True/False` and return `matplotlib.Figure`.
Saved to `outputs/figures/{fname}.png`.

### `dashboard.create_app(gdf, county_gdf, port) → Dash`
Returns a configured Plotly Dash application. Call `.run(debug=True)` to serve.
