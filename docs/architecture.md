# Architecture

## System Overview

```
texas-crime-geospatial-analysis/
│
├── main.py                         ← Pipeline orchestrator (CLI)
│
├── src/
│   ├── python/                     ← Python package
│   │   ├── data/                   ← Layer 1: Data Acquisition
│   │   │   ├── fetcher.py          │   HTTP downloads (Socrata, FBI, Census)
│   │   │   ├── loader.py           │   Reads CSV / JSON / Shapefile / Synthetic
│   │   │   └── preprocessor.py     │   Clean, validate, feature engineer
│   │   │
│   │   ├── analysis/               ← Layer 2: Spatial Analysis
│   │   │   ├── hotspot_detection.py│   KDE, Getis-Ord Gi*, LISA, Quadrat
│   │   │   ├── spatial_clustering.py│  DBSCAN, K-Means, Hexbins, Ripley's K
│   │   │   ├── statistical_analysis.py│ Moran's I, correlation, trend, risk
│   │   │   └── predictive_model.py │   RF, GBR, prediction grid, CV
│   │   │
│   │   ├── visualization/          ← Layer 3: Visualisation
│   │   │   ├── map_generator.py    │   7 Folium interactive maps
│   │   │   ├── heatmap.py          │   8 matplotlib/seaborn static figures
│   │   │   └── dashboard.py        │   Plotly Dash web application
│   │   │
│   │   └── utils/                  ← Cross-cutting utilities
│   │       ├── config.py           │   YAML settings loader
│   │       └── helpers.py          │   Logging, coord utils, scalers
│   │
│   └── cpp/                        ← C++ performance backend
│       ├── spatial_index/
│       │   ├── kdtree.h/.cpp       │   2-D KD-Tree (O(log n) queries)
│       ├── clustering/
│       │   ├── dbscan.h/.cpp       │   DBSCAN (KD-Tree accelerated)
│       └── bindings/
│           └── spatial_ext.cpp     │   pybind11 Python bindings
│
├── data/                           ← Data store (git-ignored)
│   ├── raw/                        │   Downloaded raw files
│   ├── processed/                  │   GeoParquet output
│   └── shapefiles/                 │   TIGER/Line .shp files
│
├── outputs/                        ← Generated outputs (git-ignored)
│   ├── maps/                       │   HTML interactive maps
│   ├── figures/                    │   PNG static figures
│   └── reports/                    │   Joblib model files
│
├── tests/
│   ├── cpp/                        │   C++ unit tests (Google-Test-free)
│   ├── test_data.py
│   ├── test_analysis.py
│   └── test_visualization.py
│
├── scripts/                        ← Standalone helper scripts
├── notebooks/                      ← Jupyter exploration notebooks
├── config/settings.yaml            ← Central configuration
├── CMakeLists.txt                  ← C++ build definition
├── setup.py                        ← Python package + C++ extension
└── requirements.txt
```

---

## Data Flow

```
External APIs / CSV
        │
        ▼
 [fetcher.py]          HTTP GET → data/raw/
        │
        ▼
 [loader.py]           Raw CSV/JSON → GeoDataFrame (EPSG:4326)
        │
        ▼
 [preprocessor.py]     Clean → Validate → Spatial Join → Feature Eng.
        │
        ├─────────────────────────────────────────────────────┐
        ▼                                                     ▼
 [hotspot_detection]                               [spatial_clustering]
 KDE · Gi* · LISA · Quadrat                DBSCAN · K-Means · Hexbin · Ripley
        │                                                     │
        └───────────────────┬─────────────────────────────────┘
                            │
                            ▼
                  [statistical_analysis]
                  Moran's I · Trend · Risk Score
                            │
                            ▼
                  [predictive_model]
                  Feature Matrix → RF/GBR → Prediction Grid
                            │
                            ▼
                  [visualization]
                  Folium Maps · Dash Dashboard · matplotlib
                            │
                            ▼
                   outputs/{maps,figures,reports}/
```

---

## C++ / Python Integration

```
Python (sklearn / scipy fallback)
         │
         │  import texas_crime_spatial   ←── pybind11 module
         │
         ▼
[spatial_ext.cpp]  ← pybind11 bindings
    │          │
    ▼          ▼
[kdtree.cpp] [dbscan.cpp]
  KD-Tree      DBSCAN
  O(log n)     density-based
  kNN + range  clustering
```

The Python code imports `texas_crime_spatial` at module load. If not
found (C++ not built), a `_CPP_AVAILABLE = False` flag is set and all
calls transparently fall back to `sklearn.cluster.DBSCAN`.

---

## Threading & Performance

| Operation | Implementation | Notes |
|---|---|---|
| KD-Tree build | Single thread, C++ | O(n log n) |
| kNN query | Single thread, C++ | O(k log n) |
| DBSCAN | Single thread, C++ | O(n log n) with KD-Tree |
| Random Forest | Multi-thread | `n_jobs=-1` (all cores) |
| KDE surface | Single thread, Python | Grid of 300² = 90 000 cells |
| Folium map render | Browser-side JS | |

For very large datasets (> 1M incidents), consider:
- Using projected coordinates (EPSG:3857) instead of lon/lat degrees.
- Tiling KDE computation over spatial subdomains.
- Enabling Dask for parallelised pandas operations.

---

## Extension Points

| Feature | Where to add |
|---|---|
| New crime data source | `src/python/data/fetcher.py` |
| New spatial statistic | `src/python/analysis/statistical_analysis.py` |
| New map layer | `src/python/visualization/map_generator.py` |
| New ML model | `src/python/analysis/predictive_model.py` |
| New C++ algorithm | `src/cpp/` + register in `spatial_ext.cpp` |
