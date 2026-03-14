# Texas Crime Geospatial Analysis

> **Geospatial Crime Pattern Analysis & Predictive Mapping**
> Python · C++17 · GeoPandas · Folium · Plotly Dash · PySAL · Scikit-Learn

---

## Overview

A full end-to-end spatial crime intelligence platform for the state of
Texas. The pipeline ingests public crime statistics, performs rigorous
spatial analysis, trains predictive risk models, and produces interactive
web maps and an analytics dashboard.

### What This Project Does

| Step | Methods |
|---|---|
| **Data Acquisition** | Texas DPS UCR (Socrata API), FBI Crime Data Explorer, US Census TIGER/Line shapefiles |
| **Preprocessing** | Coordinate validation, deduplication, severity scoring, spatial joins |
| **Hotspot Detection** | Kernel Density Estimation, Getis-Ord Gi*, Local Moran's I (LISA), Quadrat analysis |
| **Spatial Clustering** | DBSCAN (C++ KD-Tree accelerated), K-Means, Hexagonal binning, Ripley's K function |
| **Statistical Analysis** | Global Moran's I, bivariate correlation, Mann-Kendall trend test, crime rate normalisation |
| **Predictive Modeling** | Random Forest & Gradient Boosting regressors, risk prediction grid |
| **Visualisation** | 7 interactive Folium maps, 8 static matplotlib/seaborn figures, Plotly Dash dashboard |

---

## Demo (No Downloads Required)

```bash
git clone https://github.com/YOUR_USERNAME/texas-crime-geospatial-analysis.git
cd texas-crime-geospatial-analysis
pip install -e .
python main.py --mode demo
```

This generates 5 000 synthetic crime incidents and runs the full pipeline.
Open `outputs/maps/composite_map.html` in a browser to see the results.

---

## Project Structure

```
texas-crime-geospatial-analysis/
│
├── main.py                     ← Pipeline entry point
├── setup.py                    ← Python package + C++ extension build
├── CMakeLists.txt              ← C++ CMake build
├── requirements.txt
├── config/settings.yaml        ← All configurable parameters
│
├── src/
│   ├── python/
│   │   ├── data/               ← Fetch · Load · Preprocess
│   │   ├── analysis/           ← Hotspot · Clustering · Stats · ML
│   │   ├── visualization/      ← Folium maps · Dash · matplotlib
│   │   └── utils/              ← Config · Logging · Helpers
│   └── cpp/
│       ├── spatial_index/      ← 2-D KD-Tree (C++17)
│       ├── clustering/         ← DBSCAN (C++17)
│       └── bindings/           ← pybind11 Python bindings
│
├── data/                       ← Downloaded data (git-ignored)
├── outputs/                    ← Generated maps & figures (git-ignored)
├── tests/                      ← Python + C++ unit tests
├── scripts/                    ← Standalone helper scripts
├── notebooks/                  ← Jupyter exploration
└── docs/                       ← Full documentation
    ├── index.md
    ├── installation.md
    ├── usage.md
    ├── architecture.md
    ├── analysis/
    │   ├── hotspot_detection.md
    │   ├── spatial_clustering.md
    │   └── predictive_modeling.md
    ├── api/
    │   ├── python_api.md
    │   └── cpp_api.md
    └── data/
        └── data_sources.md
```

---

## Installation

### 1. Python Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Build C++ Extension (Optional, Recommended)

The C++ spatial backend (KD-Tree + DBSCAN) accelerates spatial queries
by **10–50×**. Falls back to scikit-learn automatically if not built.

```bash
# Via pip (easiest)
pip install -e ".[cpp]"

# Or via CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 4
```

Verify:
```python
import texas_crime_spatial as tcs
print(tcs.haversine_km(tcs.Point2D(-95.37, 29.76), tcs.Point2D(-96.80, 32.78)))
# → ~392.4 km  (Houston → Dallas)
```

---

## Usage

### CLI

```bash
# Demo run (synthetic data)
python main.py --mode demo

# Full pipeline with real data
python main.py --mode full --year-from 2019 --year-to 2023

# Use your own incident CSV
python main.py --mode full --data my_incidents.csv

# Launch interactive dashboard
python main.py --mode demo --dashboard
# Open http://localhost:8050
```

### Python API

```python
from src.python.data.loader         import generate_synthetic_dataset
from src.python.data.preprocessor   import clean_incident_gdf, add_severity
from src.python.analysis.hotspot_detection  import compute_kde
from src.python.analysis.spatial_clustering import run_dbscan
from src.python.visualization.map_generator import kde_heatmap_map, cluster_map

gdf = generate_synthetic_dataset(5000)
gdf = clean_incident_gdf(gdf)
gdf = add_severity(gdf)

kde = compute_kde(gdf, grid_size=200, hotspot_pct=90)
db  = run_dbscan(gdf, eps_deg=0.05, min_pts=10)

kde_heatmap_map(gdf)
cluster_map(db.gdf, db.cluster_gdf)
```

---

## Key Outputs

| File | Contents |
|---|---|
| `outputs/maps/composite_map.html` | Multi-layer interactive map |
| `outputs/maps/kde_heatmap.html` | Crime density heatmap |
| `outputs/maps/cluster_map.html` | DBSCAN cluster hulls |
| `outputs/maps/hotspot_map.html` | Getis-Ord Gi* hot spots |
| `outputs/figures/kde_surface.png` | Static KDE contour map |
| `outputs/figures/temporal_trend.png` | Year-over-year crime trend |

---

## Running Tests

```bash
# Python unit tests
pytest tests/ -v --cov=src

# C++ tests (after CMake build)
cd build && ctest --output-on-failure
```

---

## Configuration

All pipeline parameters live in `config/settings.yaml`:

```yaml
analysis:
  dbscan:
    eps_deg:  0.05    # neighbourhood radius in degrees (~5.5 km)
    min_pts:  10

  kde:
    grid_size:   300
    hotspot_pct: 90.0

model:
  random_forest:
    n_estimators: 200
    cv_folds:     5
```

---

## Documentation

Full documentation is in the [`docs/`](docs/) directory:

- [Installation](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [Architecture](docs/architecture.md)
- [Hotspot Detection](docs/analysis/hotspot_detection.md)
- [Spatial Clustering](docs/analysis/spatial_clustering.md)
- [Predictive Modeling](docs/analysis/predictive_modeling.md)
- [Python API](docs/api/python_api.md)
- [C++ API](docs/api/cpp_api.md)
- [Data Sources](docs/data/data_sources.md)

---

## Technologies

| Technology | Role |
|---|---|
| **Python 3.10+** | Core pipeline language |
| **C++17** | KD-Tree & DBSCAN performance backend |
| **pybind11** | C++/Python interop |
| **GeoPandas / Shapely** | Spatial data structures |
| **PySAL / esda / libpysal** | Spatial statistics (Moran's I, Gi*) |
| **Folium** | Interactive HTML maps |
| **Plotly Dash** | Web analytics dashboard |
| **Scikit-Learn** | Machine learning (RF, GBR, K-Means) |
| **matplotlib / seaborn** | Static visualisation |
| **US Census TIGER/Line** | Texas geographic boundaries |
| **Texas DPS UCR / FBI CDE** | Official crime statistics |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Run tests: `pytest tests/ -v`
4. Submit a pull request.

Please follow the existing code style (Black formatter, ruff linter).

---

*Built with Python, C++, and open geospatial data for Texas crime pattern analysis.*
