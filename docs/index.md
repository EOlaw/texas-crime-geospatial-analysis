# Texas Crime Geospatial Analysis – Documentation

> **Geospatial Crime Pattern Analysis & Predictive Mapping**
> Technologies: Python · C++ (pybind11) · GeoPandas · Folium · Plotly Dash · Scikit-Learn · PySAL

---

## Table of Contents

| Document | Description |
|---|---|
| [Installation](installation.md) | Environment setup, dependencies, C++ build |
| [Usage Guide](usage.md) | How to run the pipeline, CLI reference |
| [Architecture](architecture.md) | System design, component diagram, data flow |
| **Analysis Methods** | |
| [Hotspot Detection](analysis/hotspot_detection.md) | KDE, Getis-Ord Gi*, LISA, Quadrat |
| [Spatial Clustering](analysis/spatial_clustering.md) | DBSCAN, K-Means, Hexbins, Ripley's K |
| [Predictive Modeling](analysis/predictive_modeling.md) | Random Forest, Gradient Boosting, risk grids |
| **API Reference** | |
| [Python API](api/python_api.md) | All public Python functions & classes |
| [C++ API](api/cpp_api.md) | KD-Tree, DBSCAN, pybind11 bindings |
| **Data** | |
| [Data Sources](data/data_sources.md) | UCR, FBI CDE, TIGER/Line shapefiles |

---

## Project Overview

This project performs end-to-end spatial crime analysis for the state of Texas:

1. **Data Acquisition** – Downloads crime data from Texas DPS (Socrata API), FBI Crime Data Explorer, and US Census TIGER/Line shapefiles.
2. **Preprocessing** – Cleans, validates, and enriches raw records with severity scores, crime categories, and temporal features.
3. **Spatial Analysis** – Detects crime hotspots via KDE and Getis-Ord Gi*, measures spatial autocorrelation with Moran's I and LISA, and discovers crime clusters using DBSCAN and K-Means.
4. **Predictive Modelling** – Trains Random Forest and Gradient Boosting regressors to predict county-level crime risk.
5. **Visualisation** – Produces interactive Folium maps, a Plotly Dash dashboard, and publication-quality static figures.

### Key Features

- **C++ performance backend** – A KD-Tree and DBSCAN implementation in C++17 (exposed to Python via pybind11) accelerates spatial queries by 10–50× vs. pure Python.
- **Zero-download demo** – A synthetic Texas crime dataset lets you run the full pipeline with `python main.py --mode demo` without any data downloads.
- **Modular architecture** – Each layer (data → analysis → visualization) is independently importable and testable.
- **Deployable** – The Dash dashboard runs as a standalone web application.

---

## Quick Start

```bash
# 1. Install
pip install -e ".[cpp]"   # builds the C++ extension

# 2. Demo run (no downloads required)
python main.py --mode demo

# 3. Launch dashboard
python main.py --mode demo --dashboard
```

Outputs are written to:
- `outputs/maps/`    – Interactive HTML maps
- `outputs/figures/` – Static PNG figures
- `outputs/reports/` – Saved model files
