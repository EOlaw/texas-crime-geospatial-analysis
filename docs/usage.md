# Usage Guide

## CLI Reference

The main pipeline is invoked via `python main.py` or (after `pip install -e .`) via the `texas-crime` command.

```
usage: main.py [-h] [--mode {full,fetch,analyse,visualise,dashboard,demo}]
               [--data DATA] [--year-from YEAR] [--year-to YEAR]
               [--crime-type TYPE] [--dashboard] [--skip-fetch] [--debug]
```

### Modes

| Mode | Description |
|---|---|
| `demo` | **Default.** Full pipeline on synthetic data – no downloads required. |
| `full` | Fetch → Preprocess → Analyse → Visualise (real data). |
| `fetch` | Download raw data only. |
| `analyse` | Run analysis on existing `data/processed/incidents.parquet`. |
| `visualise` | Generate maps/figures from existing processed data. |
| `dashboard` | Launch the Plotly Dash web dashboard. |

---

## Common Workflows

### Run the demo (no internet required)

```bash
python main.py --mode demo
```

Generates 5 000 synthetic incidents across 10 Texas cities and runs the full pipeline.

### Download real data then analyse

```bash
# Download data (requires internet; FBI key is optional)
python main.py --mode fetch --fbi-key YOUR_KEY

# Analyse downloaded data
python main.py --mode analyse

# Generate visualisations
python main.py --mode visualise
```

### Use your own incident CSV

Your CSV must contain at minimum: `longitude`, `latitude`, `offense_type`, `year`.

```bash
python main.py --mode full --data /path/to/my_incidents.csv \
               --year-from 2019 --year-to 2023
```

### Filter to a specific crime type

```bash
python main.py --mode demo --crime-type "burglary"
```

### Launch dashboard after analysis

```bash
python main.py --mode demo --dashboard
# Then open http://localhost:8050 in a browser
```

---

## Individual Scripts

### `scripts/fetch_data.py`

```bash
python scripts/fetch_data.py --app-token SOCRATA_TOKEN --fbi-key FBI_KEY
```

Downloads:
- Texas DPS UCR county data from `data.texas.gov`
- FBI Crime Data Explorer state summary
- US Census TIGER/Line county and city shapefiles

### `scripts/run_analysis.py`

```bash
python scripts/run_analysis.py --data data/raw/my_data.csv \
       --eps 0.08 --min-pts 10 --k 8
```

Runs KDE, DBSCAN, K-Means, Ripley's K, temporal trend analysis, and
prints a crime summary table.

### `scripts/generate_maps.py`

```bash
python scripts/generate_maps.py --all
```

Generates all interactive HTML maps and static figures.

---

## Python API Usage

```python
from src.python.data.loader       import generate_synthetic_dataset
from src.python.data.preprocessor import clean_incident_gdf, add_severity
from src.python.analysis.hotspot_detection  import compute_kde
from src.python.analysis.spatial_clustering import run_dbscan
from src.python.visualization.map_generator import kde_heatmap_map, cluster_map

# Load data
gdf = generate_synthetic_dataset(n_incidents=5000)
gdf = clean_incident_gdf(gdf)
gdf = add_severity(gdf)

# Hotspot detection
kde = compute_kde(gdf, grid_size=200, hotspot_pct=90)
print(f"KDE bandwidth: {kde.bandwidth:.4f}")
print(f"Hotspot cells: {kde.hotspot_mask.sum()}")

# Spatial clustering
db = run_dbscan(gdf, eps_deg=0.05, min_pts=10)
print(f"DBSCAN clusters: {db.n_clusters}  noise: {db.n_noise}")

# Interactive maps
kde_heatmap_map(gdf)               # outputs/maps/kde_heatmap.html
cluster_map(db.gdf, db.cluster_gdf)  # outputs/maps/cluster_map.html
```

---

## Output Files

| Path | Contents |
|---|---|
| `outputs/maps/incident_map.html` | Interactive point map of crime incidents |
| `outputs/maps/kde_heatmap.html` | Kernel density heatmap |
| `outputs/maps/cluster_map.html` | DBSCAN cluster hulls |
| `outputs/maps/hotspot_map.html` | Getis-Ord Gi* hot/cold spot map |
| `outputs/maps/composite_map.html` | Multi-layer combined map |
| `outputs/maps/risk_prediction_map.html` | Predicted risk surface |
| `outputs/figures/kde_surface.png` | Static KDE contour map |
| `outputs/figures/crime_type_bar.png` | Crime type frequency chart |
| `outputs/figures/temporal_trend.png` | Year-over-year trend chart |
| `outputs/figures/cluster_scatter.png` | Cluster scatter plot |
| `outputs/figures/ripleys_l.png` | Ripley's L function |
| `outputs/reports/crime_model.joblib` | Serialised trained model |

---

## Configuration

Edit `config/settings.yaml` to adjust all pipeline parameters without
touching Python code:

```yaml
analysis:
  dbscan:
    eps_deg:  0.05    # neighbourhood radius in degrees
    min_pts:  10      # minimum cluster size

  kde:
    grid_size:   300
    hotspot_pct: 90.0
```

See [config/settings.yaml](../config/settings.yaml) for the full reference.
