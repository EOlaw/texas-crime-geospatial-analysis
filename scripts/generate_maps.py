#!/usr/bin/env python3
"""
scripts/generate_maps.py
========================
Generate all interactive maps and static figures.

Usage
-----
    python scripts/generate_maps.py
    python scripts/generate_maps.py --data data/raw/my_data.csv --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.python.data.loader        import generate_synthetic_dataset, load_incident_csv
from src.python.data.preprocessor  import add_severity, clean_incident_gdf
from src.python.analysis.hotspot_detection  import compute_kde
from src.python.analysis.spatial_clustering import run_dbscan, ripleys_k
from src.python.visualization.heatmap      import (
    plot_crime_type_bar, plot_kde_surface,
    plot_cluster_scatter, plot_ripleys_l,
)
from src.python.visualization.map_generator import (
    incident_point_map, kde_heatmap_map,
    cluster_map, composite_map,
)
from src.python.utils import get_logger

log = get_logger("generate_maps")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--all",  action="store_true", help="Generate all outputs")
    args = p.parse_args()

    if args.data:
        gdf = load_incident_csv(Path(args.data))
    else:
        log.info("No data path – using synthetic dataset")
        gdf = generate_synthetic_dataset(5000)

    gdf = clean_incident_gdf(gdf)
    gdf = add_severity(gdf)

    log.info("Generating crime type bar chart…")
    plot_crime_type_bar(gdf)

    log.info("Generating KDE surface…")
    kde = compute_kde(gdf, grid_size=150)
    plot_kde_surface(kde)

    log.info("Generating incident point map…")
    incident_point_map(gdf)

    log.info("Generating KDE heatmap…")
    kde_heatmap_map(gdf)

    log.info("Running DBSCAN for cluster map…")
    db = run_dbscan(gdf, eps_deg=0.08, min_pts=8)
    plot_cluster_scatter(db.gdf)
    cluster_map(db.gdf, db.cluster_gdf)

    log.info("Computing Ripley's K…")
    rk = ripleys_k(gdf, r_values=np.linspace(0.01, 1.0, 25))
    plot_ripleys_l(rk)

    if args.all:
        log.info("Generating composite map…")
        composite_map(gdf, db.cluster_gdf)

    log.info("All outputs generated. Check outputs/maps/ and outputs/figures/")


if __name__ == "__main__":
    main()
