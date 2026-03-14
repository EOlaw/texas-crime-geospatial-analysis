#!/usr/bin/env python3
"""
scripts/run_analysis.py
=======================
Run the full spatial analysis pipeline on processed data.

Usage
-----
    python scripts/run_analysis.py
    python scripts/run_analysis.py --data data/raw/my_crime_data.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.python.data.loader       import generate_synthetic_dataset, load_incident_csv
from src.python.data.preprocessor import (
    add_crime_category, add_severity, clean_incident_gdf, save_processed
)
from src.python.analysis.hotspot_detection  import compute_kde, compute_quadrat_analysis
from src.python.analysis.spatial_clustering import run_dbscan, run_kmeans, ripleys_k
from src.python.analysis.statistical_analysis import crime_summary_stats, temporal_trend
from src.python.utils import get_logger
import numpy as np
import pandas as pd

log = get_logger("run_analysis")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None,
                   help="Path to incident CSV (uses synthetic data if omitted)")
    p.add_argument("--eps",  type=float, default=0.08)
    p.add_argument("--min-pts", type=int, default=8)
    p.add_argument("--k",    type=int, default=8)
    args = p.parse_args()

    # Load
    if args.data:
        gdf = load_incident_csv(Path(args.data))
    else:
        log.info("Using synthetic dataset")
        gdf = generate_synthetic_dataset(5000)

    gdf = clean_incident_gdf(gdf)
    gdf = add_severity(gdf)
    gdf = add_crime_category(gdf)
    save_processed(gdf, "incidents")

    log.info("Incidents loaded: %d", len(gdf))

    # KDE
    log.info("Running KDE hotspot detection…")
    kde = compute_kde(gdf, grid_size=150)
    log.info("KDE bandwidth=%.4f  hotspot_cells=%d",
             kde.bandwidth, kde.hotspot_mask.sum())

    # DBSCAN
    log.info("Running DBSCAN (eps=%.3f  min_pts=%d)…", args.eps, args.min_pts)
    db = run_dbscan(gdf, eps_deg=args.eps, min_pts=args.min_pts)
    log.info("DBSCAN: %d clusters  %d noise", db.n_clusters, db.n_noise)

    # KMeans
    log.info("Running K-Means (k=%d)…", args.k)
    km = run_kmeans(gdf, k=args.k)
    log.info("K-Means inertia=%.1f", km.inertia)

    # Ripley's K
    log.info("Computing Ripley's K…")
    rk = ripleys_k(gdf, r_values=np.linspace(0.01, 1.0, 20))
    log.info("Ripley's K computed; max L-r=%.4f", rk["L_minus_r"].max())

    # Temporal trend
    if "year" in gdf.columns:
        df = gdf.drop(columns=["geometry"], errors="ignore")
        df["offense_count"] = 1
        trend = temporal_trend(df)
        log.info("Trend: %s (p=%.4f)", trend.trend_direction, trend.mk_p_value)

    # Summary
    summary = crime_summary_stats(gdf, "severity")
    print("\n── Crime Summary Statistics ──")
    print(summary.to_string(index=False))

    log.info("Analysis complete.")


if __name__ == "__main__":
    main()
