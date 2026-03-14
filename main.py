#!/usr/bin/env python3
"""
main.py
=======
Top-level entry point for the Texas Crime Geospatial Analysis pipeline.

Usage
-----
    python main.py [--mode MODE] [--year-from YEAR] [--year-to YEAR]
                   [--crime-type TYPE] [--data PATH] [--dashboard]
                   [--skip-fetch] [--output-dir DIR]

Modes
-----
    full      : Fetch data → Preprocess → Analyse → Visualise (default)
    fetch     : Download raw data only
    analyse   : Run analysis on existing processed data
    visualise : Generate maps/figures from existing analysis results
    dashboard : Launch interactive Plotly Dash dashboard
    demo      : Run full pipeline on synthetic data (no downloads required)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.python.utils import get_logger, get_data_dir, get_output_dir

log = get_logger("main")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def step_fetch(args: argparse.Namespace) -> None:
    from src.python.data.fetcher import fetch_all
    log.info("=" * 60)
    log.info("STEP 1 – Fetching raw data")
    log.info("=" * 60)
    results = fetch_all()
    for name, path in results.items():
        log.info("  %-25s → %s", name, path)


def step_load_and_preprocess(args: argparse.Namespace):
    """Load and preprocess data; returns gdf (incidents) and county_gdf."""
    from src.python.data.loader import (
        generate_synthetic_dataset,
        load_incident_csv,
        load_county_shapefile,
    )
    from src.python.data.preprocessor import (
        add_crime_category,
        add_severity,
        add_temporal_features,
        clean_incident_gdf,
        save_processed,
    )

    log.info("=" * 60)
    log.info("STEP 2 – Loading & preprocessing")
    log.info("=" * 60)

    if args.data:
        gdf = load_incident_csv(Path(args.data))
    else:
        log.info("No --data path provided – using synthetic dataset")
        gdf = generate_synthetic_dataset(n_incidents=5000)

    # Preprocess
    gdf = clean_incident_gdf(gdf,
                             year_min=args.year_from,
                             year_max=args.year_to)
    if args.crime_type and "offense_type" in gdf.columns:
        gdf = gdf[gdf["offense_type"].str.contains(args.crime_type, case=False)]

    gdf = add_severity(gdf)
    gdf = add_crime_category(gdf)
    gdf = add_temporal_features(gdf)

    save_processed(gdf, "incidents")
    log.info("Preprocessed %d incidents", len(gdf))

    # Load county shapefile (optional)
    county_gdf = None
    shp_path   = get_data_dir("shapefiles") / "tl_2022_us_county" / "tl_2022_us_county.shp"
    if shp_path.exists():
        county_gdf = load_county_shapefile(shp_path)

    return gdf, county_gdf


def step_analyse(gdf, county_gdf, args: argparse.Namespace) -> dict:
    """Run all analysis steps; returns dict of results."""
    from src.python.analysis.hotspot_detection import (
        compute_kde,
        compute_getis_ord,
        compute_lisa,
        compute_quadrat_analysis,
    )
    from src.python.analysis.spatial_clustering import (
        compute_hexbins,
        elbow_analysis,
        ripleys_k,
        run_dbscan,
        run_kmeans,
    )
    from src.python.analysis.statistical_analysis import (
        crime_summary_stats,
        global_morans_i,
        temporal_trend,
    )

    log.info("=" * 60)
    log.info("STEP 3 – Spatial analysis")
    log.info("=" * 60)

    results: dict = {}

    # KDE
    log.info("Running KDE hotspot detection…")
    results["kde"] = compute_kde(gdf, grid_size=200)

    # Quadrat
    log.info("Running quadrat analysis…")
    results["quadrat"] = compute_quadrat_analysis(gdf)

    # DBSCAN clustering
    log.info("Running DBSCAN clustering…")
    results["dbscan"] = run_dbscan(gdf, eps_deg=0.08, min_pts=8)

    # K-Means
    log.info("Running K-Means clustering (k=8)…")
    results["kmeans"] = run_kmeans(gdf, k=8)

    # Hexbins
    log.info("Computing hexagonal bins…")
    results["hexbins"] = compute_hexbins(gdf, cell_size=0.3)

    # Ripley's K
    log.info("Computing Ripley's K function…")
    import numpy as np
    results["ripleys"] = ripleys_k(gdf, r_values=np.linspace(0.01, 1.5, 30))

    # County-level Gi* and LISA (if county polygon data available)
    if county_gdf is not None and len(county_gdf) >= 4:
        from src.python.data.preprocessor import aggregate_by_county_year, attach_county

        log.info("Attaching county labels to incidents…")
        gdf_c = attach_county(gdf, county_gdf)
        county_counts = aggregate_by_county_year(gdf_c, year_col="year")
        county_merged = county_gdf.merge(
            county_counts.groupby("county")["offense_count"].sum().reset_index(),
            left_on="county_name", right_on="county", how="left",
        ).fillna({"offense_count": 0})

        if len(county_merged) >= 4:
            log.info("Running Getis-Ord Gi* analysis…")
            results["getis_ord"] = compute_getis_ord(county_merged)

            log.info("Running LISA (Local Moran's I)…")
            results["lisa"] = compute_lisa(county_merged)

            log.info("Computing Global Moran's I…")
            results["morans_i"] = global_morans_i(county_merged)

        results["county_merged"] = county_merged

    # Temporal trend
    log.info("Analysing temporal trends…")
    df = gdf.drop(columns=["geometry"], errors="ignore")
    if "year" in df.columns:
        df["offense_count"] = 1
        results["trend"] = temporal_trend(df)

    # Summary stats
    results["summary"] = crime_summary_stats(gdf, "severity")
    log.info("Summary stats:\n%s", results["summary"].to_string())

    return results


def step_visualise(gdf, county_gdf, analysis_results: dict, args: argparse.Namespace) -> None:
    """Generate all maps and figures."""
    from src.python.visualization.heatmap import (
        plot_cluster_scatter,
        plot_crime_type_bar,
        plot_kde_surface,
        plot_ripleys_l,
        plot_temporal_trend,
    )
    from src.python.visualization.map_generator import (
        cluster_map,
        composite_map,
        hotspot_map,
        kde_heatmap_map,
        incident_point_map,
    )

    log.info("=" * 60)
    log.info("STEP 4 – Generating visualisations")
    log.info("=" * 60)

    # Static figures
    plot_crime_type_bar(gdf)
    if "kde" in analysis_results:
        plot_kde_surface(analysis_results["kde"], county_gdf)
    if "dbscan" in analysis_results:
        plot_cluster_scatter(analysis_results["dbscan"].gdf)
    if "trend" in analysis_results:
        plot_temporal_trend(analysis_results["trend"].df)
    if "ripleys" in analysis_results:
        plot_ripleys_l(analysis_results["ripleys"])

    # Interactive Folium maps
    incident_point_map(gdf)
    kde_heatmap_map(gdf)

    if "dbscan" in analysis_results:
        cluster_map(analysis_results["dbscan"].gdf,
                    analysis_results["dbscan"].cluster_gdf)

    if "getis_ord" in analysis_results:
        hotspot_map(analysis_results["getis_ord"].gdf)

    if county_gdf is not None and "county_merged" in analysis_results:
        composite_map(gdf, analysis_results["county_merged"],
                      analysis_results.get("dbscan", {}).cluster_gdf
                      if "dbscan" in analysis_results else None)

    log.info("All visualisations saved to %s", get_output_dir())


def step_dashboard(gdf, county_gdf, args: argparse.Namespace) -> None:
    from src.python.visualization.dashboard import create_app
    log.info("=" * 60)
    log.info("STEP 5 – Launching dashboard (http://127.0.0.1:8050)")
    log.info("=" * 60)
    app = create_app(gdf, county_gdf)
    app.run(debug=args.debug, port=8050, host="0.0.0.0")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Texas Crime Geospatial Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",
                   choices=["full", "fetch", "analyse", "visualise", "dashboard", "demo"],
                   default="demo",
                   help="Pipeline execution mode")
    p.add_argument("--data",       type=str, default=None,
                   help="Path to a CSV incident file (lon, lat, offense_type, year)")
    p.add_argument("--year-from",  type=int, default=2018)
    p.add_argument("--year-to",    type=int, default=2023)
    p.add_argument("--crime-type", type=str, default=None,
                   help="Filter to a specific crime type substring")
    p.add_argument("--dashboard",  action="store_true",
                   help="Launch dashboard after pipeline completes")
    p.add_argument("--skip-fetch", action="store_true",
                   help="Skip data download step")
    p.add_argument("--debug",      action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    log.info("Texas Crime Geospatial Analysis  |  mode=%s", args.mode)

    if args.mode == "fetch":
        step_fetch(args)
        return

    if args.mode == "full" and not args.skip_fetch:
        step_fetch(args)

    gdf, county_gdf = step_load_and_preprocess(args)

    if args.mode in ("full", "analyse", "demo"):
        analysis_results = step_analyse(gdf, county_gdf, args)

        if args.mode in ("full", "demo"):
            step_visualise(gdf, county_gdf, analysis_results, args)

    if args.mode == "visualise":
        from src.python.data.preprocessor import load_processed
        gdf = load_processed("incidents")
        step_visualise(gdf, county_gdf, {}, args)

    if args.mode == "dashboard" or args.dashboard:
        step_dashboard(gdf, county_gdf, args)


if __name__ == "__main__":
    main()
