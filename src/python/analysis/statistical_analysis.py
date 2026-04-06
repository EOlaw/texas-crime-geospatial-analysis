"""
statistical_analysis.py
=======================
Statistical analysis of spatial crime patterns:
  1. Global Moran's I         – overall spatial autocorrelation.
  2. Bivariate correlation     – crime type vs. socioeconomic / demographic vars.
  3. Temporal trend analysis  – year-over-year changes, Mann-Kendall test.
  4. Crime rate normalisation  – crimes per 100k population.
  5. Spatial lag regression    – OLS + Spatial Lag / Spatial Error models.
  6. Risk score computation   – composite risk index per county/region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
from esda.moran import Moran
from scipy import stats

from ..utils import get_logger, min_max_scale, timed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Global Moran's I
# ---------------------------------------------------------------------------
@dataclass
class MoranResult:
    I:         float
    expected:  float
    variance:  float
    z_score:   float
    p_value:   float
    is_clustered: bool    # True if statistically significant clustering


def global_morans_i(
    polygon_gdf: gpd.GeoDataFrame,
    count_col:   str = "offense_count",
    permutations: int = 999,
) -> MoranResult:
    """Compute Global Moran's I spatial autocorrelation statistic."""
    w = libpysal.weights.Queen.from_dataframe(polygon_gdf, silence_warnings=True, use_index=False)
    w.transform = "r"

    mi = Moran(polygon_gdf[count_col].values, w, permutations=permutations)

    result = MoranResult(
        I=float(mi.I),
        expected=float(mi.EI),
        variance=float(mi.VI_norm),
        z_score=float(mi.z_norm),
        p_value=float(mi.p_norm),
        is_clustered=(mi.p_norm < 0.05 and mi.I > mi.EI),
    )
    log.info("Global Moran's I = %.4f  (E[I]=%.4f, z=%.3f, p=%.4f)",
             result.I, result.expected, result.z_score, result.p_value)
    return result


# ---------------------------------------------------------------------------
# Bivariate correlation
# ---------------------------------------------------------------------------
@dataclass
class CorrelationResult:
    variable_pairs: pd.DataFrame   # columns: var1, var2, pearson_r, p_value, spearman_r


def bivariate_correlation(df: pd.DataFrame,
                           target: str,
                           predictors: List[str]) -> CorrelationResult:
    """
    Pearson and Spearman correlations between a target variable and predictors.
    """
    rows = []
    for pred in predictors:
        if pred not in df.columns:
            continue
        valid = df[[target, pred]].dropna()
        if len(valid) < 3:
            continue

        pearson_r,  pearson_p  = stats.pearsonr(valid[target], valid[pred])
        spearman_r, spearman_p = stats.spearmanr(valid[target], valid[pred])
        rows.append({
            "var1":       target,
            "var2":       pred,
            "pearson_r":  round(pearson_r,  4),
            "pearson_p":  round(pearson_p,  4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "n":          len(valid),
        })

    result_df = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
    log.info("Bivariate correlation computed for %d predictor pairs", len(result_df))
    return CorrelationResult(variable_pairs=result_df)


# ---------------------------------------------------------------------------
# Temporal trend analysis
# ---------------------------------------------------------------------------
@dataclass
class TrendResult:
    df:          pd.DataFrame   # year, count, pct_change, rolling_avg
    mk_statistic: float
    mk_p_value:   float
    trend_direction: str        # 'increasing', 'decreasing', 'no trend'


def temporal_trend(
    df:         pd.DataFrame,
    year_col:   str = "year",
    count_col:  str = "offense_count",
    window:     int = 3,
) -> TrendResult:
    """
    Analyse year-over-year crime trend with Mann-Kendall test.
    """
    ts = (df.groupby(year_col)[count_col]
            .sum()
            .reset_index()
            .sort_values(year_col)
            .rename(columns={year_col: "year", count_col: "count"}))

    ts["pct_change"]  = ts["count"].pct_change() * 100
    ts["rolling_avg"] = ts["count"].rolling(window, center=True).mean()

    # Mann-Kendall
    try:
        from pymannkendall import original_test as mk_test
        mk = mk_test(ts["count"].values)
        mk_stat = float(mk.s)
        mk_p    = float(mk.p)
        direction = mk.trend  # 'increasing', 'decreasing', 'no trend'
    except ImportError:
        # Simple linear regression fallback
        x = np.arange(len(ts))
        y = ts["count"].values
        if len(ts) < 2 or np.std(y) == 0:
            mk_stat, mk_p, direction = 0.0, float("nan"), "no trend"
        else:
            slope, _, _, p, _ = stats.linregress(x, y)
            mk_stat = float(slope)
            mk_p    = float(p) if not np.isnan(p) else float("nan")
            direction = ("increasing" if slope > 0 and p < 0.05 else
                         "decreasing" if slope < 0 and p < 0.05 else "no trend")

    log.info("Trend analysis: %s (p=%.4f)", direction, mk_p)
    return TrendResult(df=ts,
                       mk_statistic=mk_stat,
                       mk_p_value=mk_p,
                       trend_direction=direction)


# ---------------------------------------------------------------------------
# Crime rate normalisation (per 100k population)
# ---------------------------------------------------------------------------
def compute_crime_rate(
    crime_df:    pd.DataFrame,
    pop_df:      pd.DataFrame,
    county_col:  str = "county",
    year_col:    str = "year",
    count_col:   str = "offense_count",
    pop_col:     str = "population",
) -> pd.DataFrame:
    """
    Merge crime counts with population data and compute rate per 100k.
    """
    merged = crime_df.merge(pop_df[[county_col, year_col, pop_col]],
                            on=[county_col, year_col],
                            how="left")

    merged["crime_rate_100k"] = (
        merged[count_col] / merged[pop_col] * 100_000
    ).round(2)

    log.info("Computed crime rate for %d county-year pairs", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Risk score (composite index)
# ---------------------------------------------------------------------------
def compute_risk_score(
    polygon_gdf:   gpd.GeoDataFrame,
    indicators:    Dict[str, float],   # {column_name: weight}
) -> gpd.GeoDataFrame:
    """
    Compute a weighted composite risk score for each polygon unit.

    Parameters
    ----------
    polygon_gdf : GeoDataFrame of counties / grid cells.
    indicators  : dict mapping column names to relative weights (will be normalised).

    Returns
    -------
    GeoDataFrame with added 'risk_score' column (0–100 scale).
    """
    gdf = polygon_gdf.copy()

    total_weight = sum(indicators.values())
    if total_weight == 0:
        raise ValueError("Sum of indicator weights must be > 0")

    composite = np.zeros(len(gdf))
    for col, weight in indicators.items():
        if col not in gdf.columns:
            log.warning("Indicator column '%s' not found – skipping", col)
            continue
        vals        = pd.to_numeric(gdf[col], errors="coerce").fillna(0).values
        normed      = min_max_scale(vals)
        composite  += normed * (weight / total_weight)

    gdf["risk_score"] = (composite * 100).round(2)
    log.info("Risk score computed; range %.1f – %.1f",
             gdf["risk_score"].min(), gdf["risk_score"].max())
    return gdf


# ---------------------------------------------------------------------------
# Descriptive statistics summary
# ---------------------------------------------------------------------------
def crime_summary_stats(gdf: gpd.GeoDataFrame,
                         count_col: str = "offense_count") -> pd.DataFrame:
    """Return descriptive statistics grouped by offense_type."""
    if "offense_type" not in gdf.columns:
        return gdf[[count_col]].describe()

    summary = (
        gdf.groupby("offense_type")[count_col]
           .agg(["count", "sum", "mean", "median", "std", "min", "max"])
           .round(2)
           .reset_index()
    )
    summary.columns = ["offense_type", "n_areas", "total",
                       "mean", "median", "std", "min", "max"]
    return summary.sort_values("total", ascending=False)


# ---------------------------------------------------------------------------
# Spatial autocorrelation across crime types
# ---------------------------------------------------------------------------
def crime_type_autocorrelation(
    polygon_gdf:  gpd.GeoDataFrame,
    crime_types:  Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute Global Moran's I for each crime type column separately.
    polygon_gdf must have one column per crime type with counts.
    """
    w = libpysal.weights.Queen.from_dataframe(polygon_gdf, silence_warnings=True)
    w.transform = "r"

    if crime_types is None:
        # Auto-detect: numeric columns that look like crime types
        crime_types = [c for c in polygon_gdf.select_dtypes(include=np.number).columns
                       if c not in {"GEOID", "STATEFP", "population", "risk_score"}]

    rows = []
    for ctype in crime_types:
        if ctype not in polygon_gdf.columns:
            continue
        vals = pd.to_numeric(polygon_gdf[ctype], errors="coerce").fillna(0).values
        if vals.std() == 0:
            continue
        mi = Moran(vals, w)
        rows.append({
            "crime_type": ctype,
            "morans_I":   round(float(mi.I), 4),
            "p_value":    round(float(mi.p_norm), 4),
            "clustered":  bool(mi.p_norm < 0.05 and mi.I > mi.EI),
        })

    return pd.DataFrame(rows).sort_values("morans_I", ascending=False)
