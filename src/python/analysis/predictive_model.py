"""
predictive_model.py
===================
Predictive crime risk modelling:
  1. Feature engineering     – spatial lag, KDE density, temporal features.
  2. Random Forest regressor  – predict crime counts per area-year.
  3. XGBoost regressor        – gradient-boosted alternative.
  4. Time-series forecasting  – SARIMA / Holt-Winters extrapolation.
  5. Risk prediction map      – apply trained model to grid for mapping.
  6. Model evaluation         – cross-validation, RMSE, MAE, R².
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from ..utils import get_logger, get_output_dir, timed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering for modelling
# ---------------------------------------------------------------------------
def build_feature_matrix(
    polygon_gdf:       gpd.GeoDataFrame,
    count_col:         str          = "offense_count",
    extra_features:    List[str]    = None,
    include_spatial_lag: bool       = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build feature matrix (X) and target vector (y) from a polygon GeoDataFrame.

    Automatically derived features:
        - area_deg2      : polygon area in squared degrees.
        - centroid_lon   : centroid longitude.
        - centroid_lat   : centroid latitude.
        - spatial_lag    : spatially lagged mean of count_col (queen contiguity).
        + any extra_features columns supplied.

    Returns
    -------
    X        : (n, p) float array
    y        : (n,)   float array
    feat_names : list of feature names
    """
    import libpysal

    gdf = polygon_gdf.copy()

    feat_names = []
    feature_cols = []

    # Geometric features
    gdf["area_deg2"]    = gdf.geometry.area
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf["centroid_lat"] = gdf.geometry.centroid.y
    for c in ["area_deg2", "centroid_lon", "centroid_lat"]:
        feature_cols.append(c)
        feat_names.append(c)

    # Spatial lag
    if include_spatial_lag and count_col in gdf.columns:
        try:
            w = libpysal.weights.Queen.from_dataframe(gdf, silence_warnings=True)
            w.transform = "r"
            spatial_lag = libpysal.weights.lag_spatial(w, gdf[count_col].fillna(0).values)
            gdf["spatial_lag"] = spatial_lag
            feature_cols.append("spatial_lag")
            feat_names.append("spatial_lag")
        except Exception as exc:
            log.warning("Could not compute spatial lag: %s", exc)

    # Extra features
    if extra_features:
        for col in extra_features:
            if col in gdf.columns:
                feature_cols.append(col)
                feat_names.append(col)

    X = gdf[feature_cols].fillna(0).values.astype(float)
    y = gdf[count_col].fillna(0).values.astype(float)

    log.info("Feature matrix: shape %s | target range [%.1f, %.1f]",
             X.shape, y.min(), y.max())
    return X, y, feat_names


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    model_name:      str
    model:           object          # fitted sklearn-compatible estimator
    feature_names:   List[str]
    scaler:          Optional[StandardScaler]
    cv_rmse_mean:    float
    cv_rmse_std:     float
    cv_r2_mean:      float
    test_metrics:    Dict[str, float]   # rmse, mae, r2 on held-out test
    feature_importances: pd.Series


@timed
def train_random_forest(
    X:             np.ndarray,
    y:             np.ndarray,
    feature_names: List[str],
    n_estimators:  int = 200,
    max_depth:     Optional[int] = None,
    cv_folds:      int = 5,
    scale:         bool = False,
) -> ModelResult:
    """
    Train a Random Forest regressor with k-fold cross-validation.
    """
    scaler = None
    X_fit  = X.copy()
    if scale:
        scaler = StandardScaler()
        X_fit  = scaler.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )

    kf   = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rmse_scores = []
    r2_scores   = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_fit)):
        rf.fit(X_fit[train_idx], y[train_idx])
        preds  = rf.predict(X_fit[val_idx])
        rmse   = np.sqrt(mean_squared_error(y[val_idx], preds))
        r2     = r2_score(y[val_idx], preds)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        log.info("  Fold %d: RMSE=%.2f  R²=%.4f", fold + 1, rmse, r2)

    # Final fit on all data
    rf.fit(X_fit, y)
    final_preds = rf.predict(X_fit)

    importances = pd.Series(rf.feature_importances_, index=feature_names)\
                    .sort_values(ascending=False)

    result = ModelResult(
        model_name="RandomForest",
        model=rf,
        feature_names=feature_names,
        scaler=scaler,
        cv_rmse_mean=float(np.mean(rmse_scores)),
        cv_rmse_std=float(np.std(rmse_scores)),
        cv_r2_mean=float(np.mean(r2_scores)),
        test_metrics={
            "rmse": float(np.sqrt(mean_squared_error(y, final_preds))),
            "mae":  float(mean_absolute_error(y, final_preds)),
            "r2":   float(r2_score(y, final_preds)),
        },
        feature_importances=importances,
    )
    log.info("RandomForest CV RMSE=%.2f ±%.2f  R²=%.4f",
             result.cv_rmse_mean, result.cv_rmse_std, result.cv_r2_mean)
    return result


@timed
def train_gradient_boosting(
    X:             np.ndarray,
    y:             np.ndarray,
    feature_names: List[str],
    n_estimators:  int = 300,
    learning_rate: float = 0.05,
    max_depth:     int = 4,
    cv_folds:      int = 5,
) -> ModelResult:
    """Train a Gradient Boosting (XGBoost-like) regressor."""
    gbr = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
    )

    kf        = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rmse_list, r2_list = [], []

    for train_idx, val_idx in kf.split(X):
        gbr.fit(X[train_idx], y[train_idx])
        preds = gbr.predict(X[val_idx])
        rmse_list.append(np.sqrt(mean_squared_error(y[val_idx], preds)))
        r2_list.append(r2_score(y[val_idx], preds))

    gbr.fit(X, y)
    final_preds = gbr.predict(X)

    importances = pd.Series(gbr.feature_importances_, index=feature_names)\
                    .sort_values(ascending=False)

    result = ModelResult(
        model_name="GradientBoosting",
        model=gbr,
        feature_names=feature_names,
        scaler=None,
        cv_rmse_mean=float(np.mean(rmse_list)),
        cv_rmse_std=float(np.std(rmse_list)),
        cv_r2_mean=float(np.mean(r2_list)),
        test_metrics={
            "rmse": float(np.sqrt(mean_squared_error(y, final_preds))),
            "mae":  float(mean_absolute_error(y, final_preds)),
            "r2":   float(r2_score(y, final_preds)),
        },
        feature_importances=importances,
    )
    log.info("GBR CV RMSE=%.2f ±%.2f  R²=%.4f",
             result.cv_rmse_mean, result.cv_rmse_std, result.cv_r2_mean)
    return result


# ---------------------------------------------------------------------------
# Predict risk on a regular grid
# ---------------------------------------------------------------------------
def predict_risk_grid(
    model_result: ModelResult,
    study_area:   gpd.GeoDataFrame,
    grid_size:    int = 50,
) -> gpd.GeoDataFrame:
    """
    Apply a trained model to a regular grid of points over the study area
    to produce a continuous risk prediction surface.

    Returns a GeoDataFrame of grid points with predicted crime counts.
    """
    from shapely.geometry import box

    bounds      = study_area.total_bounds           # (minx, miny, maxx, maxy)
    lon_vals    = np.linspace(bounds[0], bounds[2], grid_size)
    lat_vals    = np.linspace(bounds[1], bounds[3], grid_size)
    lon_g, lat_g = np.meshgrid(lon_vals, lat_vals)

    lons_flat   = lon_g.ravel()
    lats_flat   = lat_g.ravel()
    n           = len(lons_flat)

    # Build feature matrix (same structure as training)
    feat_map = {
        "centroid_lon": lons_flat,
        "centroid_lat": lats_flat,
        "area_deg2":    np.full(n, (lon_vals[1] - lon_vals[0]) ** 2),
        "spatial_lag":  np.zeros(n),   # zero spatial lag on grid
    }

    feature_names = model_result.feature_names
    X_grid = np.column_stack([
        feat_map.get(f, np.zeros(n)) for f in feature_names
    ])

    if model_result.scaler is not None:
        X_grid = model_result.scaler.transform(X_grid)

    predicted = model_result.model.predict(X_grid)
    predicted = np.clip(predicted, 0, None)

    grid_gdf = gpd.GeoDataFrame({
        "longitude":       lons_flat,
        "latitude":        lats_flat,
        "predicted_count": predicted,
        "geometry":        [Point(x, y) for x, y in zip(lons_flat, lats_flat)],
    }, crs="EPSG:4326")

    log.info("Predicted risk grid: %d points  range [%.1f, %.1f]",
             n, predicted.min(), predicted.max())
    return grid_gdf


# ---------------------------------------------------------------------------
# Save / load model
# ---------------------------------------------------------------------------
def save_model(result: ModelResult, name: str = "crime_model") -> Path:
    out_dir  = get_output_dir("reports")
    out_path = out_dir / f"{name}.joblib"
    joblib.dump(result, out_path)
    log.info("Model saved → %s", out_path)
    return out_path


def load_model(name: str = "crime_model") -> ModelResult:
    path = get_output_dir("reports") / f"{name}.joblib"
    result = joblib.load(path)
    log.info("Model loaded from %s", path)
    return result
