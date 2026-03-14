# Predictive Modeling

## Overview

The predictive modeling layer trains machine learning regressors to
estimate **county/area-level crime counts** given spatial and contextual
features. The trained model is then applied to a regular grid to produce
a continuous **risk prediction surface** for mapping.

---

## Feature Engineering

### Automatically Derived Features

| Feature | Description |
|---|---|
| `centroid_lon` | Polygon centroid longitude |
| `centroid_lat` | Polygon centroid latitude |
| `area_deg2` | Polygon area in squared degrees |
| `spatial_lag` | Spatially lagged mean of crime count (Queen contiguity) |

The spatial lag captures **neighbourhood effects** – areas surrounded by
high-crime neighbours tend to have higher crime themselves (positive
spatial autocorrelation).

### Optional Extra Features

Pass any numeric columns from the polygon GeoDataFrame as `extra_features`:

```python
X, y, names = build_feature_matrix(
    county_gdf,
    count_col="offense_count",
    extra_features=["population", "unemployment_rate", "poverty_pct"],
)
```

---

## Models

### 1. Random Forest Regressor

An ensemble of decision trees that averages predictions to reduce variance.

**Strengths:** Handles non-linear relationships, provides feature importances,
robust to outliers.

```python
from src.python.analysis.predictive_model import train_random_forest

result = train_random_forest(
    X, y, feature_names,
    n_estimators=200,
    cv_folds=5,
)
print(f"CV RMSE: {result.cv_rmse_mean:.2f} ± {result.cv_rmse_std:.2f}")
print(f"R²: {result.cv_r2_mean:.4f}")
print(result.feature_importances.head())
```

### 2. Gradient Boosting Regressor

Builds trees sequentially, each correcting the residuals of the previous
(equivalent to XGBoost with the same defaults).

```python
from src.python.analysis.predictive_model import train_gradient_boosting

result = train_gradient_boosting(
    X, y, feature_names,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
)
```

---

## Cross-Validation

Both models use **k-fold cross-validation** (default k=5) to estimate
generalisation error on held-out folds:

```
For each fold:
  - Train on (k-1) folds
  - Predict on held-out fold
  - Compute RMSE and R²

Report: mean ± std of RMSE and R² across folds
```

---

## Risk Prediction Grid

After training, apply the model to a regular grid covering the study area
to produce a **continuous risk surface**:

```python
from src.python.analysis.predictive_model import predict_risk_grid

grid_gdf = predict_risk_grid(result, study_area=county_gdf, grid_size=50)
# grid_gdf has columns: longitude, latitude, predicted_count, geometry
```

The grid is then visualised as a heatmap overlay on the interactive map.

---

## Model Persistence

```python
from src.python.analysis.predictive_model import save_model, load_model

# Save
save_model(result, name="crime_rf_2023")

# Load in another session
result = load_model("crime_rf_2023")
```

Models are saved as joblib files in `outputs/reports/`.

---

## Evaluation Metrics

| Metric | Formula | Notes |
|---|---|---|
| RMSE | √(mean((y - ŷ)²)) | Same units as target |
| MAE | mean(|y - ŷ|) | Robust to outliers |
| R² | 1 - SS_res/SS_tot | 1.0 = perfect, 0 = mean baseline |

---

## Limitations & Caveats

- Spatial autocorrelation in crime data violates the i.i.d. assumption of
  standard cross-validation; **spatial cross-validation** (leave-one-region-out)
  provides less optimistic estimates.
- The model predicts **counts** – not individual incident probabilities.
- Predictions on the risk grid extrapolate beyond training polygons; treat
  as indicative rather than precise.
- Socioeconomic predictors (poverty rate, unemployment) substantially improve
  accuracy but require population-level data (not included in the base dataset).

---

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting
  machine. *Annals of Statistics*, 29(5), 1189–1232.
- Chainey, S., Tompson, L., & Uhlig, S. (2008). The utility of hotspot
  mapping for predicting spatial patterns of crime. *Security Journal*, 21(1–2).
