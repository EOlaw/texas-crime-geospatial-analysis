# Hotspot Detection

## Overview

Hotspot detection identifies geographic concentrations of crime that exceed
what would be expected under a random (Complete Spatial Randomness, CSR)
distribution. This project implements four complementary methods.

---

## 1. Kernel Density Estimation (KDE)

### Methodology

KDE generates a **continuous crime density surface** by placing a kernel
function (Gaussian) at each incident location and summing contributions
across a regular grid.

```
f(x) = (1/n) Σ K_h(x - xᵢ)
```

where `K_h` is the Gaussian kernel with bandwidth `h`.

**Bandwidth selection:** Scott's rule (`h = n^{-1/5} × σ`) is used by
default. A custom bandwidth can be supplied via the `bandwidth` parameter.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `grid_size` | 300 | Resolution of the density grid (cells per axis) |
| `bandwidth` | `None` (Scott's rule) | KDE bandwidth |
| `hotspot_pct` | 90.0 | Percentile threshold for hotspot mask |

### Interpretation

- High density cells (above the `hotspot_pct` percentile) are flagged as hotspots.
- The output surface can be overlaid on a basemap as a filled contour plot.

### Python API

```python
from src.python.analysis.hotspot_detection import compute_kde

kde = compute_kde(gdf, grid_size=300, hotspot_pct=90)
print(f"Bandwidth: {kde.bandwidth:.4f}")
print(f"Hotspot cells: {kde.hotspot_mask.sum()}")
```

---

## 2. Getis-Ord Gi* Statistic

### Methodology

The Getis-Ord Gi* statistic identifies **statistically significant spatial
clusters** of high (hot spots) and low (cold spots) values at the polygon
(county / grid cell) level.

```
Gi*(d) = [Σⱼ wᵢⱼ(d) xⱼ - X̄ Σⱼ wᵢⱼ] / [S √(n Σⱼ wᵢⱼ² - (Σⱼ wᵢⱼ)²) / (n-1)]
```

where `wᵢⱼ` is the spatial weight matrix (Queen contiguity, row-standardised).

### Confidence Levels

| Z-score | p-value | Classification |
|---|---|---|
| > 2.576 | < 0.01 | Hot Spot 99% |
| > 1.960 | < 0.05 | Hot Spot 95% |
| > 1.645 | < 0.10 | Hot Spot 90% |
| -1.960 to 1.960 | ≥ 0.05 | Not Significant |
| < -1.960 | < 0.05 | Cold Spot 95% |

### Python API

```python
from src.python.analysis.hotspot_detection import compute_getis_ord

result = compute_getis_ord(county_gdf, count_col="offense_count")
print(result.global_summary)
hotspots = result.hotspot_gdf   # counties with p < 0.05 hot spots
```

---

## 3. Local Moran's I (LISA)

### Methodology

Local Indicators of Spatial Association (LISA) decompose the global
Moran's I into individual contributions per polygon, producing four
cluster/outlier types:

| Quadrant | Meaning |
|---|---|
| **HH** | High count surrounded by high counts (hotspot core) |
| **LL** | Low count surrounded by low counts (coldspot) |
| **HL** | High count surrounded by low counts (spatial outlier) |
| **LH** | Low count surrounded by high counts (spatial outlier) |

### Python API

```python
from src.python.analysis.hotspot_detection import compute_lisa

result = compute_lisa(county_gdf, count_col="offense_count")
print(f"Global Moran's I = {result.global_moran:.4f}  (p={result.global_p:.4f})")
# Significant HH clusters
hh = result.gdf[result.gdf["lisa_quadrant"] == "HH"]
```

---

## 4. Quadrat Analysis

### Methodology

The study area is divided into a regular `n_cols × n_rows` grid. Incident
counts per cell are compared against Complete Spatial Randomness (CSR)
using a chi-squared test.

**Variance-to-Mean Ratio (VMR):**
- VMR > 1 → clustered pattern
- VMR ≈ 1 → random (CSR)
- VMR < 1 → regular / dispersed pattern

### Python API

```python
from src.python.analysis.hotspot_detection import compute_quadrat_analysis

result = compute_quadrat_analysis(gdf, n_cols=20, n_rows=20)
print(f"χ² = {result.chi2_stat:.2f}  p = {result.chi2_p:.4f}")
print(f"VMR = {result.vmr:.2f}  →  {'clustered' if result.vmr > 1 else 'random'}")
```

---

## References

- Chainey, S., & Ratcliffe, J. (2005). *GIS and Crime Mapping*. Wiley.
- Ord, J. K., & Getis, A. (1995). Local spatial autocorrelation statistics.
  *Geographical Analysis*, 27(4), 286–306.
- Anselin, L. (1995). Local indicators of spatial association—LISA.
  *Geographical Analysis*, 27(2), 93–115.
