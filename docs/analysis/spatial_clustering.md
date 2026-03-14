# Spatial Clustering

## Overview

Spatial clustering groups crime incidents into geographically meaningful
clusters without requiring a pre-specified number of groups (DBSCAN) or
with a fixed-k centroid approach (K-Means). Additional tools (hexbins,
Ripley's K) provide complementary perspectives on spatial point patterns.

---

## 1. DBSCAN

### Algorithm

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
classifies each point as a **core**, **border**, or **noise** point:

```
A point p is a CORE POINT if:
    |{q : dist(p,q) ≤ ε}| ≥ MinPts

A cluster is the transitive closure of density-reachable core points.
All remaining points are NOISE (label = -1).
```

### C++ Acceleration

The C++ `texas_crime_spatial.DBSCAN` uses the custom KD-Tree for
O(log n) ε-neighbourhood queries, making the overall algorithm
O(n log n) on average — significantly faster than the brute-force
O(n²) approach.

When the C++ extension is unavailable, `sklearn.cluster.DBSCAN` is used
as a transparent fallback.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `eps_deg` | 0.05° (~5.5 km) | Neighbourhood radius |
| `min_pts` | 10 | Minimum cluster membership |

**Choosing ε:** Plot the sorted k-NN distance graph; look for the
"knee" – this inflection point is a good ε estimate.

### Interpreting Results

- Each cluster represents a **geographically contiguous crime concentration**.
- Convex hulls of clusters are computed and mapped as polygon overlays.
- Noise points are mapped separately and often represent isolated incidents.

### Python API

```python
from src.python.analysis.spatial_clustering import run_dbscan

result = run_dbscan(gdf, eps_deg=0.05, min_pts=10)
print(f"Clusters: {result.n_clusters}  Noise: {result.n_noise}")
print(f"C++ backend: {result.used_cpp}")
```

---

## 2. K-Means Clustering

### Algorithm

K-Means partitions n points into k clusters by minimising within-cluster
sum-of-squares (inertia):

```
minimise Σᵢ Σₓ∈Cᵢ ‖x − μᵢ‖²
```

Unlike DBSCAN, K-Means requires specifying k and always assigns every
point to a cluster.

### Elbow Method

Use `elbow_analysis()` to plot inertia vs. k and pick the "elbow" point:

```python
from src.python.analysis.spatial_clustering import elbow_analysis

df = elbow_analysis(gdf, k_range=range(2, 15))
# Plot df["k"] vs df["inertia"]
```

### Python API

```python
from src.python.analysis.spatial_clustering import run_kmeans

result = run_kmeans(gdf, k=8)
print(f"Inertia: {result.inertia:.1f}")
centres = result.centre_gdf   # GeoDataFrame of cluster centroids
```

---

## 3. Hexagonal Binning

Hexagonal bins provide a **regular spatial aggregation** that avoids
the edge-orientation bias of square grids.

```python
from src.python.analysis.spatial_clustering import compute_hexbins

result = compute_hexbins(gdf, cell_size=0.25)   # ~28 km cells
hex_gdf = result.hex_gdf   # GeoDataFrame with 'count' column
```

Hexbins are useful for:
- Aggregating incident density to a uniform grid.
- Input features for spatial regression models.
- Choropleth-style visualisation at customisable scales.

---

## 4. Ripley's K / L Function

The K function measures the expected number of additional points within
distance r of a randomly chosen point, normalised by intensity:

```
K(r) = λ⁻¹ E[# points within r of a typical point]
```

The stabilised L function:
```
L(r) = √(K(r) / π)
```

**Interpretation of L(r) − r:**
- L(r) − r > 0 → clustering at scale r
- L(r) − r ≈ 0 → CSR (complete spatial randomness)
- L(r) − r < 0 → regularity at scale r

```python
import numpy as np
from src.python.analysis.spatial_clustering import ripleys_k

df = ripleys_k(gdf, r_values=np.linspace(0.01, 2.0, 50))
# Peak of L(r)-r indicates the dominant clustering scale
peak_r = df.loc[df["L_minus_r"].idxmax(), "r"]
print(f"Peak clustering scale: {peak_r:.3f}° (~{peak_r * 111:.0f} km)")
```

---

## References

- Ester, M., et al. (1996). A density-based algorithm for discovering
  clusters. *KDD-96*, 226–231.
- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Trans.
  Information Theory*, 28(2), 129–137.
- Ripley, B. D. (1976). The second-order analysis of stationary point
  processes. *Journal of Applied Probability*, 13(2), 255–266.
