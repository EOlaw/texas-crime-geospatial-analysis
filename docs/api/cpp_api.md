# C++ API Reference

The C++ implementation lives in `src/cpp/` and is exposed to Python via
the `texas_crime_spatial` pybind11 extension module.

---

## Building

```bash
# Option A – pip (recommended)
pip install -e ".[cpp]"

# Option B – CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

After building, the extension is importable from the project root:

```python
import texas_crime_spatial as tcs
help(tcs)
```

---

## `texas_crime_spatial.Point2D`

2-D geographic point.

```python
p = tcs.Point2D(x=-97.74, y=30.27, id=0)
# x = longitude, y = latitude, id = optional row index
```

| Attribute | Type | Description |
|---|---|---|
| `x` | float | Longitude |
| `y` | float | Latitude |
| `id` | int | Original dataset row index |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `dist_sq(other)` | float | Squared Euclidean distance |

---

## `texas_crime_spatial.KDTree`

2-D KD-Tree for spatial queries.

```python
points = [tcs.Point2D(-97.7, 30.3, 0), tcs.Point2D(-97.6, 30.2, 1)]
tree   = tcs.KDTree(points)
```

**Constructor**

```python
KDTree(points: list[Point2D])
```

**Methods**

| Method | Signature | Description |
|---|---|---|
| `knn` | `(query, k) → list[NNResult]` | k nearest neighbours, sorted by distance |
| `range_search` | `(query, radius) → list[Point2D]` | All points within radius |
| `count_within` | `(query, radius) → int` | Count points within radius (no allocation) |
| `__len__` | `() → int` | Number of points stored |

**`NNResult` fields:**
- `point: Point2D` – the nearest point
- `dist: float` – Euclidean distance to query

**Complexity**

| Operation | Average | Worst case |
|---|---|---|
| Build | O(n log n) | O(n log n) |
| kNN query | O(k log n) | O(n) |
| Range query | O(log n + k) | O(n) |

---

## `texas_crime_spatial.DBSCAN`

Density-based spatial clustering accelerated by the KD-Tree.

```python
db     = tcs.DBSCAN(eps=0.05, min_pts=10)
labels = db.fit(points)   # list of ints; -1 = noise
```

**Constructor**

```python
DBSCAN(eps: float, min_pts: int)
```

| Parameter | Description |
|---|---|
| `eps` | Neighbourhood radius (same units as point coordinates) |
| `min_pts` | Minimum number of points to form a core point |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `fit(points)` | `list[int]` | Cluster labels (-1 = noise, ≥0 = cluster id) |

**Properties (read-only after `fit()`)**

| Property | Type | Description |
|---|---|---|
| `num_clusters` | int | Number of clusters found |
| `num_noise_points` | int | Number of noise points |
| `core_point_indices` | list[int] | Indices of core points |

---

## `texas_crime_spatial.ClusterStats`

Summary statistics for a single cluster.

| Field | Type | Description |
|---|---|---|
| `cluster_id` | int | Cluster identifier |
| `size` | int | Number of member points |
| `centroid_lon` | float | Mean longitude |
| `centroid_lat` | float | Mean latitude |
| `spread_deg` | float | RMS distance from centroid (degrees) |

---

## `texas_crime_spatial.compute_cluster_stats`

```python
stats = tcs.compute_cluster_stats(points, labels)
# → list[ClusterStats]
```

Compute per-cluster centroid and spread from `fit()` output.

---

## `texas_crime_spatial.haversine_km`

```python
dist_km = tcs.haversine_km(point_a, point_b)
```

Great-circle distance in kilometres using the Haversine formula.

---

## C++ Header API

### `kdtree.h`  –  `namespace texas_crime`

```cpp
struct Point2D { double x, y; int id; };

class KDTree {
public:
    explicit KDTree(std::vector<Point2D> pts);
    std::vector<NNResult> knn(const Point2D& query, int k) const;
    std::vector<Point2D>  range_search(const Point2D& query, double radius) const;
    int                   count_within(const Point2D& query, double radius) const;
    std::size_t           size() const;
};

double haversine_km(const Point2D& a, const Point2D& b);
```

### `dbscan.h`  –  `namespace texas_crime`

```cpp
class DBSCAN {
public:
    DBSCAN(double eps, int min_pts);
    std::vector<int> fit(const std::vector<Point2D>& points);
    int num_clusters()     const;
    int num_noise_points() const;
    const std::vector<int>& core_point_indices() const;
};

struct ClusterStats { int cluster_id, size; double centroid_lon, centroid_lat, spread_deg; };
std::vector<ClusterStats> compute_cluster_stats(
    const std::vector<Point2D>& points, const std::vector<int>& labels);
```

---

## Example: Full C++ Workflow from Python

```python
import texas_crime_spatial as tcs

# Build points from a GeoDataFrame
points = [
    tcs.Point2D(x=float(row.longitude), y=float(row.latitude), id=i)
    for i, row in gdf.iterrows()
]

# Fast spatial query: points within 10 km of Austin
austin  = tcs.Point2D(-97.74, 30.27)
eps_deg = 10.0 / 111.0    # convert km → degrees

nearby = tcs.KDTree(points).range_search(austin, eps_deg)
print(f"Incidents within 10 km of Austin: {len(nearby)}")

# Cluster with DBSCAN
db     = tcs.DBSCAN(eps=0.05, min_pts=10)
labels = db.fit(points)
print(f"Clusters: {db.num_clusters}  Noise: {db.num_noise_points}")

# Cluster statistics
stats = tcs.compute_cluster_stats(points, labels)
for s in stats[:3]:
    print(f"  Cluster {s.cluster_id}: n={s.size}  "
          f"centroid=({s.centroid_lon:.3f}, {s.centroid_lat:.3f})")
```
