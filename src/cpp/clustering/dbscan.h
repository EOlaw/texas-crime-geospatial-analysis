#pragma once
/**
 * dbscan.h
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
 * optimised for 2-D geospatial crime data using the KD-Tree for ε-neighbourhood
 * queries.
 *
 * Label convention:
 *   label == -1   →  NOISE (unclustered)
 *   label >=  0   →  cluster id (0-indexed)
 */

#include <vector>
#include "../spatial_index/kdtree.h"

namespace texas_crime {

// ---------------------------------------------------------------------------
// DBSCAN
// ---------------------------------------------------------------------------
class DBSCAN {
public:
    /**
     * @param eps       Neighbourhood radius in the same units as the points
     *                  (decimal degrees for lon/lat input, km for projected).
     * @param min_pts   Minimum number of points (including query point) to
     *                  form a dense region (core point criterion).
     */
    DBSCAN(double eps, int min_pts);

    /**
     * Fit the model to a set of points.
     * Returns a label for every input point.
     *   -1  → noise
     *   >= 0 → cluster id
     */
    std::vector<int> fit(const std::vector<Point2D>& points);

    // After fit(), these accessors are valid.
    int num_clusters()     const { return num_clusters_; }
    int num_noise_points() const { return num_noise_; }

    /**
     * Returns indices of core points (those with >= min_pts neighbours
     * within eps, including themselves).
     */
    const std::vector<int>& core_point_indices() const { return core_pts_; }

private:
    double eps_;
    int    min_pts_;
    int    num_clusters_ = 0;
    int    num_noise_    = 0;

    std::vector<int> core_pts_;

    // Expand a cluster starting at seed_idx.
    void expand_cluster(int                        seed_idx,
                        int                        cluster_id,
                        std::vector<int>&          labels,
                        const KDTree&              tree,
                        const std::vector<Point2D>& points);
};

// ---------------------------------------------------------------------------
// ClusterStats – summary statistics for a single cluster
// ---------------------------------------------------------------------------
struct ClusterStats {
    int    cluster_id;
    int    size;
    double centroid_lon;
    double centroid_lat;
    double spread_deg;    // std-dev of distance from centroid
};

// Compute per-cluster statistics after fitting.
std::vector<ClusterStats> compute_cluster_stats(
    const std::vector<Point2D>& points,
    const std::vector<int>&     labels);

}  // namespace texas_crime
