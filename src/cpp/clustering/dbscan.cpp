/**
 * dbscan.cpp
 * DBSCAN spatial clustering implementation.
 */

#include "dbscan.h"

#include <cmath>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace texas_crime {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int UNVISITED = -2;
constexpr int NOISE_ID  = -1;

// ---------------------------------------------------------------------------
// DBSCAN constructor
// ---------------------------------------------------------------------------
DBSCAN::DBSCAN(double eps, int min_pts) : eps_(eps), min_pts_(min_pts) {
    if (eps <= 0)     throw std::invalid_argument("eps must be > 0");
    if (min_pts < 1)  throw std::invalid_argument("min_pts must be >= 1");
}

// ---------------------------------------------------------------------------
// fit
// ---------------------------------------------------------------------------
std::vector<int> DBSCAN::fit(const std::vector<Point2D>& points) {
    const int n = static_cast<int>(points.size());
    if (n == 0) return {};

    // Build KD-Tree.
    KDTree tree(points);

    std::vector<int> labels(n, UNVISITED);
    num_clusters_ = 0;
    num_noise_    = 0;
    core_pts_.clear();

    for (int i = 0; i < n; ++i) {
        if (labels[i] != UNVISITED) continue;

        auto neighbours = tree.range_search(points[i], eps_);

        if (static_cast<int>(neighbours.size()) < min_pts_) {
            labels[i] = NOISE_ID;
            ++num_noise_;
            continue;
        }

        // Start a new cluster.
        int cid = num_clusters_++;
        labels[i] = cid;
        core_pts_.push_back(i);

        expand_cluster(i, cid, labels, tree, points);
    }

    // Final noise count (some NOISE points may get absorbed during expansion).
    num_noise_ = 0;
    for (int l : labels) if (l == NOISE_ID) ++num_noise_;

    return labels;
}

// ---------------------------------------------------------------------------
// expand_cluster
// ---------------------------------------------------------------------------
void DBSCAN::expand_cluster(int                         seed_idx,
                             int                         cluster_id,
                             std::vector<int>&           labels,
                             const KDTree&               tree,
                             const std::vector<Point2D>& points) {
    // BFS queue of point indices to process.
    std::queue<int> q;
    q.push(seed_idx);

    while (!q.empty()) {
        int cur = q.front();
        q.pop();

        auto neighbours = tree.range_search(points[cur], eps_);
        if (static_cast<int>(neighbours.size()) < min_pts_) continue;

        // cur is a core point.
        for (const auto& nb : neighbours) {
            int nb_id = nb.id;
            if (nb_id < 0) continue;  // safety

            if (labels[nb_id] == UNVISITED || labels[nb_id] == NOISE_ID) {
                if (labels[nb_id] == UNVISITED) q.push(nb_id);
                labels[nb_id] = cluster_id;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// compute_cluster_stats
// ---------------------------------------------------------------------------
std::vector<ClusterStats> compute_cluster_stats(
    const std::vector<Point2D>& points,
    const std::vector<int>&     labels) {

    if (points.size() != labels.size())
        throw std::invalid_argument("points and labels must have the same size");

    // Find distinct cluster ids (ignore noise = -1).
    std::unordered_set<int> ids_set;
    for (int l : labels) if (l >= 0) ids_set.insert(l);

    int max_id = ids_set.empty() ? -1 : *std::max_element(ids_set.begin(), ids_set.end());
    int nc     = max_id + 1;

    std::vector<double> sum_x(nc, 0), sum_y(nc, 0);
    std::vector<int>    cnt(nc, 0);

    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        int l = labels[i];
        if (l < 0) continue;
        sum_x[l] += points[i].x;
        sum_y[l] += points[i].y;
        cnt[l]   += 1;
    }

    std::vector<ClusterStats> stats;
    stats.reserve(nc);

    for (int c = 0; c < nc; ++c) {
        if (cnt[c] == 0) continue;

        ClusterStats s;
        s.cluster_id   = c;
        s.size         = cnt[c];
        s.centroid_lon = sum_x[c] / cnt[c];
        s.centroid_lat = sum_y[c] / cnt[c];

        // Spread: root-mean-square distance from centroid.
        double var = 0;
        Point2D centroid(s.centroid_lon, s.centroid_lat);
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            if (labels[i] != c) continue;
            var += points[i].dist_sq(centroid);
        }
        s.spread_deg = std::sqrt(var / cnt[c]);
        stats.push_back(s);
    }

    return stats;
}

}  // namespace texas_crime
