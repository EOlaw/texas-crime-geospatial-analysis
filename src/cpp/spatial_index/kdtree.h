#pragma once
/**
 * kdtree.h
 * 2-D KD-Tree for fast spatial nearest-neighbour and range queries.
 * Used to accelerate crime incident proximity analysis and hotspot detection.
 *
 * Coordinate convention:  point[0] = longitude (x), point[1] = latitude (y)
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace texas_crime {

// ---------------------------------------------------------------------------
// Point2D
// ---------------------------------------------------------------------------
struct Point2D {
    double x;   // longitude
    double y;   // latitude
    int    id;  // original dataset row index

    Point2D() : x(0), y(0), id(-1) {}
    Point2D(double x_, double y_, int id_ = -1) : x(x_), y(y_), id(id_) {}

    double operator[](int dim) const { return dim == 0 ? x : y; }

    double dist_sq(const Point2D& o) const {
        double dx = x - o.x, dy = y - o.y;
        return dx * dx + dy * dy;
    }
};

// ---------------------------------------------------------------------------
// KDTree
// ---------------------------------------------------------------------------
class KDTree {
public:
    struct NNResult {
        Point2D point;
        double  dist;   // Euclidean distance
    };

    // Build from a copy of the point set.
    explicit KDTree(std::vector<Point2D> pts);

    // k-Nearest-Neighbour search. Returns up to k results sorted by distance.
    std::vector<NNResult> knn(const Point2D& query, int k) const;

    // Range search: all points within radius (same units as coordinates).
    std::vector<Point2D> range_search(const Point2D& query, double radius) const;

    // Count points within radius (cheaper than collecting them).
    int count_within(const Point2D& query, double radius) const;

    std::size_t size() const { return nodes_.size(); }

private:
    struct Node {
        Point2D point;
        int     left  = -1;
        int     right = -1;
        int     depth = 0;
    };

    std::vector<Node> nodes_;

    // Build helpers
    int build(std::vector<int>& indices, int lo, int hi, int depth);

    // kNN helpers
    struct HeapEntry {
        double  dist_sq;
        Point2D point;
        bool operator<(const HeapEntry& o) const { return dist_sq < o.dist_sq; }
    };

    void knn_rec(int node_idx,
                 const Point2D& query,
                 int k,
                 std::vector<HeapEntry>& heap) const;

    void range_rec(int node_idx,
                   const Point2D& query,
                   double radius_sq,
                   std::vector<Point2D>& results) const;

    int count_rec(int node_idx,
                  const Point2D& query,
                  double radius_sq) const;
};

// ---------------------------------------------------------------------------
// Haversine distance (kilometres) – useful for real-world spatial checks
// ---------------------------------------------------------------------------
inline double haversine_km(const Point2D& a, const Point2D& b) {
    constexpr double R  = 6371.0;  // Earth radius km
    constexpr double PI = 3.14159265358979323846;
    auto to_rad = [&](double d) { return d * PI / 180.0; };

    double lat1 = to_rad(a.y), lat2 = to_rad(b.y);
    double dlat = to_rad(b.y - a.y);
    double dlon = to_rad(b.x - a.x);

    double s = std::sin(dlat / 2) * std::sin(dlat / 2) +
               std::cos(lat1) * std::cos(lat2) *
               std::sin(dlon / 2) * std::sin(dlon / 2);

    return 2.0 * R * std::atan2(std::sqrt(s), std::sqrt(1.0 - s));
}

}  // namespace texas_crime
