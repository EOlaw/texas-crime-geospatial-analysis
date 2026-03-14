/**
 * tests/cpp/test_dbscan.cpp
 * Unit tests for C++ DBSCAN clustering.
 */

#include <cassert>
#include <cstdio>
#include <set>
#include <vector>

#include "../../src/cpp/clustering/dbscan.h"

using namespace texas_crime;

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
    std::printf("PASS: %s\n", msg);
}

// ---------------------------------------------------------------------------
// Helper: build a dense cluster of n points around (cx, cy)
// ---------------------------------------------------------------------------
static std::vector<Point2D> make_cluster(double cx, double cy, int n, int id_offset) {
    std::vector<Point2D> pts;
    double step = 0.005;
    for (int i = 0; i < n; ++i)
        pts.push_back(Point2D(cx + (i % 5) * step, cy + (i / 5) * step, id_offset + i));
    return pts;
}

// ---------------------------------------------------------------------------
// Test 1: two distinct clusters + noise
// ---------------------------------------------------------------------------
static void test_two_clusters() {
    std::vector<Point2D> pts;
    auto c1 = make_cluster(0.0, 0.0, 20, 0);
    auto c2 = make_cluster(5.0, 5.0, 20, 20);
    pts.insert(pts.end(), c1.begin(), c1.end());
    pts.insert(pts.end(), c2.begin(), c2.end());
    pts.push_back(Point2D(100.0, 100.0, 40));  // noise

    DBSCAN db(0.05, 3);
    auto labels = db.fit(pts);

    expect(db.num_clusters() == 2, "two_clusters: num_clusters == 2");
    expect(db.num_noise_points() == 1, "two_clusters: 1 noise point");
    expect(labels[40] == -1, "two_clusters: distant point is noise");
}

// ---------------------------------------------------------------------------
// Test 2: all noise (eps too small)
// ---------------------------------------------------------------------------
static void test_all_noise() {
    std::vector<Point2D> pts = {
        {0.0, 0.0, 0}, {1.0, 0.0, 1}, {0.0, 1.0, 2},
    };
    DBSCAN db(0.001, 5);   // eps very small, min_pts > cluster size
    auto labels = db.fit(pts);

    expect(db.num_clusters() == 0, "all_noise: 0 clusters");
    expect(db.num_noise_points() == 3, "all_noise: all are noise");
}

// ---------------------------------------------------------------------------
// Test 3: single cluster
// ---------------------------------------------------------------------------
static void test_single_cluster() {
    auto pts = make_cluster(0.0, 0.0, 15, 0);
    DBSCAN db(0.1, 5);
    auto labels = db.fit(pts);

    expect(db.num_clusters() == 1, "single_cluster: 1 cluster found");
    expect(db.num_noise_points() == 0, "single_cluster: 0 noise");
}

// ---------------------------------------------------------------------------
// Test 4: cluster_stats correctness
// ---------------------------------------------------------------------------
static void test_cluster_stats() {
    auto pts = make_cluster(1.0, 2.0, 16, 0);
    DBSCAN db(0.1, 4);
    auto labels = db.fit(pts);

    auto stats = compute_cluster_stats(pts, labels);
    expect(!stats.empty(), "cluster_stats: at least one entry");

    const auto& s = stats[0];
    // Centroid should be near (1.0, 2.0)
    expect(std::fabs(s.centroid_lon - 1.0) < 0.1, "centroid_lon near 1.0");
    expect(std::fabs(s.centroid_lat - 2.0) < 0.1, "centroid_lat near 2.0");
    expect(s.size > 0, "cluster size > 0");
}

int main() {
    test_two_clusters();
    test_all_noise();
    test_single_cluster();
    test_cluster_stats();
    std::printf("\nAll DBSCAN tests passed.\n");
    return 0;
}
