/**
 * tests/cpp/test_kdtree.cpp
 * Minimal self-contained unit tests for the KD-Tree.
 * Compile & run via CMake: cmake --build . && ctest
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../../src/cpp/spatial_index/kdtree.h"

using namespace texas_crime;

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
    std::printf("PASS: %s\n", msg);
}

// ---------------------------------------------------------------------------
// Test 1: single point
// ---------------------------------------------------------------------------
static void test_single_point() {
    KDTree tree({Point2D(1.0, 2.0, 0)});
    auto nn = tree.knn(Point2D(1.0, 2.0), 1);
    expect(nn.size() == 1, "single point knn size == 1");
    expect(std::fabs(nn[0].dist) < 1e-9, "single point dist == 0");
}

// ---------------------------------------------------------------------------
// Test 2: kNN returns correct nearest points
// ---------------------------------------------------------------------------
static void test_knn() {
    std::vector<Point2D> pts = {
        {0.0, 0.0, 0}, {1.0, 0.0, 1}, {0.0, 1.0, 2},
        {5.0, 5.0, 3}, {-1.0, -1.0, 4},
    };
    KDTree tree(pts);

    auto nn = tree.knn(Point2D(0.1, 0.1), 2);
    expect(nn.size() == 2, "knn k=2 returns 2 results");
    expect(nn[0].point.id == 0, "knn nearest is origin point");
    expect(nn[0].dist < nn[1].dist, "knn results sorted by distance");
}

// ---------------------------------------------------------------------------
// Test 3: range search
// ---------------------------------------------------------------------------
static void test_range_search() {
    std::vector<Point2D> pts = {
        {0.0, 0.0, 0}, {0.5, 0.0, 1}, {0.5, 0.5, 2},
        {5.0, 5.0, 3}, {10.0, 10.0, 4},
    };
    KDTree tree(pts);

    auto in_range = tree.range_search(Point2D(0.0, 0.0), 1.0);
    expect(in_range.size() == 3, "range_search finds 3 within radius 1.0");

    int count = tree.count_within(Point2D(0.0, 0.0), 1.0);
    expect(count == 3, "count_within == 3");
}

// ---------------------------------------------------------------------------
// Test 4: haversine distance
// ---------------------------------------------------------------------------
static void test_haversine() {
    Point2D houston(-95.37, 29.76);
    Point2D dallas (-96.80, 32.78);
    double dist = haversine_km(houston, dallas);
    // known value ~392 km
    expect(dist > 350 && dist < 430, "haversine Houston–Dallas approx 392 km");
}

// ---------------------------------------------------------------------------
// Test 5: size() after build
// ---------------------------------------------------------------------------
static void test_size() {
    std::vector<Point2D> pts(100);
    for (int i = 0; i < 100; ++i) pts[i] = Point2D(i * 0.1, i * 0.1, i);
    KDTree tree(pts);
    expect(tree.size() == 100, "tree.size() == 100");
}

int main() {
    test_single_point();
    test_knn();
    test_range_search();
    test_haversine();
    test_size();
    std::printf("\nAll KD-Tree tests passed.\n");
    return 0;
}
