/**
 * kdtree.cpp
 * KD-Tree implementation for 2-D spatial queries.
 */

#include "kdtree.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace texas_crime {

// ---------------------------------------------------------------------------
// KDTree  –  constructor
// ---------------------------------------------------------------------------
KDTree::KDTree(std::vector<Point2D> pts) {
    if (pts.empty()) return;

    nodes_.resize(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i)
        nodes_[i].point = pts[i];

    // Build index array then recursively partition.
    std::vector<int> indices(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) indices[i] = static_cast<int>(i);

    build(indices, 0, static_cast<int>(indices.size()), 0);
}

// ---------------------------------------------------------------------------
// build  –  median-of-medians style recursive builder
// Returns the node index of the subtree root (same as indices[mid]).
// ---------------------------------------------------------------------------
int KDTree::build(std::vector<int>& indices, int lo, int hi, int depth) {
    if (lo >= hi) return -1;

    int axis = depth % 2;
    int mid  = lo + (hi - lo) / 2;

    std::nth_element(indices.begin() + lo,
                     indices.begin() + mid,
                     indices.begin() + hi,
                     [&](int a, int b) {
                         return nodes_[a].point[axis] < nodes_[b].point[axis];
                     });

    int root = indices[mid];
    nodes_[root].depth = depth;
    nodes_[root].left  = build(indices, lo, mid, depth + 1);
    nodes_[root].right = build(indices, mid + 1, hi, depth + 1);

    return root;
}

// ---------------------------------------------------------------------------
// knn
// ---------------------------------------------------------------------------
std::vector<KDTree::NNResult> KDTree::knn(const Point2D& query, int k) const {
    if (k <= 0) throw std::invalid_argument("k must be > 0");
    if (nodes_.empty()) return {};

    std::vector<HeapEntry> heap;
    heap.reserve(k + 1);

    // Start recursion from the virtual root built during construction.
    // After build(), nodes_[0] is NOT necessarily the root – the root is
    // the median of the full index range, stored at its original position.
    // We encoded the tree by mutating left/right pointers on the node array
    // starting at node 0's position.  Re-derive root as the node with
    // depth == 0 that has no parent (linear scan is cheap for small n; for
    // large n the root is always the global median = indices[mid] from the
    // first call).  We store root as first element touched in depth-0 pass.
    // Simpler: re-build root index by finding depth-0 node.
    int root = -1;
    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
        if (nodes_[i].depth == 0) { root = i; break; }
    }
    if (root == -1) return {};

    knn_rec(root, query, k, heap);

    // Sort heap by ascending distance.
    std::sort(heap.begin(), heap.end());
    std::vector<NNResult> results;
    results.reserve(heap.size());
    for (auto& h : heap)
        results.push_back({h.point, std::sqrt(h.dist_sq)});

    return results;
}

void KDTree::knn_rec(int idx,
                     const Point2D& query,
                     int k,
                     std::vector<HeapEntry>& heap) const {
    if (idx < 0) return;

    const Node& node = nodes_[idx];
    double dsq = node.point.dist_sq(query);

    // Maintain max-heap of size k (keep k smallest).
    if (static_cast<int>(heap.size()) < k) {
        heap.push_back({dsq, node.point});
        std::push_heap(heap.begin(), heap.end());
    } else if (dsq < heap.front().dist_sq) {
        std::pop_heap(heap.begin(), heap.end());
        heap.back() = {dsq, node.point};
        std::push_heap(heap.begin(), heap.end());
    }

    int axis      = node.depth % 2;
    double diff   = query[axis] - node.point[axis];
    int near_idx  = diff <= 0 ? node.left : node.right;
    int far_idx   = diff <= 0 ? node.right : node.left;

    knn_rec(near_idx, query, k, heap);

    // Prune far subtree if it cannot improve.
    double worst = (static_cast<int>(heap.size()) < k)
                       ? std::numeric_limits<double>::max()
                       : heap.front().dist_sq;
    if (diff * diff < worst)
        knn_rec(far_idx, query, k, heap);
}

// ---------------------------------------------------------------------------
// range_search
// ---------------------------------------------------------------------------
std::vector<Point2D> KDTree::range_search(const Point2D& query, double radius) const {
    std::vector<Point2D> results;
    if (nodes_.empty() || radius <= 0) return results;

    int root = -1;
    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
        if (nodes_[i].depth == 0) { root = i; break; }
    }
    if (root == -1) return results;

    range_rec(root, query, radius * radius, results);
    return results;
}

void KDTree::range_rec(int idx,
                       const Point2D& query,
                       double radius_sq,
                       std::vector<Point2D>& results) const {
    if (idx < 0) return;

    const Node& node = nodes_[idx];
    if (node.point.dist_sq(query) <= radius_sq)
        results.push_back(node.point);

    int    axis = node.depth % 2;
    double diff = query[axis] - node.point[axis];

    // Always visit nearer side; visit farther only if plane intersects sphere.
    int near_idx = diff <= 0 ? node.left : node.right;
    int far_idx  = diff <= 0 ? node.right : node.left;

    range_rec(near_idx, query, radius_sq, results);
    if (diff * diff <= radius_sq)
        range_rec(far_idx, query, radius_sq, results);
}

// ---------------------------------------------------------------------------
// count_within
// ---------------------------------------------------------------------------
int KDTree::count_within(const Point2D& query, double radius) const {
    if (nodes_.empty() || radius <= 0) return 0;
    int root = -1;
    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
        if (nodes_[i].depth == 0) { root = i; break; }
    }
    if (root == -1) return 0;
    return count_rec(root, query, radius * radius);
}

int KDTree::count_rec(int idx, const Point2D& query, double radius_sq) const {
    if (idx < 0) return 0;

    const Node& node  = nodes_[idx];
    int         count = (node.point.dist_sq(query) <= radius_sq) ? 1 : 0;

    int    axis     = node.depth % 2;
    double diff     = query[axis] - node.point[axis];
    int    near_idx = diff <= 0 ? node.left : node.right;
    int    far_idx  = diff <= 0 ? node.right : node.left;

    count += count_rec(near_idx, query, radius_sq);
    if (diff * diff <= radius_sq)
        count += count_rec(far_idx, query, radius_sq);

    return count;
}

}  // namespace texas_crime
