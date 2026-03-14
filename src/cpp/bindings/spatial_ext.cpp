/**
 * spatial_ext.cpp
 * pybind11 bindings exposing the C++ spatial index and clustering
 * components to Python as the module `texas_crime_spatial`.
 *
 * Build with CMake (see CMakeLists.txt) or via setup.py (see setup.py).
 *
 * Python usage example:
 *   import texas_crime_spatial as tcs
 *
 *   pts = [tcs.Point2D(x=-97.7, y=30.3, id=0),
 *          tcs.Point2D(x=-97.6, y=30.2, id=1)]
 *
 *   tree = tcs.KDTree(pts)
 *   nn   = tree.knn(tcs.Point2D(-97.65, 30.25), k=1)
 *   print(nn[0].point.x, nn[0].dist)
 *
 *   db     = tcs.DBSCAN(eps=0.05, min_pts=3)
 *   labels = db.fit(pts)
 *   print(db.num_clusters, db.num_noise_points)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../spatial_index/kdtree.h"
#include "../clustering/dbscan.h"

namespace py = pybind11;
using namespace texas_crime;

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(texas_crime_spatial, m) {
    m.doc() = "C++ spatial indexing and clustering extensions for Texas crime analysis";

    // -----------------------------------------------------------------------
    // Point2D
    // -----------------------------------------------------------------------
    py::class_<Point2D>(m, "Point2D")
        .def(py::init<>())
        .def(py::init<double, double, int>(),
             py::arg("x"), py::arg("y"), py::arg("id") = -1,
             "Create a 2-D point. x=longitude, y=latitude, id=row index.")
        .def_readwrite("x",  &Point2D::x,  "Longitude")
        .def_readwrite("y",  &Point2D::y,  "Latitude")
        .def_readwrite("id", &Point2D::id, "Original dataset row index")
        .def("dist_sq", &Point2D::dist_sq,
             py::arg("other"),
             "Squared Euclidean distance to another point.")
        .def("__repr__", [](const Point2D& p) {
            return "<Point2D lon=" + std::to_string(p.x) +
                   " lat=" + std::to_string(p.y) +
                   " id="  + std::to_string(p.id) + ">";
        });

    // -----------------------------------------------------------------------
    // KDTree::NNResult
    // -----------------------------------------------------------------------
    py::class_<KDTree::NNResult>(m, "NNResult")
        .def_readwrite("point", &KDTree::NNResult::point, "Nearest point")
        .def_readwrite("dist",  &KDTree::NNResult::dist,  "Euclidean distance")
        .def("__repr__", [](const KDTree::NNResult& r) {
            return "<NNResult dist=" + std::to_string(r.dist) + ">";
        });

    // -----------------------------------------------------------------------
    // KDTree
    // -----------------------------------------------------------------------
    py::class_<KDTree>(m, "KDTree")
        .def(py::init<std::vector<Point2D>>(),
             py::arg("points"),
             "Build a KD-tree from a list of Point2D objects.")
        .def("knn", &KDTree::knn,
             py::arg("query"), py::arg("k"),
             "Return the k nearest neighbours to query, sorted by distance.")
        .def("range_search", &KDTree::range_search,
             py::arg("query"), py::arg("radius"),
             "Return all points within radius of query (same coordinate units).")
        .def("count_within", &KDTree::count_within,
             py::arg("query"), py::arg("radius"),
             "Count points within radius without collecting them.")
        .def("__len__", &KDTree::size,
             "Number of points stored in the tree.");

    // -----------------------------------------------------------------------
    // DBSCAN
    // -----------------------------------------------------------------------
    py::class_<DBSCAN>(m, "DBSCAN")
        .def(py::init<double, int>(),
             py::arg("eps"), py::arg("min_pts"),
             "Density-based spatial clusterer.\n"
             "eps      – neighbourhood radius (coordinate units)\n"
             "min_pts  – minimum neighbour count for a core point")
        .def("fit", &DBSCAN::fit,
             py::arg("points"),
             "Fit to list of Point2D. Returns list of integer cluster labels "
             "(-1 = noise, >=0 = cluster id).")
        .def_property_readonly("num_clusters",
             &DBSCAN::num_clusters,
             "Number of clusters found (excluding noise).")
        .def_property_readonly("num_noise_points",
             &DBSCAN::num_noise_points,
             "Number of noise points after fitting.")
        .def_property_readonly("core_point_indices",
             &DBSCAN::core_point_indices,
             "Indices of core points in the fitted dataset.");

    // -----------------------------------------------------------------------
    // ClusterStats
    // -----------------------------------------------------------------------
    py::class_<ClusterStats>(m, "ClusterStats")
        .def_readwrite("cluster_id",   &ClusterStats::cluster_id)
        .def_readwrite("size",         &ClusterStats::size)
        .def_readwrite("centroid_lon", &ClusterStats::centroid_lon)
        .def_readwrite("centroid_lat", &ClusterStats::centroid_lat)
        .def_readwrite("spread_deg",   &ClusterStats::spread_deg)
        .def("__repr__", [](const ClusterStats& s) {
            return "<ClusterStats id=" + std::to_string(s.cluster_id) +
                   " size=" + std::to_string(s.size) + ">";
        });

    m.def("compute_cluster_stats",
          &compute_cluster_stats,
          py::arg("points"), py::arg("labels"),
          "Compute per-cluster statistics (centroid, spread) from fit() output.");

    // -----------------------------------------------------------------------
    // Haversine utility
    // -----------------------------------------------------------------------
    m.def("haversine_km",
          &haversine_km,
          py::arg("a"), py::arg("b"),
          "Haversine great-circle distance in kilometres between two Point2D objects.");
}
