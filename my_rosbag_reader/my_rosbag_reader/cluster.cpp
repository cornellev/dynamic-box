#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <set>
#include <tuple>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

namespace py = pybind11;
using namespace Eigen;
using namespace std;

// ======================================================
// Self-contained KdTree for Vector4d
// ======================================================
template<typename PointT>
class KdTree {
public:
    KdTree() : dim_(3) {}

    KdTree(const vector<PointT>& points, int dim = -1) {
        build(points, dim);
    }

    void build(const vector<PointT>& points, int dim = -1) {
        clear();
        points_ = points;
        dim_ = (dim > 0) ? dim : 3;

        vector<int> indices(points.size());
        iota(indices.begin(), indices.end(), 0);

        root_ = build_recursive(indices.data(), static_cast<int>(points.size()), 0);
    }

    void clear() {
        root_.reset();
        points_.clear();
    }

    struct Node {
        int idx = -1;
        array<unique_ptr<Node>, 2> next;
        int axis = -1;
    };

    int search(const PointT& query, double* minDist = nullptr) const {
        int guess = -1;
        double _minDist = numeric_limits<double>::max();
        search_recursive(query, root_.get(), &guess, &_minDist);
        if (minDist) *minDist = _minDist;
        return guess;
    }

private:
    unique_ptr<Node> build_recursive(int* indices, int npoints, int depth) {
        if (npoints <= 0) return nullptr;
        int axis = depth % dim_;
        int mid = (npoints - 1) / 2;

        nth_element(indices, indices + mid, indices + npoints,
                    [&](int lhs, int rhs){ return points_[lhs][axis] < points_[rhs][axis]; });

        auto node = make_unique<Node>();
        node->idx = indices[mid];
        node->axis = axis;
        node->next[0] = build_recursive(indices, mid, depth + 1);
        node->next[1] = build_recursive(indices + mid + 1, npoints - mid - 1, depth + 1);
        return node;
    }

    static double distance(const PointT& p, const PointT& q) {
        double dist = 0;
        for (int i = 0; i < p.size(); i++) dist += (p[i] - q[i])*(p[i] - q[i]);
        return sqrt(dist);
    }

    void search_recursive(const PointT& query, const Node* node, int* guess, double* minDist) const {
        if (!node) return;
        const PointT& train = points_[node->idx];
        double dist = distance(query, train);
        if (dist < *minDist) { *minDist = dist; *guess = node->idx; }

        int axis = node->axis;
        int dir = query[axis] < train[axis] ? 0 : 1;
        search_recursive(query, node->next[dir].get(), guess, minDist);

        double diff = fabs(query[axis] - train[axis]);
        if (diff < *minDist) search_recursive(query, node->next[dir==0].get(), guess, minDist);
    }

    unique_ptr<Node> root_;
    vector<PointT> points_;
    int dim_;
};

// ======================================================
// Euclidean clustering function
// ======================================================
py::tuple euclidean_cluster(const Vector4d seeds,
                            const MatrixXd& cloud_input,
                            double radius,
                            int MIN_CLUSTER_SIZE = 1,
                            const string& mode = "cartesian",
                            const vector<Vector4d>& cloud_prev = {},
                            bool reorder = true, 
                            double MAX_CLUSTER_NUM = numeric_limits<double>::infinity()
                            ) {
    // Reorder cloud if needed
    VectorXd x, y, z;
    if (reorder) {
        if (mode == "spherical") {
            z = cloud_input.col(0).array() * (cloud_input.col(2).array().sin() * cloud_input.col(1).array().cos());
            x = cloud_input.col(0).array() * (cloud_input.col(2).array().sin() * cloud_input.col(1).array().sin());
            y = cloud_input.col(0).array() * cloud_input.col(2).array().cos();
        } else {
            x = cloud_input.col(0);
            y = cloud_input.col(1);
            z = cloud_input.col(2);
        }
    } else {
        x = cloud_input.col(0);
        y = cloud_input.col(1);
        z = cloud_input.col(2);
    }

    MatrixXd cloud(cloud_input.rows(), 4);
    cloud.col(0) = x;
    cloud.col(1) = y;
    cloud.col(2) = z;
    cloud.col(3) = cloud_input.col(3);

    vector<Vector4d> points(cloud.rows());
    for (int i = 0; i < cloud.rows(); ++i) points[i] = cloud.row(i).transpose();

    // Build KD-tree
    KdTree<Vector4d> kd_tree(points);

    // Initialize unexplored set
    set<tuple<double,double,double>> unexplored;
    for (const auto& p : points)
        unexplored.insert(make_tuple(p[0], p[1], p[2]));

    vector<vector<Vector4d>> C;
    vector<Vector4d> prev;

    vector<vector<Vector4d>> cloud_prev_clusters;
    if (!cloud_prev.empty()) cloud_prev_clusters.push_back(cloud_prev);

    int iter = 0;
    while (!unexplored.empty() && C.size() < MAX_CLUSTER_NUM) {
        Vector4d next_point;
        if (iter == 0) next_point = seeds;
        else {
            auto it = unexplored.begin();
            next_point << get<0>(*it), get<1>(*it), get<2>(*it), 0.0;
        }

        C.push_back({next_point});
        unexplored.erase(make_tuple(next_point[0], next_point[1], next_point[2]));

        vector<Vector4d> stack = {next_point};
        while (!stack.empty()) {
            Vector4d query = stack.back();
            stack.pop_back();

            for (const auto& p : points) {
                Vector3d diff = (p.head<3>() - query.head<3>()).cwiseAbs();
                double local_radius = radius;
                if (query.head<3>().norm() > 5.0) local_radius *= 2;

                if (diff[0] < local_radius && diff[1] < local_radius && diff[2] < 2*local_radius) {
                    tuple<double,double,double> key = make_tuple(p[0], p[1], p[2]);
                    if (unexplored.find(key) != unexplored.end()) {
                        C.back().push_back(p);
                        stack.push_back(p);
                        unexplored.erase(key);
                    }
                }
            }
        }

        prev.push_back(Vector4d::Zero());
        iter++;
    }

    // eventually just directly publish points from here
    py::list py_clusters;
    for (auto& c : C) {
        MatrixXd mat(c.size(), 4);
        for (size_t i = 0; i < c.size(); ++i) mat.row(i) = c[i];
        py_clusters.append(py::cast(mat));
    }

    py::list py_prevs;
    for (auto& v : prev) py_prevs.append(py::cast(v));

    return py::make_tuple(py_clusters, py_prevs);
}

// ======================================================
// Pybind11 module
// ======================================================
PYBIND11_MODULE(cluster_cpp, m) {
    m.def("euclidean_cluster", &euclidean_cluster,
          py::arg("seeds"),
          py::arg("cloud"),
          py::arg("radius"),
          py::arg("MIN_CLUSTER_SIZE") = 1,
          py::arg("mode") = "cartesian",
          py::arg("cloud_prev") = std::vector<Vector4d>{},
          py::arg("reorder") = true,
          py::arg("MAX_CLUSTER_NUM") = numeric_limits<double>::infinity(),
          "Cluster a point cloud using internal KdTree");
}
