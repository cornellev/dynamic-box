#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace Eigen;
using namespace std;

struct Node {
    MatrixXd data;           // Nx4 matrix of points [x,y,z,I]
    int axis = 0;
    Node* left = nullptr;
    Node* right = nullptr;
    Vector4d prev = Vector4d::Zero();

    Node() {}

    Node(const MatrixXd& points, int a, Node* left_n, Node* right_n) {
        data = points;
        axis = a;
        left = left_n;
        right = right_n;
    }

    static set<tuple<double, double, double>> unexplored;

    // Faster KD-Tree construction using partial sort
    Node* make_kdtree(const MatrixXd& points, int axis, int dim) {
        if (points.rows() <= 10) {
            // left = nullptr, right = nullptr is a Leaf
            return (points.rows() > 5) ? new Node(points, axis, nullptr, nullptr) : new Node();
        }

        // Find median index using partial sort (np.argpartition equivalent)
        vector<int> indices(points.rows());
        iota(indices.begin(), indices.end(), 0);
        int median_id = points.rows() / 2;
        nth_element(indices.begin(), indices.begin() + median_id, indices.end(),
                    [&](int i, int j){ return points(i, axis) < points(j, axis); });
        median_id = indices[median_id];
        double median = points(median_id, axis);

        // Partition points into left/right matrices
        vector<int> left_idx, right_idx;
        for (int i = 0; i < points.rows(); ++i) {
            if (points(i, axis) < median) left_idx.push_back(i);
            else right_idx.push_back(i);
        }

        MatrixXd left_pts(left_idx.size(), points.cols());
        for (size_t i = 0; i < left_idx.size(); ++i)
            left_pts.row(i) = points.row(left_idx[i]);

        MatrixXd right_pts(right_idx.size(), points.cols());
        for (size_t i = 0; i < right_idx.size(); ++i)
            right_pts.row(i) = points.row(right_idx[i]);

        return new Node(points, axis,
                        make_kdtree(left_pts, (axis + 1) % dim, dim),
                        make_kdtree(right_pts, (axis + 1) % dim, dim));
    }

    vector<MatrixXd> search_point(const vector<MatrixXd>& point, double radius,
                                const vector<MatrixXd>& C, const vector<MatrixXd>& C_prev) {
        // Extract the column into a std::vector
        VectorXd col = data.col(axis);
        std::vector<double> v(col.data(), col.data() + col.size());

        // Find median index
        size_t mid = v.size() / 2;

        // Partially sort to get the median
        std::nth_element(v.begin(), v.begin() + mid, v.end());

        // Assign the median value
        double split_axis = v[mid];
        // approximate median of the axis column
        
        if (point[0].row(0).head<3>().transpose().norm() > 5) {
            radius = radius * 2;
        }

        vector<MatrixXd> result;
        if (data.rows() > 0) {
            // Compute differences between all points and point[0] in first 3 dims
            MatrixXd diff = (data.leftCols(3)).rowwise() - point[0].row(0).leftCols(3);

            // Compute mask where all abs(diff) < [radius, radius, 2*radius]
            vector<int> mask_indices;
            for (int i = 0; i < diff.rows(); ++i) {
                Vector3d d = diff.row(i).transpose().cwiseAbs();
                if (d(0) < radius && d(1) < radius && d(2) < 2 * radius) {
                    mask_indices.push_back(i);
                }
            }

            if (!mask_indices.empty()) {
                MatrixXd in_radius(mask_indices.size(), data.cols());
                for (size_t i = 0; i < mask_indices.size(); ++i)
                    in_radius.row(i) = data.row(mask_indices[i]);

                result.push_back(point[0]);
                result.push_back(in_radius);
                return result;
            }
        }

        if (point[0](0, axis) - radius < split_axis) {
            if (left)
                return left->search_point(point, radius, C, C_prev);
        } 
        else if (point[0](0, axis) + radius >= split_axis) {
            if (right)
                return right->search_point(point, radius, C, C_prev);
        }

        return point;
    }


    vector<Vector4d> search_tree(Node* root, const Vector4d& start_point,
                                double radius,
                                vector<vector<Vector4d>>& C,
                                const vector<vector<Vector4d>>& C_prev) {
        // Initialize stack with start_point
        vector<Vector4d> stack = {start_point};

        // Convert unexplored points to a set of tuples
        set<tuple<double,double,double>> unexplored_set;
        for (const auto& p : Node::unexplored) {
            unexplored_set.insert(make_tuple(get<0>(p), get<1>(p), get<2>(p)));
        }

        while (!stack.empty()) {
            // Pop last point from stack
            Vector4d point = stack.back();
            stack.pop_back();

            // Convert C to vector<MatrixXd> for search_point
            vector<MatrixXd> C_matrices;
            for (auto& cluster : C) {
                MatrixXd mat(cluster.size(), 4);
                for (size_t i = 0; i < cluster.size(); ++i)
                    mat.row(i) = cluster[i];
                C_matrices.push_back(mat);
            }

            // Convert C_prev to vector<MatrixXd> for search_point
            vector<MatrixXd> C_prev_matrices;
            for (auto& cluster : C_prev) {
                MatrixXd mat(cluster.size(), 4);
                for (size_t i = 0; i < cluster.size(); ++i)
                    mat.row(i) = cluster[i];
                C_prev_matrices.push_back(mat);
            }

            // Get neighbors: search_point returns vector<MatrixXd>, take elements from index 1 onward
            vector<MatrixXd> search_result = root->search_point(
                vector<MatrixXd>{point.transpose()},
                radius,
                C_matrices,
                C_prev_matrices
            );

            vector<MatrixXd> neighbors_vec(search_result.begin() + 1, search_result.end());

            // Convert neighbors to tuples and filter those in unexplored_set
            vector<Vector4d> neighbors;
            for (auto& mat : neighbors_vec) {
                for (int i = 0; i < mat.rows(); ++i) {
                    Vector4d n = mat.row(i).transpose();
                    tuple<double,double,double> key = make_tuple(n(0), n(1), n(2));
                    if (unexplored_set.find(key) != unexplored_set.end())
                        neighbors.push_back(n);
                }
            }

            // Explore neighbors
            for (auto& neighbor : neighbors) {
                if (find(C.back().begin(), C.back().end(), neighbor) == C.back().end()) {
                    C.back().push_back(neighbor);    // Add to current cluster
                    stack.push_back(neighbor);       // Add to stack
                    unexplored_set.erase(make_tuple(neighbor(0), neighbor(1), neighbor(2)));
                }
            }
        }

        // Return unexplored points (or could update Node::unexplored externally)
        vector<Vector4d> remaining_unexplored;
        for (auto& t : unexplored_set)
            remaining_unexplored.push_back(Vector4d(get<0>(t), get<1>(t), get<2>(t), 0.0));

        return remaining_unexplored;
    }
};

set<tuple<double, double, double>> Node::unexplored;

py::tuple euclidean_cluster(const Eigen::Vector4d seeds,
                            const MatrixXd& cloud_input,
                            double radius,
                            int MIN_CLUSTER_SIZE = 1,
                            const string& mode = "cartesian",
                            const vector<Vector4d>& cloud_prev = {},
                            bool reorder = true, 
                            double MAX_CLUSTER_NUM = std::numeric_limits<double>::infinity()
                            ) {
    // if MAX_CLUSTER_NUM = inf -> then grow as many clusters as needed
    // otherwise, maximum number of clusters grown = MAX_CLUSTER_NUM
    vector<vector<Vector4d>> C;   // Cluster list
    vector<Vector4d> prev;        // Previous centroids

    VectorXd x, y, z;
    if (reorder) {
        if (mode == "spherical") {
            z = cloud_input.col(0).array() * (cloud_input.col(2).array().sin() * cloud_input.col(1).array().cos());
            x = cloud_input.col(0).array() * (cloud_input.col(2).array().sin() * cloud_input.col(1).array().sin());
            y = cloud_input.col(0).array() * cloud_input.col(2).array().cos();
        } else {
            z = cloud_input.col(0);
            x = cloud_input.col(1);
            y = cloud_input.col(2);
        }
    } else {
        x = cloud_input.col(0);
        y = cloud_input.col(1);
        z = cloud_input.col(2);
    }

    // Reconstruct cloud as Nx4 matrix
    MatrixXd cloud(cloud_input.rows(), 4);
    cloud.col(0) = x;
    cloud.col(1) = y;
    cloud.col(2) = z;
    cloud.col(3) = cloud_input.col(3);

    // Initialize unexplored set
    Node::unexplored.clear();
    for (int i = 0; i < cloud.rows(); ++i)
        Node::unexplored.insert(make_tuple(cloud(i,0), cloud(i,1), cloud(i,2)));

    // Build KD-tree
    Node* kd_tree = Node().make_kdtree(cloud, 0, 3);

    // Wrap cloud_prev as vector<vector<Vector4d>> if non-empty
    vector<vector<Vector4d>> cloud_prev_clusters;
    if (!cloud_prev.empty()) {
        cloud_prev_clusters.push_back(cloud_prev);
    }

    // Clustering loop
    int iter = 0;
    while (!Node::unexplored.empty() && C.size() < MAX_CLUSTER_NUM) {
        // Get next unexplored point
        Vector4d next_point;
        
        if (iter == 0) {
            next_point = seeds;
            C.push_back({next_point});  
            Node::unexplored.erase(make_tuple(seeds[0], seeds[1], seeds[2]));
        }
        else {
            auto it = Node::unexplored.begin();
            next_point << get<0>(*it), get<1>(*it), get<2>(*it), cloud(0,3);

            C.push_back({next_point});                 // Initialize new cluster
            Node::unexplored.erase(it);               // Remove from unexplored
        }
        
        // Search tree and update unexplored
        vector<Vector4d> remaining_unexplored = kd_tree->search_tree(kd_tree, next_point,
                                                                    radius,
                                                                    C, cloud_prev_clusters);
        Node::unexplored.clear();
        for (auto& p : remaining_unexplored)
            Node::unexplored.insert(make_tuple(p(0), p(1), p(2)));

        // Store centroid
        prev.push_back(kd_tree->prev);
        iter += 1;
    }

    // Convert clusters to Python list of NumPy arrays
    py::list py_clusters;
    for (auto& c : C) {
        MatrixXd mat(c.size(), 4);
        for (size_t i = 0; i < c.size(); ++i) mat.row(i) = c[i];
        py_clusters.append(py::cast(mat));
    }

    // Convert prevs to Python list of NumPy arrays
    py::list py_prevs;
    for (auto& v : prev) py_prevs.append(py::cast(v));

    return py::make_tuple(py_clusters, py_prevs);
}


PYBIND11_MODULE(cluster_cpp, m) {
    m.def("euclidean_cluster", &euclidean_cluster,
          py::arg("seeds"),
          py::arg("cloud"),
          py::arg("radius"),
          py::arg("MIN_CLUSTER_SIZE") = 1,
          py::arg("mode") = "cartesian",
          py::arg("cloud_prev") = std::vector<Vector4d>{},
          py::arg("reorder") = true,
          py::arg("MAX_CLUSTER_NUM") = std::numeric_limits<double>::infinity(),
          "Cluster a point cloud");
}
