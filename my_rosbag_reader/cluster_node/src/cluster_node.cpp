#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <cev_msgs/msg/obstacles.hpp>
// #include <cluster_node/msg/obstacle_array.hpp>
#include <vector>
#include <set>
#include <tuple>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <open3d/Open3D.h>

using namespace Eigen;
using namespace std;

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

class ClusterNode : public rclcpp::Node {
public:
    ClusterNode()
    : Node("cluster_cpp"), iter_(0)
    {
        C_prev_ = MatrixXd::Zero(1, 4);
        data_ = MatrixXd(0, 4);

        taken_;

        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/rslidar_points", 10, 
            bind(&ClusterNode::listenerCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "ClusterNode started - waiting for PointCloud2 on '/rslidar_points'");
    }

private:

    void listenerCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        ++iter_;

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        vector<Vector3d> cloud_init;

        size_t total_points = 0;
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            cloud_init.push_back({*iter_x, *iter_y, *iter_z});
            ++total_points;
        }

        open3d::geometry::PointCloud pcd;
        pcd.points_.assign(cloud_init.begin(), cloud_init.end());

        auto downsampled = pcd.VoxelDownSample(0.05);

        const auto& pts = downsampled->points_;
        size_t n = pts.size();

        MatrixXd mat(n, 3);
        for (size_t i = 0; i < n; ++i)
            mat.row(i) = pts[i];

        MatrixXd cloud(n, 4);
        cloud.leftCols<3>() = mat;
        cloud.col(3) = mat.col(2); 

        if (cloud.size() > 0) {
            RCLCPP_INFO(this->get_logger(), "LIVE: Recieved %zu points", cloud.rows());
            data_ = cloud;
            taken_.clear();
            taken_.resize(data_.rows());

            for (auto &flag : taken_) {
                flag.store(false);
            }

            // turn this into a get seeds function
            std::shared_ptr<open3d::geometry::PointCloud> cloud_pcd(new open3d::geometry::PointCloud);
            for (int i = 0; i < cloud.rows(); ++i)
                cloud_pcd->points_.push_back(cloud.row(i).head<3>());

            open3d::geometry::KDTreeFlann kdtree(*cloud_pcd);

            std::vector<Vector4d> seeds;

            for (int i = 0; i < C_prev_.rows(); ++i) {
                std::vector<int> indices(1);
                std::vector<double> dists(1);

                Vector3d query_point = C_prev_.row(i).head<3>();
                kdtree.SearchKNN(query_point, 1, indices, dists);

                Vector4d nearest_point = cloud.row(indices[0]);
                seeds.push_back(nearest_point);
            }

            // --- Remove duplicates (like np.unique) ---
            auto hash_vec4d = [](const Vector4d& v) -> size_t {
                size_t h1 = std::hash<double>{}(v[0]);
                size_t h2 = std::hash<double>{}(v[1]);
                size_t h3 = std::hash<double>{}(v[2]);
                size_t h4 = std::hash<double>{}(v[3]);
                return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
            };

            std::unordered_set<size_t> unique_hashes;
            std::vector<Vector4d> unique_seeds;
            for (auto& s : seeds) {
                size_t h = hash_vec4d(s);
                if (unique_hashes.insert(h).second)
                    unique_seeds.push_back(s);
            }

            // ###########################
            std::vector<std::future<std::vector<std::vector<Vector4d>>>> futures;
            for (auto& seed : unique_seeds) {
                futures.push_back(std::async(std::launch::async,
                    [this, seed]() {
                        std::vector<Vector4d> C_prev_vec;
                        for (int i = 0; i < this->C_prev_.rows(); ++i)
                            C_prev_vec.push_back(this->C_prev_.row(i));

                        return this->grow_seed(seed, this->data_, this->iter_, C_prev_vec);
                    }
                ));
            }

            std::vector<std::vector<Vector4d>> C;
            for (auto& f : futures) {
                auto clusters = f.get();
                C.insert(C.end(), clusters.begin(), clusters.end());
            }

        }
    }

    vector<vector<Eigen::Vector4d>> grow_seed(
        const Vector4d seed, 
        const MatrixXd& data, 
        int itr, 
        const vector<Vector4d>& C_prev
    ) {
        if (itr == 0) {
            auto [c, prev] = euclidean_cluster(seed, data, 0.1, 10, "cartesian", C_prev, false);
            return c;
        } else {
            auto [c, prev] = euclidean_cluster(seed, data, 0.1, 10, "cartesian", C_prev, false, 1);
            return c;
        }
    }

    std::tuple<std::vector<std::vector<Eigen::Vector4d>>, std::vector<Eigen::Vector4d>> 
        euclidean_cluster(const Vector4d seeds,
                        const MatrixXd& cloud_input,
                        double radius,
                        int MIN_CLUSTER_SIZE = 1,
                        const string& mode = "cartesian",
                        const vector<Vector4d>& cloud_prev = {},
                        bool reorder = true, 
                        double MAX_CLUSTER_NUM = numeric_limits<double>::infinity()
                        ) {
        // WANT REMOVAL OF POINTS FROM CLOUD_INPUT TO PROPAGATE TO ALL THREADS
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

        return make_tuple(C, prev);
    }

    int iter_;
    MatrixXd C_prev_;
    MatrixXd data_;
    std::vector<std::atomic<bool>> taken_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = make_shared<ClusterNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
