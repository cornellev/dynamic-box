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
#include <chrono>
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
    KdTree(int leaf_size = 10) : dim_(3), leaf_size_(leaf_size) {}

    KdTree(const vector<PointT>& points, int dim = -1, int leaf_size = 10) {
        build(points, dim, leaf_size);
    }

    void build(const vector<PointT>& points, int dim = -1, int leaf_size = 10) {
        clear();
        points_ = points;
        dim_ = (dim > 0) ? dim : 3;
        leaf_size_ = leaf_size;

        vector<int> indices(points.size());
        iota(indices.begin(), indices.end(), 0);

        root_ = build_recursive(indices.data(), static_cast<int>(points.size()), 0);
    }

    void clear() {
        root_.reset();
        points_.clear();
    }

    struct Node {
        int axis = -1;
        double split_value = 0.0;
        array<unique_ptr<Node>, 2> next;
        
        vector<int> point_indices;
        
        bool is_leaf() const { return !point_indices.empty(); }
    };

    int search(const PointT& query, double* minDist = nullptr) const {
        int guess = -1;
        double _minDist = numeric_limits<double>::max();
        search_recursive(query, root_.get(), &guess, &_minDist);
        if (minDist) *minDist = _minDist;
        return guess;
    }

    vector<int> search_radius(const PointT& query, double radius) const {
        vector<int> result;
        search_radius_recursive(query, root_.get(), radius, result);
        return result;
    }

private:
    unique_ptr<Node> build_recursive(int* indices, int npoints, int depth) {
        if (npoints <= 0) return nullptr;

        auto node = make_unique<Node>();
        
        if (npoints <= leaf_size_) {
            node->point_indices.assign(indices, indices + npoints);
            return node;
        }

        int axis = depth % dim_;
        int mid = (npoints - 1) / 2;

        nth_element(indices, indices + mid, indices + npoints,
                    [&](int lhs, int rhs){ return points_[lhs][axis] < points_[rhs][axis]; });

        node->split_value = points_[indices[mid]][axis];
        node->axis = axis;
        node->next[0] = build_recursive(indices, mid, depth + 1);
        node->next[1] = build_recursive(indices + mid + 1, npoints - mid - 1, depth + 1);
        return node;
    }

    static double distance(const PointT& p, const PointT& q) {
        double dx = p[0] - q[0];
        double dy = p[1] - q[1];
        double dz = p[2] - q[2];
        return sqrt(dx*dx + dy*dy + dz*dz);
    }

    void search_recursive(const PointT& query, const Node* node, 
                         int* guess, double* minDist) const {
        if (!node) return;
        
        if (node->is_leaf()) {
            for (int idx : node->point_indices) {
                double dist = distance(query, points_[idx]);
                if (dist < *minDist) {
                    *minDist = dist;
                    *guess = idx;
                }
            }
            return;
        }
        
        int axis = node->axis;
        int dir = query[axis] < node->split_value ? 0 : 1;
        search_recursive(query, node->next[dir].get(), guess, minDist);
        
        double diff = fabs(query[axis] - node->split_value);
        if (diff < *minDist) {
            search_recursive(query, node->next[1 - dir].get(), guess, minDist);
        }
    }
    
    void search_radius_recursive(const PointT& query, const Node* node, 
                                double radius, vector<int>& result) const {
        if (!node) return;
        
        // Leaf node: check all points
        if (node->is_leaf()) {
            for (int idx : node->point_indices) {
                double dist = distance(query, points_[idx]);
                if (dist != 0 && dist <= radius) {
                    result.push_back(idx);
                }
            }
            return;
        }
        
        // Internal node
        int axis = node->axis;
        double diff = query[axis] - node->split_value;
        
        if (diff < 0) {
            search_radius_recursive(query, node->next[0].get(), radius, result);
            if (-diff <= radius) {
                search_radius_recursive(query, node->next[1].get(), radius, result);
            }
        } else {
            search_radius_recursive(query, node->next[1].get(), radius, result);
            if (diff <= radius) {
                search_radius_recursive(query, node->next[0].get(), radius, result);
            }
        }
    }

    unique_ptr<Node> root_;
    vector<PointT> points_;
    int dim_;
    int leaf_size_;
};

struct AtomicBoolWrapper {
    std::atomic<bool> flag;
    AtomicBoolWrapper() : flag(false) {}
    AtomicBoolWrapper(const AtomicBoolWrapper& other) : flag(other.flag.load()) {}
    AtomicBoolWrapper& operator=(const AtomicBoolWrapper& other) {
        flag.store(other.flag.load());
        return *this;
    }
};

class ClusterNode : public rclcpp::Node {
public:
    ClusterNode()
    : Node("cluster_cpp"), iter_(0)
    {
        C_prev_ = MatrixXd::Zero(1, 4);
        data_ = MatrixXd(0, 4);

        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/rslidar_points", 10, 
            bind(&ClusterNode::listenerCallback, this, std::placeholders::_1));

        obs_pub_ = this->create_publisher<cev_msgs::msg::Obstacles>(
            "/rslidar_obstacles", 10);

        RCLCPP_INFO(this->get_logger(), "ClusterNode started - waiting for PointCloud2 on '/rslidar_points'");
    }

private:
    void listenerCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        std::lock_guard<std::mutex> lock(callback_mutex_);

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
        for (size_t i = 0; i < n; ++i) mat.row(i) = pts[i];

        MatrixXd cloud(n, 4);
        cloud.leftCols<3>() = mat;
        
        for (size_t i = 0; i < n; ++i) cloud(i, 3) = static_cast<double>(i);

        if (cloud.size() > 0) {
            auto start = std::chrono::high_resolution_clock::now();
            RCLCPP_INFO(this->get_logger(), "LIVE: Recieved %zu points", cloud.rows());
            data_ = cloud;
            taken_.clear();
            taken_.resize(data_.rows());
            for (size_t i = 0; i < data_.rows(); ++i) {
                taken_[i].flag.store(false, std::memory_order_relaxed);
            }

            open3d::geometry::PointCloud pcd;
            for (int i = 0; i < this->data_.rows(); ++i) {
                pcd.points_.push_back(this->data_.row(i).head<3>());  // Extract x, y, z
            }
            open3d::geometry::KDTreeFlann kdtree(pcd);

            // Find closest point for each past centroid
            vector<Vector4d> seeds;
            seeds.reserve(C_prev_.rows());

            for (int i = 0; i < C_prev_.rows(); ++i) {
                Vector4d centroid_4d = C_prev_.row(i);
                
                // Check validity
                if (!std::isfinite(centroid_4d[0]) || !std::isfinite(centroid_4d[1]) || !std::isfinite(centroid_4d[2])) {
                    RCLCPP_WARN(this->get_logger(), "Skipping invalid centroid at index %d", i);
                    continue;
                }
            
                // Search for nearest neighbor
                Vector3d query_point = centroid_4d.head<3>();
                vector<int> indices(1);
                vector<double> dists(1);
                
                int found = kdtree.SearchKNN(query_point, 1, indices, dists);
                
                if (found > 0 && indices[0] >= 0 && indices[0] < this->data_.rows()) {
                    Vector4d nearest_point = this->data_.row(indices[0]);
                    seeds.push_back(nearest_point);
                } else {
                    RCLCPP_WARN(this->get_logger(), "KNN search failed for centroid %d", i);
                }
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
            vector<Vector4d> unique_seeds;
            for (auto& s : seeds) {
                size_t h = hash_vec4d(s);
                if (unique_hashes.insert(h).second)
                    unique_seeds.push_back(s);
            }

            // ###########################
            RCLCPP_INFO(this->get_logger(), "Unique Seeds: %zu", C_prev_.rows());
            vector<std::future<std::tuple<vector<vector<Vector4d>>, std::set<int>>>> futures;
            for (auto& seed : unique_seeds) {
                futures.push_back(std::async(std::launch::async,
                    [this, seed]() {
                        return this->grow_seed(seed, this->data_, this->iter_);
                    }
                ));
            }

            // threadpool, waiting for all futures to complete before continuing
            std::set<int> visited_indices;
            vector<vector<Vector4d>> C;
            for (auto& f : futures) {
                auto [clusters, visited_indices] = f.get();
                C.insert(C.end(), clusters.begin(), clusters.end());
            }

            vector<Vector4d> unclaimed_points;
            for (int i = 0; i < data_.rows(); ++i) {
                if (!visited_indices.count(i)) {
                    unclaimed_points.push_back(data_.row(i).transpose());
                }
            }
            
            // Rebuild data_ matrix with only unclaimed points
            if (!unclaimed_points.empty()) {
                MatrixXd remaining_cloud(unclaimed_points.size(), 4);
                for (size_t i = 0; i < unclaimed_points.size(); ++i) {
                    remaining_cloud.row(i) = unclaimed_points[i].transpose();
                }
                auto [clusters, visited_inds] = euclidean_cluster(remaining_cloud.row(0), remaining_cloud, visited_indices, 0.1, 10, "cartesian", false, false);
                C.insert(C.end(), clusters.begin(), clusters.end());
            }

            int MIN_CLUSTER_SIZE = 15;
            C_prev_.resize(0, 4);
            for (int i = static_cast<int>(C.size()) - 1; i >= 0; --i) {
                if (C[i].size() < MIN_CLUSTER_SIZE) {
                    C.erase(C.begin() + i);
                } 
                else {
                    Vector4d centroid = computeMean(C[i]);
                    C_prev_.conservativeResize(C_prev_.rows() + 1, Eigen::NoChange);
                    C_prev_.row(C_prev_.rows() - 1) = centroid.transpose();
                }
            }

            outputClusters(C);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            RCLCPP_INFO(this->get_logger(), "RAN FOR: %ld ms", duration.count());
        }
        ++iter_;
    }

   void outputClusters(const vector<vector<Vector4d>>& clusters) {
        cev_msgs::msg::Obstacles obstacles_msg;
        obstacles_msg.obstacles.reserve(clusters.size());

        for (size_t i = 0; i < clusters.size(); ++i) {
            const auto& cluster = clusters[i];
            
            // Create PointCloud2 message
            sensor_msgs::msg::PointCloud2 pc_msg;
            
            // Set up fields
            pc_msg.fields.resize(4);
            
            pc_msg.fields[0].name = "x";
            pc_msg.fields[0].offset = 0;
            pc_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
            pc_msg.fields[0].count = 1;
            
            pc_msg.fields[1].name = "y";
            pc_msg.fields[1].offset = 4;
            pc_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
            pc_msg.fields[1].count = 1;
            
            pc_msg.fields[2].name = "z";
            pc_msg.fields[2].offset = 8;
            pc_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
            pc_msg.fields[2].count = 1;
            
            pc_msg.fields[3].name = "id";
            pc_msg.fields[3].offset = 12;
            pc_msg.fields[3].datatype = sensor_msgs::msg::PointField::INT32;
            pc_msg.fields[3].count = 1;
            
            // Set metadata
            pc_msg.is_bigendian = false;
            pc_msg.point_step = 16;  // 3 floats (12 bytes) + 1 int (4 bytes)
            pc_msg.row_step = pc_msg.point_step * cluster.size();
            pc_msg.is_dense = true;
            pc_msg.width = cluster.size();
            pc_msg.height = 1;
            
            // Allocate data buffer
            pc_msg.data.resize(pc_msg.row_step);
            
            // Fill data buffer
            uint8_t* ptr = pc_msg.data.data();
            for (const auto& point : cluster) {
                // Pack x, y, z as float32
                float x = static_cast<float>(point[0]);
                float y = static_cast<float>(point[1]);
                float z = static_cast<float>(point[2]);
                uint32_t id = static_cast<uint32_t>(i);
                
                std::memcpy(ptr, &x, sizeof(float));
                ptr += sizeof(float);
                std::memcpy(ptr, &y, sizeof(float));
                ptr += sizeof(float);
                std::memcpy(ptr, &z, sizeof(float));
                ptr += sizeof(float);
                std::memcpy(ptr, &id, sizeof(uint32_t));
                ptr += sizeof(uint32_t);
            }
            
            obstacles_msg.obstacles.push_back(pc_msg);
        }
        
        obs_pub_->publish(obstacles_msg);
        RCLCPP_INFO(this->get_logger(), "Published %zu obstacles", obstacles_msg.obstacles.size());
    }

    Vector4d computeMean(const vector<Vector4d>& points) {
        if (points.empty()) {
            return Vector4d::Zero();
        }

        Vector4d sum = Vector4d::Zero();
        for (const auto& p : points) {
            sum.head<3>() += p.head<3>();
        }
        sum.head<3>() /= static_cast<double>(points.size());

        return sum;
    }

    std::tuple<vector<vector<Eigen::Vector4d>>, std::set<int>> grow_seed(
        const Vector4d seed, 
        const MatrixXd& data, 
        int itr
    ) {
        std::set<int> visited_indices;
        if (itr == 0) {
            return euclidean_cluster(seed, data, visited_indices, 0.1, 10, "cartesian", false, true);
        } else {
            return euclidean_cluster(seed, data, visited_indices, 0.1, 10, "cartesian", false, true, 2);
        }
    }

    bool set_atomic_flag(int idx) {
        bool expected = false;
        return taken_[idx].flag.compare_exchange_strong(expected, true,
                                                    std::memory_order_acquire,
                                                    std::memory_order_relaxed);
    }

    std::tuple<vector<vector<Eigen::Vector4d>>, std::set<int>>
        euclidean_cluster(const Vector4d seeds,
                        const MatrixXd& cloud_input, 
                        std::set<int> visited_indices,
                        double radius,
                        int MIN_CLUSTER_SIZE = 1,
                        const string& mode = "cartesian",
                        bool reorder = true, 
                        bool is_parallel = false,
                        double MAX_CLUSTER_NUM = numeric_limits<double>::infinity()
                        ) {
        // WANT REMOVAL OF POINTS FROM CLOUD_INPUT TO PROPAGATE TO ALL THREADS
        vector<vector<Eigen::Vector4d>> cheese;
        vector<Eigen::Vector4d> burgers;

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

        // only consider cloud points that are not atomic flagged
        vector<Vector4d> points;
        for (int i = 0; i < cloud.rows(); ++i) {
            int idx = static_cast<int>(cloud(i, 3));
            if (is_parallel && taken_[idx].flag.load(std::memory_order_acquire)) {
                continue;
            }
            points.push_back(cloud.row(i).transpose());
        };

        // Build KD-tree
        KdTree<Vector4d> kd_tree(points, 3);

        // Initialize unexplored set
        set<int> unexplored;
        for (const auto& p : points)
            unexplored.insert(static_cast<int>(p[3]));

        vector<vector<Vector4d>> C;

        int iter = 0;
        while (!unexplored.empty() && C.size() < MAX_CLUSTER_NUM) {
            Vector4d next_point;
            int next_idx;

            if (iter == 0) {
                next_point = seeds; // if seeds immediately is an outlier, try searching its neighbors, really need icp here
                next_idx = static_cast<int>(seeds[3]);
            } else {
                next_idx = *unexplored.begin();
                unexplored.erase(next_idx);
                next_point = points[next_idx];
            }
            iter++;

            unexplored.erase(next_idx);
            if (is_parallel && taken_[next_idx].flag.load(std::memory_order_acquire)) {
                continue;
            }
            if (visited_indices.count(next_idx)) {
                continue;
            }
            C.push_back({next_point});
            // next_point is the first point to query in new cluster, need to set atomic flag after getting more neighbors

            vector<Vector4d> stack = {next_point};
            while (!stack.empty()) {
                int sz = stack.size();
                Vector4d query = stack.back();
                stack.pop_back();
                // should be safe to set_atomic_flag here, will prevent other threads from searching this point, but allows for kd_tree search still here
                int query_idx = static_cast<int>(query[3]);
                if (is_parallel && taken_[query_idx].flag.load(std::memory_order_acquire)) continue;
                if (visited_indices.count(query_idx)) continue;
                // should be safe to set_atomic_flag here, will prevent other threads from searching this point, but allows for kd_tree search still here
                // only set atomic_flag when the point is actively being explored
                set_atomic_flag(static_cast<int>(query_idx));

                vector<int> neighbors = kd_tree.search_radius(query, radius);
                for (int idx_n : neighbors) {
                    if (idx_n == query[3]) continue;
                    const auto& p = points[idx_n];
                    int point_idx = static_cast<int>(p[3]);

                    if (is_parallel && taken_[point_idx].flag.load(std::memory_order_acquire)) continue;
                    if (!unexplored.count(point_idx)) continue;
                    if (visited_indices.count(point_idx)) continue;

                    // don't set atomic flag here, or else can't explore above
                    C.back().push_back(p);
                    stack.push_back(p);
                    unexplored.erase(point_idx);
                }

                // add to visited once new neighbors discovered
                visited_indices.insert(query_idx);
            }
        }
        return make_tuple(C, visited_indices);
    }

    int iter_;
    MatrixXd C_prev_;
    MatrixXd data_;
    // check device, if CPU, use std::atomic_flag, if GPU use CUDA's atomic
    vector<AtomicBoolWrapper> taken_;
    std::mutex callback_mutex_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Publisher<cev_msgs::msg::Obstacles>::SharedPtr obs_pub_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = make_shared<ClusterNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
