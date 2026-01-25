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
#include <unordered_set>
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

        if (sqrt(query[0]*query[0] + query[1]*query[1] + query[2]*query[2]) > 5.0) radius *= 2;

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
        double dz = (p[2] - q[2])/2;
        return sqrt(dx*dx + dy*dy + dz*dz);
    }

    static tuple<double, double> lidar_constrained_distance(const PointT& p, double dhor, double dvert, double voxel_size) {
        // dhor: horizontal angle resolution of lidar in radians
        // dvert: vertical angle resolution of lidar in radians
        // voxel_size: voxel grid size used in downsampling
        // returns max dxy and dz for p's neighbors to be physically of the same object as p
        double rp = sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
        
        double dxy = rp * dhor + sqrt(2) * voxel_size;
        double dz = rp * dvert + voxel_size;

        return make_tuple(dxy, dz);
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

        const double PI = 3.14159265358979323846;
        auto [dist_xy, dist_z] = lidar_constrained_distance(query, 0.4*(PI/180), 0.5*(PI/180), 0.05);
        // Leaf node: check all points
        if (node->is_leaf()) {
            for (int idx : node->point_indices) {
                double dx = points_[idx][0] - query[0];
                double dy = points_[idx][1] - query[1];
                double dz = (points_[idx][2] - query[2]) / 2;

                if (sqrt(dx*dx + dy*dy) <= (radius + dist_xy) && dz <= 2 * (radius + dist_z)) {
                // if (std::abs(dx) < radius && std::abs(dy) < radius && std::abs(dz) < 2 * radius) {
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

        obs_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/rslidar_clusters", 10);

        obs_arr_pub_ = this->create_publisher<cev_msgs::msg::Obstacles>(
            "/rslidar_obstacles", 10);
        
        RCLCPP_INFO(this->get_logger(), "ClusterNode started - waiting for PointCloud2 on '/rslidar_points'");
    }

private:
    void listenerCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        sensor_msgs::msg::PointCloud2 obs_points;
        obs_points.header = msg->header;
        obs_points.header.stamp = msg->header.stamp;

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        std::lock_guard<std::mutex> lock(callback_mutex_);

        vector<Vector3d> cloud_init;

        size_t total_points = 0;
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z))
                continue;
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

            std::unordered_set<int> visited_indices;
            C_prev_.resize(0, 4);
            auto [C, visited_inds] = euclidean_cluster(this->data_.row(0), this->data_, visited_indices, 0.1, 10, "cartesian", false, false);

            outputClusters(C, obs_points, total_points);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            RCLCPP_INFO(this->get_logger(), "RAN FOR: %ld ms", duration.count());
        }
        ++iter_;
    }

   void outputClusters(const vector<vector<Vector4d>>& clusters, sensor_msgs::msg::PointCloud2 obs_points, int max_num_points) {
        cev_msgs::msg::Obstacles obstacles_msg;
        obstacles_msg.obstacles.reserve(clusters.size());

        obs_points.header.frame_id = "rslidar";
        obs_points.height = 1;

        sensor_msgs::PointCloud2Modifier modifier(obs_points);
        modifier.setPointCloud2Fields(
            4,
            "x", 1, sensor_msgs::msg::PointField::FLOAT32,
            "y", 1, sensor_msgs::msg::PointField::FLOAT32,
            "z", 1, sensor_msgs::msg::PointField::FLOAT32,
            "id", 1, sensor_msgs::msg::PointField::UINT32
        );
        modifier.resize(max_num_points);

        sensor_msgs::PointCloud2Iterator<float> out_x(obs_points, "x");
        sensor_msgs::PointCloud2Iterator<float> out_y(obs_points, "y");
        sensor_msgs::PointCloud2Iterator<float> out_z(obs_points, "z");
        sensor_msgs::PointCloud2Iterator<float> out_id(obs_points, "id");

        int actual_num_points = 0;

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

                *out_x = x;
                *out_y = y;
                *out_z = z;
                *out_id = id;
                
                std::memcpy(ptr, &x, sizeof(float));
                ptr += sizeof(float);
                std::memcpy(ptr, &y, sizeof(float));
                ptr += sizeof(float);
                std::memcpy(ptr, &z, sizeof(float));
                ptr += sizeof(float);
                std::memcpy(ptr, &id, sizeof(uint32_t));
                ptr += sizeof(uint32_t);

                ++out_x; ++out_y; ++out_z; ++out_id;
                ++actual_num_points;
            }
            
            obstacles_msg.obstacles.push_back(pc_msg);
        }
        
        modifier.resize(actual_num_points);
        obs_pub_->publish(obs_points);
        obs_arr_pub_->publish(obstacles_msg);
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

    std::tuple<vector<vector<Eigen::Vector4d>>, std::unordered_set<int>> grow_seed(
        const Vector4d seed, 
        const MatrixXd& data, 
        int itr
    ) {
        std::unordered_set<int> visited_indices;
        if (true) {
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

    std::tuple<vector<vector<Eigen::Vector4d>>, std::unordered_set<int>>
    euclidean_cluster(const Vector4d seeds,
                    const MatrixXd& cloud_input, 
                    std::unordered_set<int> visited_indices,
                    double radius,
                    int MIN_CLUSTER_SIZE = 1,
                    const string& mode = "cartesian",
                    bool reorder = true, 
                    bool is_parallel = false,
                    double MAX_CLUSTER_NUM = numeric_limits<double>::infinity()
                    ) {
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

        // Build points vector with mapping
        vector<Vector4d> points;
        vector<int> point_to_original_idx;
        unordered_map<int, int> original_to_point_idx;  // NEW: Reverse mapping
        
        for (int i = 0; i < cloud.rows(); ++i) {
            int original_idx = static_cast<int>(cloud(i, 3));
            if (is_parallel && taken_[original_idx].flag.load(std::memory_order_acquire)) {
                continue;
            }
            original_to_point_idx[original_idx] = points.size();  // NEW
            point_to_original_idx.push_back(original_idx);
            points.push_back(cloud.row(i).transpose());
        }

        // Build KD-tree
        KdTree<Vector4d> kd_tree(points, 3);

        // Initialize unexplored set with ORIGINAL indices
        unordered_set<int> unexplored;
        for (int points_idx = 0; points_idx < points.size(); ++points_idx) {
            unexplored.insert(point_to_original_idx[points_idx]);
        }

        vector<vector<Vector4d>> C;

        int iter = 0;
        while (!unexplored.empty() && C.size() < MAX_CLUSTER_NUM) {
            Vector4d next_point;
            int next_original_idx;
            int next_points_idx;

            if (iter == 0) {
                // Seed iteration
                next_original_idx = static_cast<int>(seeds[3]);
                
                // Check if seed exists in points vector
                if (original_to_point_idx.find(next_original_idx) == original_to_point_idx.end()) {
                    // Seed was filtered out, skip
                    iter++;
                    continue;
                }
                
                next_points_idx = original_to_point_idx[next_original_idx];
                next_point = points[next_points_idx];
            } else {
                // Pick any unexplored point
                next_original_idx = *unexplored.begin();
                unexplored.erase(next_original_idx);
                
                if (original_to_point_idx.find(next_original_idx) == original_to_point_idx.end()) {
                    continue;
                }
                
                next_points_idx = original_to_point_idx[next_original_idx];
                next_point = points[next_points_idx];
            }
            iter++;

            unexplored.erase(next_original_idx);
            if (is_parallel && taken_[next_original_idx].flag.load(std::memory_order_acquire)) {
                continue;
            }
            if (visited_indices.count(next_original_idx)) {
                continue;
            }
            
            C.push_back({next_point});

            vector<Vector4d> stack = {next_point};
            while (!stack.empty()) {
                Vector4d query = stack.back();
                stack.pop_back();
                
                int query_original_idx = static_cast<int>(query[3]);
                if (is_parallel && taken_[query_original_idx].flag.load(std::memory_order_acquire)) continue;
                if (visited_indices.count(query_original_idx)) continue;
                
                set_atomic_flag(query_original_idx);

                // Search returns indices into points[] vector
                vector<int> neighbors = kd_tree.search_radius(query, radius);
                for (int neighbor_points_idx : neighbors) {
                    const auto& p = points[neighbor_points_idx];
                    int neighbor_original_idx = point_to_original_idx[neighbor_points_idx];

                    if (is_parallel && taken_[neighbor_original_idx].flag.load(std::memory_order_acquire)) continue;
                    if (!unexplored.count(neighbor_original_idx)) continue;
                    if (visited_indices.count(neighbor_original_idx)) continue;

                    C.back().push_back(p);
                    stack.push_back(p);
                    unexplored.erase(neighbor_original_idx);
                }

                visited_indices.insert(query_original_idx);
            }
            if (C.back().size() < 10) {
                C.pop_back();
            } else {
                RCLCPP_INFO(this->get_logger(), "cluster size %d\n", C.back().size());
                Vector4d centroid = computeMean(C.back());
                C_prev_.conservativeResize(C_prev_.rows() + 1, Eigen::NoChange);
                C_prev_.row(C_prev_.rows() - 1) = centroid.transpose();
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
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obs_pub_;
    rclcpp::Publisher<cev_msgs::msg::Obstacles>::SharedPtr obs_arr_pub_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = make_shared<ClusterNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
