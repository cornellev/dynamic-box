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
#include <queue>
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

constexpr double deg2rad(double deg) {
    return deg * M_PI / 180.0;
};

static constexpr int NUM_RINGS = 32;
static constexpr double AZ_RES = deg2rad(0.4);
static constexpr int NUM_COLS = static_cast<int>(2.0 * M_PI / AZ_RES);

/* Helios-32 vertical angles (official pattern) */
static const std::array<double, NUM_RINGS> HELIOS32_VERT_DEG = {
    -25,-23,-21,-19,-17,-15,-13,-11,
     -9, -7, -5, -3, -1,  1,  3,  5,
      7,  9, 11, 13, 15,-24,-22,-20,
    -18,-16,-14,-12,-10, -8, -6, -4
};

struct RangeNode {
    float range = 0.f;
    int point_idx = -1;
    bool valid = false;
};

class TwoLayerNode : public rclcpp::Node {
public:
    TwoLayerNode()
    : Node("two_layer_cpp"), iter_(0)
    {
        for (int i = 0; i < NUM_RINGS; ++i)
            vert_angles_rad_[i] = deg2rad(HELIOS32_VERT_DEG[i]);

        range_image_.resize(NUM_RINGS * NUM_COLS);
        labels_.resize(NUM_RINGS * NUM_COLS);

        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
        C_prev_ = MatrixXd::Zero(1, 4);
        data_ = MatrixXd(0, 4);

        // lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        //     "/rslidar_points", 10, 
        //     bind(&TwoLayerNode::listenerCallback, this, std::placeholders::_1));

        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/sensing/lidar/top/rectified/pointcloud", qos, 
            bind(&TwoLayerNode::listenerCallback, this, std::placeholders::_1));

        obs_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/rslidar_clusters", 10);

        obs_arr_pub_ = this->create_publisher<cev_msgs::msg::Obstacles>(
            "/rslidar_obstacles", 10);
        
        RCLCPP_INFO(this->get_logger(), "TwoLayerNode started - waiting for PointCloud2 on '/rslidar_points'");
    }

// define the ground plane given PCA of all points cut off z < height of the car:
// axis align bounding box to a ground plane
private:
    void listenerCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        sensor_msgs::msg::PointCloud2 obs_points;
        obs_points.header = msg->header;
        obs_points.header.stamp = msg->header.stamp;

        vector<Vector3d> points;
        points.reserve(msg->width * msg->height);

        sensor_msgs::PointCloud2ConstIterator<float> ix(*msg,"x"), iy(*msg,"y"), iz(*msg,"z");

        size_t total_points = 0;
        for (; ix != ix.end(); ++ix, ++iy, ++iz) {
            if (!std::isfinite(*ix) || !std::isfinite(*iy) || !std::isfinite(*iz))
                continue;
            if (*iz < -1.2 || std::abs(*ix) > 20 || std::abs(*iy) > 20)
                continue;
            points.emplace_back(*ix,*iy,*iz);
            ++total_points;
        }

        if (points.empty()) return;

        constructRangeGraph(msg);
        int nclusters = rangeGraphClustering();

        auto C = extractClusters(points);
        outputClusters(C, msg->header);
    }

    void constructRangeGraph(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        std::fill(range_image_.begin(), range_image_.end(), RangeNode{});
        std::fill(labels_.begin(), labels_.end(), -1);

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        vector<Vector3d> cloud_init;

        size_t i = 0;
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            if (*iter_z < -1.2 || *iter_y > 20 || *iter_y < -20 || *iter_x > 20 || *iter_x < -20 || !std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z))
                continue;

            double x = *iter_x, y = *iter_y, z = *iter_z;
            double r = std::sqrt(x*x + y*y + z*z);
            if (r < 0.1) continue;

            double az = std::atan2(y,x);
            if (az < 0) az += 2*M_PI;
            int col = static_cast<int>(az / AZ_RES);
            if (col < 0 || col >= NUM_COLS) continue;

            double el = std::asin(z / r);
            int ring = -1;
            double md = 1e9;

            for (int k=0;k<NUM_RINGS;k++) {
                double d = std::abs(el - vert_angles_rad_[k]);
                if (d < md) { md = d; ring = k; }
            }

            int idx = ring * NUM_COLS + col;
            if (!range_image_[idx].valid || r < range_image_[idx].range)
                range_image_[idx] = {static_cast<float>(r),(int)i,true};

            i++;
        }
    }

    inline void neighbors(int r,int c,vector<int>& out) {
        out.clear();
        int l=(c-1+NUM_COLS)%NUM_COLS, rr=(c+1)%NUM_COLS;
        out.push_back(r*NUM_COLS+l);
        out.push_back(r*NUM_COLS+rr);
        if (r>0) out.push_back((r-1)*NUM_COLS+c);
        if (r<NUM_RINGS-1) out.push_back((r+1)*NUM_COLS+c);
    }

    int rangeGraphClustering() {
        int cid = 0;
        std::queue<int> q;
        vector<int> nbrs;

        for (int r=0;r<NUM_RINGS;r++) {
            for (int c=0;c<NUM_COLS;c++) {
                int idx=r*NUM_COLS+c;
                if (!range_image_[idx].valid || labels_[idx]!=-1) continue;

                labels_[idx]=cid;
                q.push(idx);

                while(!q.empty()) {
                    int u=q.front(); q.pop();
                    int ur=u/NUM_COLS, uc=u%NUM_COLS;
                    neighbors(ur,uc,nbrs);

                    for(int v:nbrs) {
                        if(!range_image_[v].valid || labels_[v]!=-1) continue;
                        if(std::abs(range_image_[u].range-range_image_[v].range)<0.6f) {
                            labels_[v]=cid;
                            q.push(v);
                        }
                    }
                }
                cid++;
            }
        }
        return cid;
    }

    vector<vector<Vector3d>> extractClusters(const vector<Vector3d>& pts) {
        std::unordered_map<int,vector<Vector3d>> mp;

        for (size_t i=0;i<range_image_.size();i++) {
            if (!range_image_[i].valid) continue;
            int c=labels_[i];
            mp[c].push_back(pts[range_image_[i].point_idx]);
        }

        vector<vector<Vector3d>> out;
        for(auto& kv:mp)
            if(kv.second.size()>=10)
                out.push_back(std::move(kv.second));
        return out;
    }


    void publishClusters(const vector<vector<Vector3d>>& clusters,
                     const std_msgs::msg::Header& header) {

        cev_msgs::msg::Obstacles obs;

        for (size_t i=0;i<clusters.size();i++) {
            sensor_msgs::msg::PointCloud2 pc;
            pc.header=header;
            pc.height=1;
            pc.width=clusters[i].size();
            pc.is_dense=true;
            pc.is_bigendian=false;
            pc.point_step=16;
            pc.row_step=pc.width*pc.point_step;

            pc.fields.resize(4);
            pc.fields[0]={"x",0,sensor_msgs::msg::PointField::FLOAT32,1};
            pc.fields[1]={"y",4,sensor_msgs::msg::PointField::FLOAT32,1};
            pc.fields[2]={"z",8,sensor_msgs::msg::PointField::FLOAT32,1};
            pc.fields[3]={"id",12,sensor_msgs::msg::PointField::UINT32,1};

            pc.data.resize(pc.row_step);
            uint8_t* ptr=pc.data.data();

            for(auto& p:clusters[i]) {
                float x=p(0),y=p(1),z=p(2);
                uint32_t id=i;
                memcpy(ptr,&x,4); ptr+=4;
                memcpy(ptr,&y,4); ptr+=4;
                memcpy(ptr,&z,4); ptr+=4;
                memcpy(ptr,&id,4); ptr+=4;
            }
            obs.obstacles.push_back(pc);
        }

        pub_obs_->publish(obs);
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
            return kd_euclidean_cluster(seed, data, visited_indices, 0.1, 10, "cartesian", false, true);
        } else {
            return kd_euclidean_cluster(seed, data, visited_indices, 0.1, 10, "cartesian", false, true, 2);
        }
    }

    std::tuple<vector<vector<Eigen::Vector4d>>, std::unordered_set<int>>
    // add something here that will, for each cluster, keep track of the z_min of that cluster, such that when we have finished
    // growing this cluster, we can determine whether this obstacle will actually even be vertically in range of our car
    // and we can remove a cluster from the list of cluster if z_min >> range of the car.
    kd_euclidean_cluster(const Vector4d seeds,
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
        
        for (int i = 0; i < cloud.rows(); ++i) {
            points.push_back(cloud.row(i).transpose());
        }

        // Build KD-tree
        KdTree<Vector4d> kd_tree(points, 3);

        // Initialize unexplored set with ORIGINAL indices
        unordered_set<int> unexplored;
        for (int points_idx = 0; points_idx < points.size(); ++points_idx) {
            unexplored.insert(points_idx);
        }

        vector<vector<Vector4d>> C;

        int iter = 0;
        while (!unexplored.empty() && C.size() < MAX_CLUSTER_NUM) {
            Vector4d next_point;
            int next_idx;

            if (iter == 0) {
                // Seed iteration
                next_point = points[static_cast<int>(seeds[3])];
            } else {
                // Pick any unexplored point
                next_idx = *unexplored.begin();
                unexplored.erase(next_idx);
                next_point = points[next_idx];
            }
            iter++;

            unexplored.erase(next_idx);

            if (visited_indices.count(next_idx)) {
                continue;
            }
            
            C.push_back({next_point});

            vector<Vector4d> stack = {next_point};
            while (!stack.empty()) {
                Vector4d query = stack.back();
                stack.pop_back();
                
                int query_idx = static_cast<int>(query[3]);
                if (visited_indices.count(query_idx)) continue;

                // Search returns indices into points[] vector
                vector<int> neighbors = kd_tree.search_radius(query, radius);
                for (int neighbor_idx : neighbors) {
                    const auto& p = points[neighbor_idx];

                    if (!unexplored.count(neighbor_idx)) continue;
                    if (visited_indices.count(neighbor_idx)) continue;

                    C.back().push_back(p);
                    stack.push_back(p);
                    unexplored.erase(neighbor_idx);
                }

                visited_indices.insert(query_idx);
            }
            if (C.back().size() < 10) {
                C.pop_back();
            } else {
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
    std::array<double,NUM_RINGS> vert_angles_rad_;
    vector<RangeNode> range_image_;
    vector<int> labels_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obs_pub_;
    rclcpp::Publisher<cev_msgs::msg::Obstacles>::SharedPtr obs_arr_pub_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = make_shared<TwoLayerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
