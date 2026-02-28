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

constexpr double deg2rad(double deg) {
    return deg * M_PI / 180.0;
};

// 32 beams = rows: vertical angles: vertical angle resolution is < 0.5 deg.
static constexpr int NUM_RINGS = 32;
// horizontal angle resolution is 0.4 deg.
static constexpr double AZ_RES = deg2rad(0.4);
static constexpr int NUM_COLS = static_cast<int>(2.0 * M_PI / AZ_RES);

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

        auto start = std::chrono::high_resolution_clock::now();
        constructRangeGraph(points);
        int nclusters = rangeGraphClustering();

        auto C = extractClusters(points);
        outputClusters(C, obs_points, total_points);
        RCLCPP_INFO(this->get_logger(), "LIVE: Recieved %zu points", total_points);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        RCLCPP_INFO(this->get_logger(), "RAN FOR: %ld ms", duration.count());
    }

    void constructRangeGraph(const vector<Vector3d>& points) {
        std::fill(range_image_.begin(), range_image_.end(), RangeNode{});
        std::fill(labels_.begin(), labels_.end(), -1);

        for (size_t i = 0; i < points.size(); ++i) {
            const double x = points[i][0];
            const double y = points[i][1];
            const double z = points[i][2];

            double r = std::sqrt(x*x + y*y + z*z);
            if (r < 0.1) continue;

            double az = std::atan2(y, x);
            if (az < 0) az += 2*M_PI;
            int col = static_cast<int>(az / AZ_RES);
            if (col < 0 || col >= NUM_COLS) continue;

            double el = std::asin(z / r);

            int ring = -1;
            double md = 1e9;
            for (int k = 0; k < NUM_RINGS; ++k) {
                double d = std::abs(el - vert_angles_rad_[k]);
                if (d < md) { md = d; ring = k; }
            }
            if (ring < 0) continue;

            int idx = ring * NUM_COLS + col;

            if (!range_image_[idx].valid || r < range_image_[idx].range) {
                range_image_[idx] = {
                    static_cast<float>(r),
                    static_cast<int>(i),
                    true
                };
            }
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

    void outputClusters(const vector<vector<Vector3d>>& clusters, sensor_msgs::msg::PointCloud2 obs_points, int max_num_points) {
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
