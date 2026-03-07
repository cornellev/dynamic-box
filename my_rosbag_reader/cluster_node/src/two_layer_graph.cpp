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
    float x         = 0.f;
    float y         = 0.f;
    float z         = 0.f;
    float range     = 0.f;  // d
    int point_idx   = -1;
    bool valid      = false;
    int alpha       = -1;
    int beta        = -1;
};

struct GraphNode {
    float range     = 0.f; 
    float x_mean    = 0.f;
    float y_mean    = 0.f;
    int   alpha     = -1;
    int   cluster   = -1;
    int   start_pos = -1;
    int   end_pos   = -1;
    // indices into G_r that belong to this node
    vector<int> members;
};


class TwoLayerNode : public rclcpp::Node {
public:
    TwoLayerNode()
    : Node("two_layer_cpp"), iter_(0)
    {
        for (int i = 0; i < NUM_RINGS; ++i)
            vert_angles_rad_[i] = deg2rad(HELIOS32_VERT_DEG[i]);

        G_r.resize(NUM_RINGS * NUM_COLS);
        cluster_ids_.resize(NUM_RINGS * NUM_COLS);

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

        // constructing the range graph
        constructRangeGraph(points);

        // 1st segmentation
        firstSegmentation(Thd_, Thz_);

        // build set graph G_c
        buildSetGraph();

        // 2nd segmentation
        int nclusters = secondSegmentation(Thd_);

        auto C = extractClusters(points);
        outputClusters(C, obs_points, total_points);
        
        RCLCPP_INFO(this->get_logger(), "LIVE: Recieved %zu points", total_points);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        RCLCPP_INFO(this->get_logger(), "RAN FOR: %ld ms", duration.count());
    }

    // discretize point cloud into bins determined by horizontal angles (basically rows of lidar rays)
    // and vertical angles
    void constructRangeGraph(const vector<Vector3d>& points) {
        // each cell of G_r stores range, point index, valid flag
        std::fill(G_r.begin(), G_r.end(), RangeNode{});
        std::fill(cluster_ids_.begin(), cluster_ids_.end(), -1);

        for (size_t i = 0; i < points.size(); ++i) {
            // iterate over each point, then compute the angle of this point to the origin to determine channel
            const double x = points[i][0];
            const double y = points[i][1];
            const double z = points[i][2];

            double r = std::sqrt(x*x + y*y + z*z);
            if (r < 0.1) continue;

            double az = std::atan2(y, x);
            if (az < 0) az += 2*M_PI;
            // AZ_RES is horizontal angle resolution, each index of range_graph corresponsds to 
            // a horizontal angle det by az / AZ_RES
            int col = static_cast<int>(az / AZ_RES);
            if (col < 0 || col >= NUM_COLS) continue;

            double el = std::asin(z / r);
            int ring = -1;
            double md = 1e9;
            // determine the closest vertical angle
            for (int k = 0; k < NUM_RINGS; ++k) {
                double d = std::abs(el - vert_angles_rad_[k]);
                if (d < md) { md = d; ring = k; }
            }
            if (ring < 0) continue;

            // index into range graph determined by vertical and horizontal angle
            int idx = ring * NUM_COLS + col;

            if (!G_r[idx].valid || r < G_r[idx].range) {
                G_r[idx] = {
                    static_cast<float>(x),
                    static_cast<float>(y),
                    static_cast<float>(z),
                    static_cast<float>(r),
                    static_cast<int>(i),
                    true, -1, -1
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

    void firstSegmentation(float Th_d, float Th_z)
    {
        for (int i = 0; i < NUM_RINGS; ++i) // for i = 0; i < row; i++ do
        {
            int L_cnt   = 0;                // L_cnt = 0
            int pre_pos = -1;               

            for (int j = 0; j < NUM_COLS; ++j) // for j = 1; j < col; j++ do
            {
                int idx = i * NUM_COLS + j; 
                if (!G_r[idx].valid) continue; // if G_r(i,j) -> valid = F (False) then continue

                if (L_cnt == 0) {              // if L_cnt == 0 then
                    L_cnt++;                   // L_cnt++;
                    pre_pos = j;               // Start Pos = j; Pre Pose = j;
                    G_r[idx].alpha = L_cnt;    // G_r(i,j) -> alpha = L_cnt
                } else {                       // else
                    int pre_idx = i * NUM_COLS + pre_pos;

                    float dist = pointDist(idx, pre_idx);  // Dis(G_r(i,j), G_r(i,Pre Pos))
                    auto vecs = getVecs(idx, pre_idx, i);  // init V1, V2
                    if (!vecs) {
                        pre_pos = j;
                        G_r[idx].alpha = L_cnt;
                        continue;
                    }
                    auto [V1, V2] = *vecs;
                    float angle = std::acos( V1.dot(V2) / (V1.norm() * V2.norm()) ); // Ang(G_r(i,j), G_r(i, Pre Pos))
                    Eigen::Vector2f V_anglebisector = (V1 + V2) / (V1 + V2).norm();
                    Eigen::Vector2f V_mo(-(G_r[idx].x + G_r[pre_idx].x)/2.f, -(G_r[idx].y + G_r[pre_idx].y)/2.f);
        
                    // RCLCPP_INFO(this->get_logger(), "dist %f < Th_d %f, angle %f < Th_z %f", dist, Th_d, angle, Th_z);
                    // if Dis(G_r(i,j), G_r(i, Pre Pos)) > Th_d || (Ang(G_r(i,j), G_r(i, Pre Pos)) < Th_z & V_angle-bisectorV_m-o > 0) then
                    if (dist > Th_d || (angle < Th_z && V_mo.dot(V_anglebisector) > 0))
                    {
                        L_cnt++;                // L_cnt++;
                        pre_pos = j;            // Start Pos = j; Pre Pos = j
                        G_r[idx].alpha = L_cnt; // G_r(i,j) -> alpha = L_cnt
                    } else {
                        pre_pos = j;            // Pre Pos = j;
                        G_r[idx].alpha = L_cnt; // G_r(i,j) -> alpha = L_cnt
                    }
                }
            }
        }
    }

    void buildSetGraph()
    {
        G_c.assign(NUM_RINGS, {});

        for (int i = 0; i < NUM_RINGS; ++i)
        {
            std::unordered_map<int, GraphNode> seg_map;

            for (int j = 0; j < NUM_COLS; ++j)
            {
                int idx = i * NUM_COLS + j;
                if (!G_r[idx].valid) continue;

                int a = G_r[idx].alpha;
                auto& node = seg_map[a];
                node.alpha = a;
                node.members.push_back(idx);
                node.range += G_r[idx].range;
                node.x_mean += G_r[idx].x;
                node.y_mean += G_r[idx].y;

                if (node.start_pos == -1 || j < node.start_pos) node.start_pos = j;
                if (node.end_pos  == -1 || j > node.end_pos)   node.end_pos   = j;
            }

            for (auto& kv : seg_map)
            {
                float n = static_cast<float>(kv.second.members.size());
                kv.second.x_mean /= n;
                kv.second.y_mean /= n;
                kv.second.range /= n;
                G_c[i].push_back(std::move(kv.second));
            }

            // sort by alpha so that segment ordering is deterministic
            std::sort(G_c[i].begin(), G_c[i].end(),
    [](const GraphNode& a, const GraphNode& b){ return a.start_pos < b.start_pos; });
        }
    }

    int secondSegmentation(float Th_d)
    {
        for (int i = 0; i < NUM_RINGS; ++i) {   // Init(V_m), visit map
            for (auto& n : G_c[i]) {
                n.cluster = -1;
            }
        }
        int L_cnt = 0;                          // L_cnt = 0
        std::queue<std::pair<int,int>> Q;       // (ring, node-index)

        for (int i = 0; i < NUM_RINGS; ++i) {   // for i = 0; i < row; i++ do
            for (int j = 0; j < static_cast<int>(G_c[i].size()); ++j)  // for j = 1; j < Node(i) -> size; j++ do
            {
                GraphNode& cur = G_c[i][j];

                if (cur.cluster != -1) { continue; }        // if v_m(i, j) = T (True, != -1) then 
                L_cnt++;                                    // L_cnt++; 
                Q.push({i, j});                             // Q -> push(Node(i,j))
                cur.cluster = L_cnt;                        // V_m(i,j) = T (True, != -1)

                while (!Q.empty()) {                        // while Q -> empty = F (False) do
                    auto [ri, ci] = Q.front(); Q.pop();     // ri = front    
                    GraphNode& N_deal = G_c[ri][ci];        // N_deal = Q.front

                    for (auto& [rn, cn] : getNeighbors(ri, ci)) {   // for N_link in neighbor of N_deal do
                        GraphNode& N_link = G_c[rn][cn];            // N_link

                        if (N_link.cluster != -1) { continue; }     // if V_m(N_link) = T (True, != -1) then

                        if (nodeDist(N_deal, N_link) < Th_d * 3.0f) {   // if Dis(N_deal, N_link) < Th_d then
                            N_link.cluster = L_cnt;             
                            Q.push({rn, cn});                       // Q -> push(N_link)
                        }
                    }
                }
            }
        }

        // Propagate cluster IDs back to G_r cells so that
        // extractClusters() can group raw 3-D points.
        for (int i = 0; i < NUM_RINGS; ++i)
            for (auto& node : G_c[i])
                for (int idx : node.members)
                    cluster_ids_[idx] = node.cluster;

        return L_cnt;
    }

    vector<pair<int,int>> getNeighbors(int ri, int ci) const
    {
        // search the nodes near the current node (the nodes in upper and lower adjacent channels, 
        // which are connected to the current node and the first unconnected node on both sides
        vector<pair<int,int>> nbrs;
        const GraphNode& cur = G_c[ri][ci];

        for (int dr : {-1, 1})
        {
            int rn = ri + dr;
            if (rn < 0 || rn >= NUM_RINGS) continue;

            for (int cn = 0; cn < static_cast<int>(G_c[rn].size()); ++cn)
            {
                const GraphNode& cand = G_c[rn][cn];

                // horizontal: column spans must overlap (cylindrical adjacency)
                if (cand.end_pos   < cur.start_pos) continue;
                if (cand.start_pos > cur.end_pos)   break;    // sorted by start_pos, can early exit

                // vertical: XY plane distance must be within Th_Dis_ (equation 8)
                float dx = cand.x_mean - cur.x_mean;
                float dy = cand.y_mean - cur.y_mean;
                float xy_dist = std::sqrt(dx*dx + dy*dy);
                if (xy_dist < Thd_ * 3.0f)
                    nbrs.push_back({rn, cn});
            }
        }
        return nbrs;
    }

    // ==============================================================
    // Distance between two graph nodes (mean-range difference)
    // ==============================================================
    float nodeDist(const GraphNode& a, const GraphNode& b) const
    {
        float dx = a.x_mean - b.x_mean;
        float dy = a.y_mean - b.y_mean;
        return std::sqrt(dx*dx + dy*dy);
    }

    // ==============================================================
    // Euclidean distance between two range-image cells
    // ==============================================================
    float pointDist(int idx_a, int idx_b) const
    {
        float dx = G_r[idx_a].x - G_r[idx_b].x;
        float dy = G_r[idx_a].y - G_r[idx_b].y;
        return std::sqrt(dx*dx + dy*dy);
    }

    std::optional<std::array<Eigen::Vector2f, 2>> getVecs(int idx_cur, int idx_pre, int ring) const
    {
        // in one horizontal ring, this is represented by one row in G_r: after seg, we get
        // |** * ***|        |*** * ****|, 2 different pcs
        //       idx_pre  idx_cur
        // idx_pre is the last cell of the previous segment (Pre Pos in Alg 1)
        // idx_cur is the first cell of the current segment

        int col_pre = idx_pre % NUM_COLS;                           // furthest in V1
        int col_cur = idx_cur % NUM_COLS;                           // furthest in V2

        int col_pre_inner = (col_pre - 1 + NUM_COLS) % NUM_COLS;    // 2nd furthest in V1
        int idx_pre_inner = ring * NUM_COLS + col_pre_inner;
        int col_cur_inner = (col_cur + 1) % NUM_COLS;               // 2nd furthest in V2
        int idx_cur_inner = ring * NUM_COLS + col_cur_inner;

        // Need inner cells to be valid for a meaningful vector
        if (!G_r[idx_pre_inner].valid || !G_r[idx_cur_inner].valid)
            return std::nullopt;

        // V2: direction along previous segment, pointing away from boundary
        Eigen::Vector2f p_pre(G_r[idx_pre].x,
                            G_r[idx_pre].y);
        Eigen::Vector2f p_pre_in(G_r[idx_pre_inner].x,
                                G_r[idx_pre_inner].y);
        Eigen::Vector2f V2 = p_pre_in - p_pre;  // boundary → interior of seg_pre

        // V1: direction along current segment, pointing away from boundary
        Eigen::Vector2f p_cur(G_r[idx_cur].x,
                            G_r[idx_cur].y);
        Eigen::Vector2f p_cur_in(G_r[idx_cur_inner].x,
                                G_r[idx_cur_inner].y);
        Eigen::Vector2f V1 = p_cur_in - p_cur;  // boundary → interior of seg_cur

        return std::array<Eigen::Vector2f, 2>{V1, V2};
    }

    vector<vector<Vector3d>> extractClusters(const vector<Vector3d>& pts) {
        std::unordered_map<int,vector<Vector3d>> mp;

        for (size_t i=0;i<G_r.size();i++) {
            if (!G_r[i].valid) continue;
            int c=cluster_ids_[i];
            // int c = G_r[i].alpha;
            mp[c].push_back(pts[G_r[i].point_idx]);
        }

        vector<vector<Vector3d>> out;
        out.reserve(mp.size());
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
    float Thd_ = 0.5f;
    float Thz_ = static_cast<float>(deg2rad(1.0));

    MatrixXd C_prev_;
    MatrixXd data_;
    std::array<double,NUM_RINGS> vert_angles_rad_;
    vector<RangeNode> G_r;
    vector<vector<GraphNode>> G_c;
    vector<int> cluster_ids_;
    vector<int> channel_ids_;
    vector<bool> visited_; 

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
