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

// version of two_layer_graph that is supposed to be accurate but is extremely slow

using namespace Eigen;
using namespace std;

constexpr double deg2rad(double deg) {
    return deg * M_PI / 180.0;
};

// 32 beams = rows: vertical angles: vertical angle resolution is < 0.5 deg.
static constexpr int NUM_RINGS = 32;
// horizontal angle resolution is 0.4 deg.
static constexpr double AZ_RES = deg2rad(0.4);
static constexpr int NUM_COLS = 3600;

// static const std::array<double, NUM_RINGS> HELIOS32_VERT_DEG = {
//     -25,-23,-21,-19,-17,-15,-13,-11,
//      -9, -7, -5, -3, -1,  1,  3,  5,
//       7,  9, 11, 13, 15,-24,-22,-20,
//     -18,-16,-14,-12,-10, -8, -6, -4
// };

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

    float x_min =  std::numeric_limits<float>::max();
    float x_max = -std::numeric_limits<float>::max();
    float y_min =  std::numeric_limits<float>::max();
    float y_max = -std::numeric_limits<float>::max();
    // indices into G_r that belong to this node
    vector<int> members;
};


class TwoLayerNode : public rclcpp::Node {
public:
    TwoLayerNode()
    : Node("two_layer_cpp"), iter_(0)
    {
        // for (int i = 0; i < NUM_RINGS; ++i)
        //     vert_angles_rad_[i] = deg2rad(HELIOS32_VERT_DEG[i]);

        G_r.resize(NUM_RINGS * NUM_COLS);
        cluster_ids_.resize(NUM_RINGS * NUM_COLS);

        RCLCPP_INFO(this->get_logger(), "%d", NUM_COLS);

        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
        C_prev_ = MatrixXd::Zero(1, 4);
        data_ = MatrixXd(0, 4);

        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/rslidar_points", 10, 
            bind(&TwoLayerNode::listenerCallback, this, std::placeholders::_1));

        // lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        //     "/sensing/lidar/top/rectified/pointcloud", qos, 
        //     bind(&TwoLayerNode::listenerCallback, this, std::placeholders::_1));

        bev_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/bev_obstacles", 10);
    
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
        if (!angles_calibrated_) {
            calibrateFromOrganizedCloud(msg);
            return;
        }

        sensor_msgs::msg::PointCloud2 obs_points;
        obs_points.header = msg->header;

        auto start = std::chrono::high_resolution_clock::now();

        constructRangeGraph(msg);          // takes msg directly
        firstSegmentation(Thd_, Thz_);
        buildSetGraph();
        secondSegmentation(Thd_);

        auto C = extractClusters();        // reads from G_r directly
        outputClusters(C, obs_points, msg->width * msg->height);

        RCLCPP_INFO(this->get_logger(), "RAN FOR: %ld ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start).count());
    }

    void calibrateFromOrganizedCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        const uint32_t num_cols  = msg->height;
        const uint32_t num_rings = msg->width;

        std::vector<std::vector<double>> elevations(num_rings);
        for (auto& v : elevations) v.reserve(num_cols);

        sensor_msgs::PointCloud2ConstIterator<float> ix(*msg,"x"), iy(*msg,"y"), iz(*msg,"z");
        for (uint32_t row = 0; row < num_cols; ++row) {
            for (uint32_t ring = 0; ring < num_rings; ++ring, ++ix, ++iy, ++iz) {
                float x = *ix, y = *iy, z = *iz;
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
                float r = std::sqrt(x*x + y*y + z*z);
                if (r < 0.1f) continue;
                double el = std::asin(std::clamp((double)z / r, -1.0, 1.0)) * 180.0 / M_PI;
                elevations[ring].push_back(el);
            }
        }

        // Compute median elevation per ring
        std::vector<std::pair<double, int>> angle_and_orig_ring(num_rings);
        for (uint32_t ring = 0; ring < num_rings; ++ring) {
            auto& v = elevations[ring];
            if (v.empty()) {
                angle_and_orig_ring[ring] = {0.0, (int)ring};
                continue;
            }
            std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
            angle_and_orig_ring[ring] = {v[v.size()/2], (int)ring};
        }

        // Sort by elevation angle (low → high)
        std::vector<std::pair<double, int>> sorted_rings = angle_and_orig_ring;
        std::sort(sorted_rings.begin(), sorted_rings.end(),
            [](const auto& a, const auto& b){ return a.first < b.first; });

        // sorted_ring_order_[sorted_idx] = original_ring_idx
        // ring_remap_[original_ring_idx] = sorted_idx
        sorted_ring_order_.resize(num_rings);
        ring_remap_.resize(num_rings);
        for (int sorted_idx = 0; sorted_idx < (int)num_rings; ++sorted_idx) {
            int orig = sorted_rings[sorted_idx].second;
            sorted_ring_order_[sorted_idx] = orig;
            ring_remap_[orig] = sorted_idx;
            vert_angles_rad_[sorted_idx] = sorted_rings[sorted_idx].first * M_PI / 180.0;
            RCLCPP_INFO(this->get_logger(), "  sorted ring %2d: %.4f deg (orig ring %d)",
                sorted_idx, sorted_rings[sorted_idx].first, orig);
        }

        angles_calibrated_ = true;
    }

    // discretize point cloud into bins determined by horizontal angles (basically rows of lidar rays)
    // and vertical angles
    void constructRangeGraph(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        std::fill(G_r.begin(), G_r.end(), RangeNode{});
        std::fill(cluster_ids_.begin(), cluster_ids_.end(), -1);

        const uint32_t num_cols  = msg->height;  // 3600
        const uint32_t num_rings = msg->width;   // 32

        sensor_msgs::PointCloud2ConstIterator<float> ix(*msg,"x"), iy(*msg,"y"), iz(*msg,"z");

        for (uint32_t col = 0; col < num_cols; ++col) {
            for (uint32_t orig_ring = 0; orig_ring < num_rings; ++orig_ring, ++ix, ++iy, ++iz) {
                float x = *ix, y = *iy, z = *iz;
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
                float r = std::sqrt(x*x + y*y + z*z);
                if (r < 0.1f || z < -1.2f || std::abs(x) > 20.f || std::abs(y) > 20.f) continue;

                // Remap from hardware ring order to sorted elevation order
                int ring = ring_remap_[orig_ring];
                int idx  = ring * NUM_COLS + col;

                if (!G_r[idx].valid || r < G_r[idx].range)
                    G_r[idx] = { x, y, z, r, (int)(col * num_rings + orig_ring), true, -1, -1 };
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
         int L_cnt   = 0;                // L_cnt = 0
        for (int i = 0; i < NUM_RINGS; ++i) // for i = 0; i < row; i++ do
        {
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
                    // if (dist > Th_d || (angle < Th_z && V_mo.dot(V_anglebisector) > 0))
                    if (dist > 0.05f + dist * deg2rad(0.4))
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

                node.x_min = std::min(node.x_min, G_r[idx].x);
                node.x_max = std::max(node.x_max, G_r[idx].x);
                node.y_min = std::min(node.y_min, G_r[idx].y);
                node.y_max = std::max(node.y_max, G_r[idx].y);

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

                    // for (auto& [rn, cn] : getNeighbors(ri, ci)) {   // for N_link in neighbor of N_deal do
                    //     GraphNode& N_link = G_c[rn][cn];            // N_link

                    //     if (N_link.cluster != -1) { continue; }     // if V_m(N_link) = T (True, != -1) then

                    //     float dist = nodeDist(N_deal, N_link);

                    //     if (dist < 0.1f + dist * deg2rad(0.4)) {   // if Dis(N_deal, N_link) < Th_d then
                    //         N_link.cluster = L_cnt;             
                    //         Q.push({rn, cn});                       // Q -> push(N_link)
                    //     }
                    // }
                    for (auto& [rn, cn] : getNeighbors(ri, ci)) {
                        GraphNode& N_link = G_c[rn][cn];
                        if (N_link.cluster != -1) continue;

                        float boundary_dist = std::abs(nodeDist(N_deal, N_link, rn - ri));

                        float dx = std::max(N_deal.x_min - N_link.x_max, N_link.x_min - N_deal.x_max);

                        float dy;
                        if ((rn - ri) == 1) {
                            // b is upper neighbor: gap from a's top to b's bottom
                            dy = N_link.y_min - N_deal.y_max;
                        } else if ((rn - ri) == -1) {
                            // b is lower neighbor: gap from b's top to a's bottom
                            dy = N_deal.y_min - N_link.y_max;
                        } else {
                            // no direction specified: symmetric min gap
                            dy = std::max(N_deal.y_min - N_link.y_max, N_link.y_min - N_deal.y_max);
                        }

                        // Threshold: adaptive based on range only, not on boundary_dist itself
                        float rp = (N_deal.range + N_link.range) / 2.0f;
                        float Th_d = static_cast<float>(0.5f + boundary_dist * deg2rad(0.5));
                        RCLCPP_INFO(this->get_logger(), "[%d, %d] -> [%d, %d], dist %f, Thd %f", ri, ci, rn, cn, sqrt(dx*dx), 2 * (0.1f + N_link.range * deg2rad(0.4)));

                        if (sqrt(dx*dx) <= (0.3f + N_link.range * deg2rad(0.4)) && std::abs(dy / 2.f) <= 2 * (1.5f + N_link.range * deg2rad(0.5))) {
                        // if (boundary_dist < 2 * Th_d) {
                            N_link.cluster = L_cnt;
                            Q.push({rn, cn});
                        }
                    }
                }
            }
        }

        // Propagate cluster IDs back to G_r cells so that
        // extractClusters() can group raw 3-D points.
        for (int i = 0; i < NUM_RINGS; ++i) {
            for (auto& node : G_c[i]) {
                for (int idx : node.members) {
                    cluster_ids_[idx] = node.cluster;
                    G_r[idx].alpha = node.cluster;
                }
            }
        }


        return L_cnt;
    }

    // vector<pair<int,int>> getNeighbors(int ri, int ci) const
    // {
    //     vector<pair<int,int>> nbrs;
    //     const GraphNode& cur = G_c[ri][ci];

    //     for (int dr : {-1, 1})
    //     {
    //         int rn = ri + dr;
    //         if (rn < 0 || rn >= NUM_RINGS) continue;

    //         bool found_right_gap = false;

    //         for (int cn = 0; cn < (int)G_c[rn].size(); ++cn)
    //         {
    //             const GraphNode& cand = G_c[rn][cn];

    //             // Cylindrical overlap: gap in either direction around the cylinder
    //             int gap_fwd = (cand.start_pos - cur.end_pos   + NUM_COLS) % NUM_COLS;
    //             int gap_bwd = (cur.start_pos  - cand.end_pos  + NUM_COLS) % NUM_COLS;
    //             bool overlaps = (gap_fwd == 0 || gap_bwd == 0 ||
    //                             (cand.start_pos <= cur.end_pos && cand.end_pos >= cur.start_pos));

    //             float dx = cand.x_mean - cur.x_mean;
    //             float dy = cand.y_mean - cur.y_mean;
    //             float dist = std::sqrt(dx*dx + dy*dy);
    //             if (overlaps) {
    //                 if (dist < 2 * (0.1f + dist * deg2rad(0.5)))
    //                     nbrs.push_back({rn, cn});
    //             } else if (!found_right_gap && cand.start_pos > cur.end_pos) {
    //                 // first non-overlapping segment to the right — paper says check it
    //                 if (dist < 2 * (0.1f + dist * deg2rad(0.5)))
    //                     nbrs.push_back({rn, cn});
    //                 found_right_gap = true;
    //                 // no break — wrap-around candidates may still exist at end of list
    //             }
    //         }
    //     }
    //     return nbrs;
    // }

    vector<pair<int,int>> getNeighbors(int ri, int ci) const
    {
        vector<pair<int,int>> nbrs;
        const GraphNode& cur = G_c[ri][ci];

        for (int dr : {-1, 1})
        {
            int rn = ri + dr;
            if (rn < 0 || rn >= NUM_RINGS) continue;

            bool found_right_gap = false;

            for (int cn = 0; cn < (int)G_c[rn].size(); ++cn)
            {
                const GraphNode& cand = G_c[rn][cn];

                int gap_fwd = (cand.start_pos - cur.end_pos   + NUM_COLS) % NUM_COLS;
                int gap_bwd = (cur.start_pos  - cand.end_pos  + NUM_COLS) % NUM_COLS;
                bool overlaps = (gap_fwd == 0 || gap_bwd == 0 ||
                                (cand.start_pos <= cur.end_pos && cand.end_pos >= cur.start_pos));

                // Directed boundary distance:
                // dr = +1 → cand is a higher ring (more upward beam)
                //   compare cur's y_max (top edge) with cand's y_min (bottom edge)
                // dr = -1 → cand is a lower ring (more downward beam)  
                //   compare cur's y_min (bottom edge) with cand's y_max (top edge)
                float boundary_dist;
                if (dr == 1) {
                    // searching upward: gap from cur's top to cand's bottom
                    float dy = std::max(0.f, cand.y_min - cur.y_max);
                    float dx = std::max(0.f, std::max(cur.x_min - cand.x_max,
                                                    cand.x_min - cur.x_max));
                    boundary_dist = std::sqrt(dx*dx + dy*dy);
                } else {
                    // searching downward: gap from cand's top to cur's bottom
                    float dy = std::max(0.f, cur.y_min - cand.y_max);
                    float dx = std::max(0.f, std::max(cur.x_min - cand.x_max,
                                                    cand.x_min - cur.x_max));
                    boundary_dist = std::sqrt(dx*dx + dy*dy);
                }

                float rp = (cur.range + cand.range) / 2.0f;
                float Th_d = static_cast<float>(0.05f + boundary_dist * deg2rad(0.5));

                if (overlaps) {
                    if (boundary_dist < Th_d)
                        nbrs.push_back({rn, cn});
                } else if (!found_right_gap && cand.start_pos > cur.end_pos) {
                    if (boundary_dist < Th_d)
                        nbrs.push_back({rn, cn});
                    found_right_gap = true;
                }
            }
        }
        return nbrs;
    }

    // ==============================================================
    // Distance between two graph nodes (mean-range difference)
    // ==============================================================
    // float nodeDist(const GraphNode& a, const GraphNode& b) const
    // {
    //     float dx = a.x_mean - b.x_mean;
    //     float dy = a.y_mean - b.y_mean;
    //     return std::sqrt(dx*dx + dy*dy);
    // }

    float nodeDist(const GraphNode& a, const GraphNode& b, int dr = 0) const
    {
        float dx = std::max(a.x_min - b.x_max, b.x_min - a.x_max);

        float dy;
        if (dr == 1) {
            // b is upper neighbor: gap from a's top to b's bottom
            dy = b.y_min - a.y_max;
        } else if (dr == -1) {
            // b is lower neighbor: gap from b's top to a's bottom
            dy = a.y_min - b.y_max;
        } else {
            // no direction specified: symmetric min gap
            dy = std::max(a.y_min - b.y_max, b.y_min - a.y_max);
        }

        // return std::sqrt(dx*dx + dy*dy);
        return dy / 2.0f;
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

    vector<vector<Vector3d>> extractClusters() {
        std::unordered_map<int, vector<Vector3d>> mp;

        for (size_t i = 0; i < G_r.size(); ++i) {
            if (!G_r[i].valid) continue;
            int c = cluster_ids_[i];
            // int c = G_r[i].alpha;
            if (c < 0) continue;
            mp[c].emplace_back(G_r[i].x, G_r[i].y, G_r[i].z);
        }

        vector<vector<Vector3d>> out;
        out.reserve(mp.size());
        for (auto& kv : mp)
            if (kv.second.size() >= 10)
                out.push_back(std::move(kv.second));
        return out;
    }

    void outputClusters(const vector<vector<Vector3d>>& clusters, sensor_msgs::msg::PointCloud2 obs_points, int max_num_points) {
        sensor_msgs::msg::PointCloud2 bev_points;
        bev_points.header.frame_id = "rslidar";
        bev_points.height = 1;

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

        sensor_msgs::PointCloud2Modifier modifier_bev(bev_points);
        modifier_bev.setPointCloud2FieldsByString(2, "xyz", "rgb");
        modifier_bev.resize(max_num_points);

        sensor_msgs::PointCloud2Iterator<float> out_x(obs_points, "x");
        sensor_msgs::PointCloud2Iterator<float> out_y(obs_points, "y");
        sensor_msgs::PointCloud2Iterator<float> out_z(obs_points, "z");
        sensor_msgs::PointCloud2Iterator<float> out_id(obs_points, "id");

        sensor_msgs::PointCloud2Iterator<float> bev_x(bev_points, "x");
        sensor_msgs::PointCloud2Iterator<float> bev_y(bev_points, "y");
        sensor_msgs::PointCloud2Iterator<float> bev_z(bev_points, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> bev_r(bev_points, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> bev_g(bev_points, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> bev_b(bev_points, "b");

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

                // std_msgs::msg::ColorRGBA color = getColorFromId(id);

                *out_x = x;
                *out_y = y;
                *out_z = z;
                *out_id = id;

                *bev_x = x;
                *bev_y = y;
                *bev_z = z;
                 uint32_t hash = static_cast<uint32_t>(id * 2654435761 % 4294967296);
                *bev_r =  ((hash & 0xFF0000) >> 16) / 255.0f * 255;
                *bev_g = ((hash & 0x00FF00) >> 8) / 255.0f * 255;
                *bev_b = (hash & 0x0000FF) / 255.0f * 255;
                
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
                ++bev_x; ++bev_y; ++bev_z; ++bev_r;
                ++bev_g; ++bev_b;
            }
            
            obstacles_msg.obstacles.push_back(pc_msg);
        }
        
        bev_points.width = static_cast<uint32_t>(bev_points.data.size() / bev_points.point_step);
        bev_points.row_step = bev_points.point_step * bev_points.width;

        bev_pub_->publish(bev_points);

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
    float Thz_ = static_cast<float>(deg2rad(10.0));

    MatrixXd C_prev_;
    MatrixXd data_;
    std::array<double,NUM_RINGS> vert_angles_rad_;
    vector<RangeNode> G_r;
    vector<vector<GraphNode>> G_c;
    vector<int> cluster_ids_;
    vector<int> channel_ids_;
    vector<bool> visited_; 
    bool angles_calibrated_;
    std::vector<int> sorted_ring_order_;  // sorted_idx → original ring idx
    std::vector<int> ring_remap_;         // original ring idx → sorted_idx
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr bev_pub_;

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
