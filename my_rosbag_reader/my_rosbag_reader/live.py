import time
import numpy as np
import json
import matplotlib.pyplot as plt
import open3d as o3d
import rclpy
from rclpy.node import Node
from cev_msgs.msg import Obstacle
from cev_msgs.msg import Obstacles
import sensor_msgs
import struct
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from collections import defaultdict
# from ouster.sdk import open_source, pcap, client, _bindings
# from google.cloud import storage
# from google.oauth2 import service_account
import socketio
import cluster_cpp
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# import pcl
# from pcl_helper import pcl_to_ros

# TURN THIS INTO A PUBLISHER THAT PUBLISHES COLMAP_MODEL.JSON

sio = socketio.Client()

COLORS = {}
BACKEND_URL = "http://localhost:5000" # Use localhost for development
# BACKEND_URL = "https://im-map.onrender.com"

# Connect the SocketIO client
try:
    sio.connect(BACKEND_URL, transports=["websocket"])
except Exception as e:
    print(f"Failed to connect to WebSocket server: {e}")
    exit() # Exit if we can't connect


# credentials = service_account.Credentials.from_service_account_info(CLOUD_KEY)
# g_client = storage.Client(credentials=credentials, project='im-map')

np.set_printoptions(suppress=True)

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('rosbag_subscriber')
        self.iter = 0
        self.C_prev = np.array([[0,0,0,0]])
        self.data = np.array([])
        self.subscription = self.create_subscription(
            PointCloud2,
            '/rslidar_points',  # Topic name
            self.listener_callback,
            10  # Queue size
        )

        self.publisher = self.create_publisher(
            Obstacles, 
            '/rslidar_obstacles',
            10
        )

    def listener_callback(self, msg):
        # Z IS Y (map from 0 to 720), Y IS X (map from 0 to 1280)
        # POINT CLOUD PREPROCESSING
        cloud = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        cloud = np.array(cloud.tolist())

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])

        # Downsample the point cloud
        downsampled = pcd.voxel_down_sample(voxel_size=0.05)

        # Convert back to numpy
        cloud = np.asarray(downsampled.points)
        cloud = np.vstack((cloud[:,0], cloud[:,1], cloud[:,2], cloud[:,2])).T

        if cloud.size > 0:
            # start = time.time()
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            self.get_logger().info(f"LIVE: Received {cloud.shape[0]} points")
            self.data = cloud 
            
            # map past centroids to closest point in new point cloud to get seeds:
            # cluster growing on seeds is parallelized (run time is constrained on the largest resultant cluster)
            tree = cKDTree(cloud)
            _, indices = tree.query(self.C_prev)
            seeds = np.unique(cloud[indices], axis = 0)

            C = []
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(grow_seed, seed, self.data, self.iter, self.C_prev)
                    for seed in seeds
                ]
                results = [future.result() for future in futures]

            for c in results:
                C.extend(c)
            
            C = np.array(C, dtype=object)
            mask = np.array([cluster.shape[0] > 10 for cluster in np.array(C)])
            C = C[mask]

            # unable to parallelize new obstacles or outlier points, have to use 1 seed -> n clusters extraction
            remaining_C = []
            cluster_points = np.unique(np.vstack(C) if len(C) > 0 else np.empty((0, self.data.shape[1])), axis = 0)
            mask = np.invert(np.any((cloud[:, None] == cluster_points).all(-1), axis=1))
            remaining_cloud = cloud[mask]

            c, _ = cluster_cpp.euclidean_cluster(seeds = remaining_cloud[0], cloud = remaining_cloud, radius = 0.1, MIN_CLUSTER_SIZE = 10, mode = "cartesian", reorder = False, cloud_prev = self.C_prev)
            remaining_C.extend(c)
            remaining_C = np.array(remaining_C, dtype=object)
            mask = np.array([cluster.shape[0] > 10 for cluster in np.array(remaining_C)])
            
            if remaining_C[mask].size:
                C = np.hstack((C, remaining_C[mask]))

            # need to reorder here from x, y, z -> z, x, y
            self.display_clusters(ax, np.array(C), [np.median(c, axis = 0) for c in C])
            self.C_prev = np.array([np.median(c, axis = 0) for c in C])
        else:
            self.get_logger().warn("Received empty point cloud")

        self.iter += 1

    def display_clusters(self, ax, clusters, prev, reorder=True): 
        global COLORS
        new_points = []
        colmap_model = {}

        colors = plt.cm.hsv(np.linspace(0, 0.8, len(clusters)))
        
        sorted_indices = np.argsort([np.mean(arr) for arr in clusters])
        clusters = clusters[sorted_indices]
        prev_centroids = np.array(prev)[sorted_indices]

        rgb_colors = (colors[:, :3] * 255)

        if len(COLORS.keys()) != 0:
            prev_rgb_colors = COLORS
        else:
            prev_rgb_colors = {}

        COLORS = {}
        
        # ith key in rgb_colors should correspond to ith prev_centroid color
        obstacles_arr = []

        for i, _ in enumerate(clusters):
            # DISPLAY PREVIOUS CENTROID ELLIPSOID
            data = clusters[i][:, :3]
                
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data)

            # Downsample the point cloud
            downsampled = pcd.voxel_down_sample(voxel_size=0.1)

            # Convert back to numpy
            data = np.asarray(downsampled.points)
            
            if i < len(list(prev_rgb_colors.values())) and tuple(prev_centroids[i][:3]) in prev_rgb_colors:
                COLORS[tuple(np.mean(clusters[i], axis=0)[:3])] = prev_rgb_colors[tuple(prev_centroids[i][:3])]
            elif i % 2 == 0:
                r, g, b = rgb_colors[i]
                COLORS[tuple(np.mean(clusters[i], axis=0)[:3])] = [255 - r, 255 - g, 255 - b]
            else:
                COLORS[tuple(np.mean(clusters[i], axis=0)[:3])] = rgb_colors[i]
                
            color = COLORS[tuple(np.mean(clusters[i], axis=0)[:3])]
            for _, points in enumerate(np.array(data)):
                y, z, x = points
                new_points = new_points + [{'id': i, 'x': x, 'y': y, 'z': z, 'r': color[0], 'g': color[1], 'b': color[2]}]
            y, z, x = np.mean(clusters[i], axis=0)[:3]
            new_points = new_points + [{'id': i, 'x': x, 'y': y, 'z': z, 'r': max(0, color[0] - 30), 'g': max(0, color[1] - 30), 'b': max(0, color[2] - 30)}]

            pc_msg = PointCloud2()
            pc_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]

            pc_msg.is_bigendian = False
            pc_msg.point_step = 12  # 3 * 4 bytes
            pc_msg.row_step = pc_msg.point_step * data.shape[0]
            pc_msg.is_dense = True

            buffer = []
            for p in data:
                buffer.append(struct.pack('fff', p[0], p[1], p[2]))
            pc_msg.data = b"".join(buffer)

            # msg = Obstacle()
            # msg.x, msg.y, msg.z = x, y, z
            obstacles_arr.append(pc_msg)

        obstacles_msg = Obstacles()
        obstacles_msg.obstacles = obstacles_arr
        self.publisher.publish(obstacles_msg)
        self.get_logger().info(f'Published {len(obstacles_arr)} obstacles')
        
        colmap_model["points"] = new_points
        colmap_model["cameras"] = []
        colmap_model["median"] = [0,0,0]
        colmap_model["mean"] = [0,0,0]

        # bucket = g_client.get_bucket(f'public_matches')
        with open(f"colmap_model.json", "w") as f:
            try:
                if sio.connected:
                    sio.emit('live_lidar_updates', colmap_model) # This is the key change
                    print(f"Successfully emitted 'live_lidar_updates' event: {len(colmap_model['points'])}")
                else:
                    print("SocketIO client is not connected. Skipping emission.")
            except Exception as e:
                print(f"Error emitting WebSocket event: {e}")
            colmap_model = json.dump(colmap_model, f, indent=4)

        # blob = bucket.blob(f"Alexander_Nevsky_Cathedral,_Sofia/sparse/optical_flow/colmap_model.json")
        # blob.upload_from_filename(f"colmap_model.json")

class Node(object):
    def __init__(self, data=None, axis=None, left=None, right=None):
        self.data = data
        self.axis = axis
        self.left = left
        self.right = right
        self.prev = np.array([0,0,0,0])
        self.C_prev = np.array([], dtype=object)

    def make_kdtree(self, points, axis, dim):
        if (points.shape[0] <= 10):
            # left = None, right = None is a Leaf
            return Node(data = points, axis = axis) if points.shape[0] > 5 else Node()

        median_id = np.argpartition(points[:, axis], len(points)//2)[len(points)//2]
        median = points[median_id, axis]
        left = points[points[:, axis] < median]
        right = points[points[:, axis] >= median]

        return Node(data = points, axis = axis, 
                    left = self.make_kdtree(points = left, axis = (axis + 1) % dim, dim = dim),
                    right = self.make_kdtree(points = right, axis = (axis + 1) % dim, dim = dim))
    
    def search_point(self, ax, point, radius, thres, C, C_prev):
        split_axis = np.median(self.data[:, self.axis])
        scaled_radius = radius
        # scaled_radius = max(radius, radius / np.sqrt(np.sum(point[0][:3]**2)))

        if (len(self.data[np.linalg.norm(self.data[:,:3] - point[0][:3], axis=1) <= scaled_radius]) > 0):
            # centroids = np.array([np.mean(C, axis = 0)[:3] for C in C_prev])
            # try:
            #     # match current point to closest centroid in cluster: 
            #     curr_centroid = np.mean(C[-1], axis = 0)[:3]
            #     i = np.argmin(np.sum((centroids - curr_centroid) ** 2, axis = 1))
            #     # print(np.array([np.median(C, axis = 0)[3] for C in C_prev])[i])
            #     past_centroid = centroids[i]
            #     self.C_prev = centroids
            #     self.prev = np.hstack((past_centroid, np.array([np.array([np.median(C, axis = 0)[3] for C in C_prev])[i]])))
            # except:
            #     pass

            diff = np.abs(self.data[:, :3] - point[0][:3])
            mask = np.all(diff < np.array([radius, radius, 2 * radius]), axis=1)
            in_radius = self.data[mask]
  
            return point + [in_radius]
        
        if (point[0][:3][self.axis] - scaled_radius < split_axis):
            point = self.left.search_point(ax, point, radius, thres, C, C_prev)
        elif (point[0][:3][self.axis] + scaled_radius >= split_axis):
            point = self.right.search_point(ax, point, radius, thres, C, C_prev)

        return point

    def search_tree(self, ax, root, start_point, radius, thres, C, C_prev):
        stack = [start_point]
        unexplored_set = set(map(tuple, Node.unexplored))

        while stack:
            point = stack.pop()
            neighbors = np.vstack(self.search_point(ax, [point], radius, thres, C, C_prev)[1:])
            neighbors = [tuple(p) for p in neighbors if tuple(p) in unexplored_set]
            
            for neighbor in neighbors:
                if neighbor not in C[-1]:
                    C[-1].append(neighbor)
                    stack.append(np.array(neighbor))  # Add to stack for further exploration
                    unexplored_set.remove(neighbor)

        return Node.unexplored

def grow_seed(seed, data, itr, C_prev):
    if itr == 0:
        c, _ = cluster_cpp.euclidean_cluster(seeds = seed, cloud = data, radius = 0.1, MIN_CLUSTER_SIZE = 10, mode = "cartesian", reorder = False, cloud_prev = C_prev)
    else:
        c, _ = cluster_cpp.euclidean_cluster(seeds = seed, cloud = data, radius = 0.1, MIN_CLUSTER_SIZE = 10, mode = "cartesian", reorder = False, cloud_prev = C_prev, MAX_CLUSTER_NUM = 1)
    return c
         
def merge(data, thres):
    merged_dict = defaultdict(list)
    for arr, x, y, z, I in data:
        if (abs(np.median(arr[:, 3]) - I) < 100):
            coords_tuple = (x,y,z,I) 
            # Append the first column (arr) to the corresponding coordinate key
            merged_dict[coords_tuple].extend(arr)
        else:
            coords_tuple = (x,y,z,np.median(arr[:, 3]))
            merged_dict[coords_tuple].extend(arr)

    # Convert the merged values back to NumPy arrays
    merged_result = np.array([[np.array(values), coords[0], coords[1], coords[2]] for coords, values in merged_dict.items()], dtype=object)
    return merged_result

def euclidean_cluster(ax, cloud, radius, intensity_threshold, MIN_CLUSTER_SIZE = 1, mode = "cartesian", cloud_prev = np.array([]), reorder=True):
    C = []
    prev = []
    
    if reorder:
        if (mode == "spherical"):
            z, x, y = cloud[:, 0]*np.sin(cloud[:, 2])*np.cos(cloud[:, 1]), cloud[:, 0]*np.sin(cloud[:, 2])*np.sin(cloud[:, 1]), cloud[:, 0]*np.cos(cloud[:, 2])
        else:
            z, x, y = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    else:
        x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]

    cloud = np.array([x, y, z, cloud[:, 3]]).T
    Node.unexplored = set(map(tuple, cloud))

    kd_tree = Node().make_kdtree(cloud, 0, 3)

    while len(Node.unexplored) != 0:
        
        next_point = np.array(next(iter(Node.unexplored)))
        C.append([tuple(next_point)])

        Node.unexplored.remove(tuple(next_point))

        Node.unexplored = kd_tree.search_tree(ax, kd_tree, next_point, radius, intensity_threshold, C, cloud_prev)
        
        prev.append(kd_tree.prev)

    clusters = np.empty(len(C), dtype=object)

    for i, cluster in enumerate(C):
        clusters[i] = np.array(cluster)
    
    clusters = np.array([cluster for cluster in np.column_stack((clusters, prev)) if cluster[0].shape[0] > MIN_CLUSTER_SIZE], dtype = object)

    return clusters[:,0], clusters[:,1:]

def euclidean_cluster_2d(ax, cloud, radius, intensity_threshold, MIN_CLUSTER_SIZE = 1, mode = "cartesian", cloud_prev = np.array([])):
    C = []
    prevs = []

    for i, c in enumerate(cloud):
        try:
            c_out, prev = euclidean_cluster(ax, c * [1, 0, 1, 1], radius, intensity_threshold, MIN_CLUSTER_SIZE, mode, cloud_prev, reorder=False)  
        except:
            c_out, prev = [np.array(c)], [list(np.median(cloud_prev[i], axis=0))] 
            
        C = C + [c for c in c_out]
        prevs = prevs + [p for p in prev]
    
    clusters = np.empty(len(C), dtype=object)

    for i, cluster in enumerate(C):
        clusters[i] = cluster
    
    return clusters, prevs


# pcap_path = '1024x10-dual.pcap'
# metadata_path = '1024x10-dual.json'

# source = open_source(pcap_path, meta = [metadata_path], index=True)
# with open(metadata_path, 'r') as f:
#     info = client.SensorInfo(f.read())

# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# C_prev = np.array([])

# ctr = 0
# source_iter = iter(source)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    
    main()

    # for scan in source_iter:
    #     ctr += 1
    #     xyz = client.XYZLut(info)(scan)
    #     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.reshape((-1, 3))))

    #     # Extract point data
    #     ros_cloud = np.asarray(pcd.points)
    #     o3d_pcd = o3d.geometry.PointCloud()
    #     o3d_pcd.points = o3d.utility.Vector3dVector(ros_cloud)
    #     downsamp = o3d_pcd.voxel_down_sample(0.3)
    #     cloud = np.asarray(downsamp.points)

    #     # o3d.visualization.draw_geometries([downsamp])
    #     cloud = np.vstack((cloud[:,0], cloud[:,1], cloud[:,2], cloud[:,2])).T

    #     if cloud.size > 0:
    #         C, prev = euclidean_cluster(ax = ax, cloud = cloud, radius = 0.3, intensity_threshold = 0.3, MIN_CLUSTER_SIZE = 10, mode = "cartesian", cloud_prev = C_prev)
    #         # print(f"BEFORE {C.shape}")
    #         # C_, prev_ = euclidean_cluster_2d(ax = ax, cloud = C, radius = 0.29, intensity_threshold = 0.3, MIN_CLUSTER_SIZE = 5, mode = "cartesian", cloud_prev = C)
    #         # print(f"AFTER {C_.shape}")
    #         # update previous clustering with new C
    #         display_clusters(ax, C, prev)
            
    #         # display_clusters(ax, C_, prev_)
    #         C_prev = C
