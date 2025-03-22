import rclpy
import json
import os
import numpy as np
import cv2
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

import math
import sys
import matplotlib.pyplot as plt
import open3d as o3d
import timeit
from my_rosbag_reader import fuse 

np.set_printoptions(suppress=True)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('rosbag_subscriber')
        self.iter = 0
        self.C_prev = np.array([])
        self.data = np.array([])
        self.subscription = self.create_subscription(
            PointCloud2,
            'unilidar/cloud',  # Topic name
            self.listener_callback,
            10  # Queue size
        )

    def listener_callback(self, msg):
        # Z IS Y (map from 0 to 720), Y IS X (map from 0 to 1280)
        # POINT CLOUD PREPROCESSING
        cloud = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        cloud = np.array(cloud.tolist())
        cloud = cloud[cloud[:, 2] < 1.0]
        cloud_rho = np.sqrt(cloud[:, 0]**2 + cloud[:, 1]**2 + cloud[:, 2]**2)
        cloud_theta = np.arctan(cloud[:, 1]/cloud[:, 0])
        cloud_phi = np.arccos(cloud[:, 2]/cloud_rho)

        # Point cloud in spherical coordinates, rad = depth.
        cloud_sphr = np.array([cloud_rho, cloud_theta, cloud_phi, cloud[:, 3]]).T
        cloud_sphr = cloud_sphr[cloud_sphr[:, 0] < 4.0]
        cloud_sphr = cloud_sphr[cloud_sphr[:, 0] > 0.5]
        cloud_sphr = cloud_sphr[cloud_sphr[:, 1] > -math.pi/3]
        cloud_sphr = cloud_sphr[cloud_sphr[:, 1] < math.pi/3]
        cloud_sphr = cloud_sphr[cloud_sphr[:, 2] > math.pi/4]

        if cloud.size > 0:
            if (self.iter % 2 == 0):
                if (self.iter):
                    # cluster every 3 iteration-overlayed point clouds
                    C = fuse.euclidean_cluster(ax = ax, cloud = self.data, radius = 0.2, intensity_threshold = 20, MIN_CLUSTER_SIZE = 10, mode = "spherical", cloud_prev = self.C_prev)
                    # update previous clustering with new C
                    fuse.display_clusters(ax, C)
                    self.C_prev = C
                self.data = cloud_sphr
            else:
                self.data = np.vstack((self.data, cloud_sphr))      
            self.iter += 1

            # self.get_logger().info(f"Received {len(cloud)} points at iteration {self.iter-1}, data is {self.data.shape}")
        else:
            self.get_logger().warn("Received empty point cloud")


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
