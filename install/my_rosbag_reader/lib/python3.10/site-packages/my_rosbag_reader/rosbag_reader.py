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
import matplotlib.pyplot as plt

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('rosbag_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            'unilidar/cloud',  # Topic name
            self.listener_callback,
            10  # Queue size
        )

    def listener_callback(self, msg):
        cloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        if cloud.size > 0:
            try: 
                with open(os.path.join("src/ackermann_kf/my_rosbag_reader", "point_cloud.json"), "r") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError): 
                print("HERE")
                json.dump({"points":cloud.tolist()}, f)

            self.get_logger().info(f"Received {len(cloud)} points")

            data["points"] = data["points"] + (cloud.tolist())
            with open(os.path.join("src/ackermann_kf/my_rosbag_reader", "point_cloud.json"), "w") as f:
                json.dump(data, f)
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
