import rclpy
import json
import os
import numpy as np
import cv2
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

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
        cloud_points = pc2.read_points(msg, field_names=("x", "y", "z"), reshape_organized_cloud=True, skip_nans=True)
        if cloud_points.size > 0:
            self.get_logger().info(f"Received {len(cloud_points)} points")

            with open(os.path.join("src/ackermann_kf/my_rosbag_reader", "point_cloud.json"), "w") as f:
                json.dump({"points":cloud_points.tolist()}, f)

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
