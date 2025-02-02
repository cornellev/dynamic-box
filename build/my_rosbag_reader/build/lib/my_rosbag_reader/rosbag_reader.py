import rclpy
import pcl_ros
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2

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
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
