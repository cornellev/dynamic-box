import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialReader
from std_msgs.msg import String
from sensor_msgs.msg import Image
from rclpy.serialization import deserialize_message
import os

class RosbagReader(Node):
    def __init__(self):
        super().__init__('rosbag_reader')
        self.bag_file_path = '/path/to/your/rosbag'  # Path to your rosbag file
        self.topic_data = {}

        self.reader = SequentialReader()
        self.reader.open(self.bag_file_path)

        # Start a timer to read the rosbag at a set frequency
        self.timer = self.create_timer(1.0, self.read_rosbag)

    def read_rosbag(self):
        try:
            # Get the next message in the bag file
            if self.reader.has_next():
                topic, msg, t = self.reader.read_next()
                self.get_logger().info(f'Reading from topic: {topic}')
                self.store_message_data(topic, msg)
            else:
                self.get_logger().info('End of bag file reached')
        except Exception as e:
            self.get_logger().error(f"Error reading rosbag: {str(e)}")

    def store_message_data(self, topic, msg):
        # Store data from the message (this example assumes a String message)
        if topic not in self.topic_data:
            self.topic_data[topic] = []
        
        # Deserialize message and store it
        if isinstance(msg, String):
            self.topic_data[topic].append(msg.data)
            self.get_logger().info(f"Stored message: {msg.data}")
        elif isinstance(msg, Image):
            # If it's an Image message, store metadata or the image data
            self.topic_data[topic].append(msg.header.stamp)
            self.get_logger().info(f"Stored Image message at time {msg.header.stamp.sec}")
        else:
            # Store other types of messages in a similar way
            self.topic_data[topic].append(msg)

    def shutdown(self):
        self.reader.close()
        self.get_logger().info('Reader closed and bag file shutdown.')


def main(args=None):
    rclpy.init(args=args)
    rosbag_reader = RosbagReader()

    rclpy.spin(rosbag_reader)

    # Shutdown the reader and close the rosbag file when done
    rosbag_reader.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
