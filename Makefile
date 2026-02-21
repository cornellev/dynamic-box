.PHONY: all lidar cluster bag obstacle

all: lidar cluster bag obstacle

lidar:
	gnome-terminal -- bash -c "ros2 launch rslidar_sdk start.py; exec bash"

cluster:
	gnome-terminal -- bash -c "cd /dynamic_box/my_rosbag_reader && \
	colcon build --packages-select cluster_node && \
	source install/setup.bash && \
	ros2 run cluster_node cluster_cpp; exec bash"

bag:
	gnome-terminal -- bash -c "cd /dynamic_box/baggies && \
	source install/setup.bash && \
	ros2 bag play <ROS_BAG_NAME>; exec bash"

obstacle:
	gnome-terminal -- bash -c "cd /Obstacle_node && \
	colcon build --packages-select obstacle && \
	source install/setup.bash && \
	ros2 run obstacle obstacle_tracker; exec bash"
