.PHONY: all lidar cluster bag obstacle

all: lidar cluster bag obstacle

lidar:
	gnome-terminal -- bash -c "source install/setup.bash && \
	ros2 launch rslidar_sdk start.py; exec bash"

cluster:
	gnome-terminal -- bash -c "source install/setup.bash && \
	cd dynamic-box/my_rosbag_reader/cluster_node && \
	colcon build --packages-select cluster_node && \
	source install/setup.bash && \
	ros2 run cluster_node cluster_cpp; exec bash"

bag:
	gnome-terminal -- bash -c "source install/setup.bash && \
	cd dynamic-box/baggies/rosbag2_2022_04_14-ped_vehicle && \
	ros2 bag play rosbag2_2022_04_14-16_52_40_0.db3 --loop; exec bash"

obstacle:
	gnome-terminal -- bash -c "source install/setup.bash && \
	cd Obstacle_node && \
	colcon build --packages-select obstacle && \
	source install/setup.bash && \
	ros2 run obstacle obstacle_tracker; exec bash"
