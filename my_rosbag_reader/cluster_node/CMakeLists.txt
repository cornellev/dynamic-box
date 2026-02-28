cmake_minimum_required(VERSION 3.8)
project(cluster_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(Open3D_DIR "/usr/lib/cmake/Open3D")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Open3D REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(cev_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Obstacle.msg"
  "msg/ObstacleArray.msg"
  DEPENDENCIES std_msgs geometry_msgs cev_msgs
)

add_executable(cluster_cpp src/cluster_node.cpp)
add_executable(two_layer_cpp src/two_layer_graph.cpp)

ament_target_dependencies(
  cluster_cpp
  rclcpp 
  cev_msgs
  sensor_msgs 
  std_msgs
  pcl_conversions
  pcl_ros
  Open3D
  geometry_msgs
  visualization_msgs
)

ament_target_dependencies(
  two_layer_cpp
  rclcpp 
  cev_msgs
  sensor_msgs 
  std_msgs
  pcl_conversions
  pcl_ros
  Open3D
  geometry_msgs
  visualization_msgs
)

ament_export_dependencies(rosidl_default_runtime)

rosidl_target_interfaces(cluster_cpp
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
install(TARGETS
  cluster_cpp
  DESTINATION lib/${PROJECT_NAME})

target_link_libraries(cluster_cpp
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${Open3D_LIBRARIES}
)

rosidl_target_interfaces(two_layer_cpp
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
install(TARGETS
  two_layer_cpp
  DESTINATION lib/${PROJECT_NAME})

target_link_libraries(two_layer_cpp
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${Open3D_LIBRARIES}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
