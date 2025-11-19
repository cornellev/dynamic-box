// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from cluster_node:msg/ObstacleArray.idl
// generated code does not contain a copyright notice

#ifndef CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "cluster_node/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "cluster_node/msg/detail/obstacle_array__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace cluster_node
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cluster_node
cdr_serialize(
  const cluster_node::msg::ObstacleArray & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cluster_node
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  cluster_node::msg::ObstacleArray & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cluster_node
get_serialized_size(
  const cluster_node::msg::ObstacleArray & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cluster_node
max_serialized_size_ObstacleArray(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace cluster_node

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cluster_node
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, cluster_node, msg, ObstacleArray)();

#ifdef __cplusplus
}
#endif

#endif  // CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
