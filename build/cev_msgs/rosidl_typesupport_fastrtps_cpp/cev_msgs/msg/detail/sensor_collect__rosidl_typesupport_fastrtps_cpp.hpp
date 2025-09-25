// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from cev_msgs:msg/SensorCollect.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "cev_msgs/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "cev_msgs/msg/detail/sensor_collect__struct.hpp"

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

namespace cev_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cev_msgs
cdr_serialize(
  const cev_msgs::msg::SensorCollect & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cev_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  cev_msgs::msg::SensorCollect & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cev_msgs
get_serialized_size(
  const cev_msgs::msg::SensorCollect & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cev_msgs
max_serialized_size_SensorCollect(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace cev_msgs

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cev_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, cev_msgs, msg, SensorCollect)();

#ifdef __cplusplus
}
#endif

#endif  // CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
