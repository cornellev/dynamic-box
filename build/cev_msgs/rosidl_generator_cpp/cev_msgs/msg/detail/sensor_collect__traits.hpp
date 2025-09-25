// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cev_msgs:msg/SensorCollect.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__TRAITS_HPP_
#define CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "cev_msgs/msg/detail/sensor_collect__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace cev_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const SensorCollect & msg,
  std::ostream & out)
{
  out << "{";
  // member: timestamp
  {
    out << "timestamp: ";
    rosidl_generator_traits::value_to_yaml(msg.timestamp, out);
    out << ", ";
  }

  // member: velocity
  {
    out << "velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.velocity, out);
    out << ", ";
  }

  // member: steering_angle
  {
    out << "steering_angle: ";
    rosidl_generator_traits::value_to_yaml(msg.steering_angle, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SensorCollect & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: timestamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "timestamp: ";
    rosidl_generator_traits::value_to_yaml(msg.timestamp, out);
    out << "\n";
  }

  // member: velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.velocity, out);
    out << "\n";
  }

  // member: steering_angle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "steering_angle: ";
    rosidl_generator_traits::value_to_yaml(msg.steering_angle, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SensorCollect & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace cev_msgs

namespace rosidl_generator_traits
{

[[deprecated("use cev_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const cev_msgs::msg::SensorCollect & msg,
  std::ostream & out, size_t indentation = 0)
{
  cev_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cev_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const cev_msgs::msg::SensorCollect & msg)
{
  return cev_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<cev_msgs::msg::SensorCollect>()
{
  return "cev_msgs::msg::SensorCollect";
}

template<>
inline const char * name<cev_msgs::msg::SensorCollect>()
{
  return "cev_msgs/msg/SensorCollect";
}

template<>
struct has_fixed_size<cev_msgs::msg::SensorCollect>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<cev_msgs::msg::SensorCollect>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<cev_msgs::msg::SensorCollect>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__TRAITS_HPP_
