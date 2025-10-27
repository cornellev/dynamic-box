// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cev_msgs:msg/Waypoint.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__WAYPOINT__TRAITS_HPP_
#define CEV_MSGS__MSG__DETAIL__WAYPOINT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "cev_msgs/msg/detail/waypoint__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace cev_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Waypoint & msg,
  std::ostream & out)
{
  out << "{";
  // member: x
  {
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << ", ";
  }

  // member: y
  {
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << ", ";
  }

  // member: v
  {
    out << "v: ";
    rosidl_generator_traits::value_to_yaml(msg.v, out);
    out << ", ";
  }

  // member: tau
  {
    out << "tau: ";
    rosidl_generator_traits::value_to_yaml(msg.tau, out);
    out << ", ";
  }

  // member: theta
  {
    out << "theta: ";
    rosidl_generator_traits::value_to_yaml(msg.theta, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Waypoint & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << "\n";
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << "\n";
  }

  // member: v
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "v: ";
    rosidl_generator_traits::value_to_yaml(msg.v, out);
    out << "\n";
  }

  // member: tau
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tau: ";
    rosidl_generator_traits::value_to_yaml(msg.tau, out);
    out << "\n";
  }

  // member: theta
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "theta: ";
    rosidl_generator_traits::value_to_yaml(msg.theta, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Waypoint & msg, bool use_flow_style = false)
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
  const cev_msgs::msg::Waypoint & msg,
  std::ostream & out, size_t indentation = 0)
{
  cev_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cev_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const cev_msgs::msg::Waypoint & msg)
{
  return cev_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<cev_msgs::msg::Waypoint>()
{
  return "cev_msgs::msg::Waypoint";
}

template<>
inline const char * name<cev_msgs::msg::Waypoint>()
{
  return "cev_msgs/msg/Waypoint";
}

template<>
struct has_fixed_size<cev_msgs::msg::Waypoint>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<cev_msgs::msg::Waypoint>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<cev_msgs::msg::Waypoint>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CEV_MSGS__MSG__DETAIL__WAYPOINT__TRAITS_HPP_
