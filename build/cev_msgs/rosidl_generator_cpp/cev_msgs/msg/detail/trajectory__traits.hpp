// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cev_msgs:msg/Trajectory.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__TRAJECTORY__TRAITS_HPP_
#define CEV_MSGS__MSG__DETAIL__TRAJECTORY__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "cev_msgs/msg/detail/trajectory__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'waypoints'
#include "cev_msgs/msg/detail/waypoint__traits.hpp"

namespace cev_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Trajectory & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: waypoints
  {
    if (msg.waypoints.size() == 0) {
      out << "waypoints: []";
    } else {
      out << "waypoints: [";
      size_t pending_items = msg.waypoints.size();
      for (auto item : msg.waypoints) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: timestep
  {
    out << "timestep: ";
    rosidl_generator_traits::value_to_yaml(msg.timestep, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Trajectory & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: waypoints
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.waypoints.size() == 0) {
      out << "waypoints: []\n";
    } else {
      out << "waypoints:\n";
      for (auto item : msg.waypoints) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: timestep
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "timestep: ";
    rosidl_generator_traits::value_to_yaml(msg.timestep, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Trajectory & msg, bool use_flow_style = false)
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
  const cev_msgs::msg::Trajectory & msg,
  std::ostream & out, size_t indentation = 0)
{
  cev_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cev_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const cev_msgs::msg::Trajectory & msg)
{
  return cev_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<cev_msgs::msg::Trajectory>()
{
  return "cev_msgs::msg::Trajectory";
}

template<>
inline const char * name<cev_msgs::msg::Trajectory>()
{
  return "cev_msgs/msg/Trajectory";
}

template<>
struct has_fixed_size<cev_msgs::msg::Trajectory>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<cev_msgs::msg::Trajectory>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<cev_msgs::msg::Trajectory>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CEV_MSGS__MSG__DETAIL__TRAJECTORY__TRAITS_HPP_
