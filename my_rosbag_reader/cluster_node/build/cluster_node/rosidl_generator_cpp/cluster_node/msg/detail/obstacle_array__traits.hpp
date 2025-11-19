// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cluster_node:msg/ObstacleArray.idl
// generated code does not contain a copyright notice

#ifndef CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__TRAITS_HPP_
#define CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "cluster_node/msg/detail/obstacle_array__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'obstacles'
#include "cluster_node/msg/detail/obstacle__traits.hpp"

namespace cluster_node
{

namespace msg
{

inline void to_flow_style_yaml(
  const ObstacleArray & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: obstacles
  {
    if (msg.obstacles.size() == 0) {
      out << "obstacles: []";
    } else {
      out << "obstacles: [";
      size_t pending_items = msg.obstacles.size();
      for (auto item : msg.obstacles) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ObstacleArray & msg,
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

  // member: obstacles
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.obstacles.size() == 0) {
      out << "obstacles: []\n";
    } else {
      out << "obstacles:\n";
      for (auto item : msg.obstacles) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ObstacleArray & msg, bool use_flow_style = false)
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

}  // namespace cluster_node

namespace rosidl_generator_traits
{

[[deprecated("use cluster_node::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const cluster_node::msg::ObstacleArray & msg,
  std::ostream & out, size_t indentation = 0)
{
  cluster_node::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cluster_node::msg::to_yaml() instead")]]
inline std::string to_yaml(const cluster_node::msg::ObstacleArray & msg)
{
  return cluster_node::msg::to_yaml(msg);
}

template<>
inline const char * data_type<cluster_node::msg::ObstacleArray>()
{
  return "cluster_node::msg::ObstacleArray";
}

template<>
inline const char * name<cluster_node::msg::ObstacleArray>()
{
  return "cluster_node/msg/ObstacleArray";
}

template<>
struct has_fixed_size<cluster_node::msg::ObstacleArray>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<cluster_node::msg::ObstacleArray>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<cluster_node::msg::ObstacleArray>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__TRAITS_HPP_
