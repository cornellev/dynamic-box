// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cluster_node:msg/Obstacle.idl
// generated code does not contain a copyright notice

#ifndef CLUSTER_NODE__MSG__DETAIL__OBSTACLE__TRAITS_HPP_
#define CLUSTER_NODE__MSG__DETAIL__OBSTACLE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "cluster_node/msg/detail/obstacle__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'pose'
#include "geometry_msgs/msg/detail/pose__traits.hpp"

namespace cluster_node
{

namespace msg
{

inline void to_flow_style_yaml(
  const Obstacle & msg,
  std::ostream & out)
{
  out << "{";
  // member: id
  {
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << ", ";
  }

  // member: pose
  {
    out << "pose: ";
    to_flow_style_yaml(msg.pose, out);
    out << ", ";
  }

  // member: length
  {
    out << "length: ";
    rosidl_generator_traits::value_to_yaml(msg.length, out);
    out << ", ";
  }

  // member: width
  {
    out << "width: ";
    rosidl_generator_traits::value_to_yaml(msg.width, out);
    out << ", ";
  }

  // member: z_min
  {
    out << "z_min: ";
    rosidl_generator_traits::value_to_yaml(msg.z_min, out);
    out << ", ";
  }

  // member: z_max
  {
    out << "z_max: ";
    rosidl_generator_traits::value_to_yaml(msg.z_max, out);
    out << ", ";
  }

  // member: blocking
  {
    out << "blocking: ";
    rosidl_generator_traits::value_to_yaml(msg.blocking, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Obstacle & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << "\n";
  }

  // member: pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pose:\n";
    to_block_style_yaml(msg.pose, out, indentation + 2);
  }

  // member: length
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "length: ";
    rosidl_generator_traits::value_to_yaml(msg.length, out);
    out << "\n";
  }

  // member: width
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "width: ";
    rosidl_generator_traits::value_to_yaml(msg.width, out);
    out << "\n";
  }

  // member: z_min
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "z_min: ";
    rosidl_generator_traits::value_to_yaml(msg.z_min, out);
    out << "\n";
  }

  // member: z_max
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "z_max: ";
    rosidl_generator_traits::value_to_yaml(msg.z_max, out);
    out << "\n";
  }

  // member: blocking
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "blocking: ";
    rosidl_generator_traits::value_to_yaml(msg.blocking, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Obstacle & msg, bool use_flow_style = false)
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
  const cluster_node::msg::Obstacle & msg,
  std::ostream & out, size_t indentation = 0)
{
  cluster_node::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cluster_node::msg::to_yaml() instead")]]
inline std::string to_yaml(const cluster_node::msg::Obstacle & msg)
{
  return cluster_node::msg::to_yaml(msg);
}

template<>
inline const char * data_type<cluster_node::msg::Obstacle>()
{
  return "cluster_node::msg::Obstacle";
}

template<>
inline const char * name<cluster_node::msg::Obstacle>()
{
  return "cluster_node/msg/Obstacle";
}

template<>
struct has_fixed_size<cluster_node::msg::Obstacle>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Pose>::value> {};

template<>
struct has_bounded_size<cluster_node::msg::Obstacle>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Pose>::value> {};

template<>
struct is_message<cluster_node::msg::Obstacle>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CLUSTER_NODE__MSG__DETAIL__OBSTACLE__TRAITS_HPP_
