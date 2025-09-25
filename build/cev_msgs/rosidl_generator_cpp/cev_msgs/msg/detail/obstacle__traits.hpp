// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cev_msgs:msg/Obstacle.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__OBSTACLE__TRAITS_HPP_
#define CEV_MSGS__MSG__DETAIL__OBSTACLE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "cev_msgs/msg/detail/obstacle__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace cev_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Obstacle & msg,
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

  // member: z
  {
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << ", ";
  }

  // member: max_radius
  {
    out << "max_radius: ";
    rosidl_generator_traits::value_to_yaml(msg.max_radius, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Obstacle & msg,
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

  // member: z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << "\n";
  }

  // member: max_radius
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "max_radius: ";
    rosidl_generator_traits::value_to_yaml(msg.max_radius, out);
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

}  // namespace cev_msgs

namespace rosidl_generator_traits
{

[[deprecated("use cev_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const cev_msgs::msg::Obstacle & msg,
  std::ostream & out, size_t indentation = 0)
{
  cev_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cev_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const cev_msgs::msg::Obstacle & msg)
{
  return cev_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<cev_msgs::msg::Obstacle>()
{
  return "cev_msgs::msg::Obstacle";
}

template<>
inline const char * name<cev_msgs::msg::Obstacle>()
{
  return "cev_msgs/msg/Obstacle";
}

template<>
struct has_fixed_size<cev_msgs::msg::Obstacle>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<cev_msgs::msg::Obstacle>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<cev_msgs::msg::Obstacle>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CEV_MSGS__MSG__DETAIL__OBSTACLE__TRAITS_HPP_
