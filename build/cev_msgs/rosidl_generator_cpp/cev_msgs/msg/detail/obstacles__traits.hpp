// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cev_msgs:msg/Obstacles.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__OBSTACLES__TRAITS_HPP_
#define CEV_MSGS__MSG__DETAIL__OBSTACLES__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "cev_msgs/msg/detail/obstacles__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'obstacles'
#include "cev_msgs/msg/detail/obstacle__traits.hpp"

namespace cev_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Obstacles & msg,
  std::ostream & out)
{
  out << "{";
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
  const Obstacles & msg,
  std::ostream & out, size_t indentation = 0)
{
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

inline std::string to_yaml(const Obstacles & msg, bool use_flow_style = false)
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
  const cev_msgs::msg::Obstacles & msg,
  std::ostream & out, size_t indentation = 0)
{
  cev_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cev_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const cev_msgs::msg::Obstacles & msg)
{
  return cev_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<cev_msgs::msg::Obstacles>()
{
  return "cev_msgs::msg::Obstacles";
}

template<>
inline const char * name<cev_msgs::msg::Obstacles>()
{
  return "cev_msgs/msg/Obstacles";
}

template<>
struct has_fixed_size<cev_msgs::msg::Obstacles>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<cev_msgs::msg::Obstacles>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<cev_msgs::msg::Obstacles>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CEV_MSGS__MSG__DETAIL__OBSTACLES__TRAITS_HPP_
