// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cev_msgs:msg/Obstacles.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__OBSTACLES__BUILDER_HPP_
#define CEV_MSGS__MSG__DETAIL__OBSTACLES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cev_msgs/msg/detail/obstacles__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cev_msgs
{

namespace msg
{

namespace builder
{

class Init_Obstacles_obstacles
{
public:
  Init_Obstacles_obstacles()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::cev_msgs::msg::Obstacles obstacles(::cev_msgs::msg::Obstacles::_obstacles_type arg)
  {
    msg_.obstacles = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cev_msgs::msg::Obstacles msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cev_msgs::msg::Obstacles>()
{
  return cev_msgs::msg::builder::Init_Obstacles_obstacles();
}

}  // namespace cev_msgs

#endif  // CEV_MSGS__MSG__DETAIL__OBSTACLES__BUILDER_HPP_
