// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cev_msgs:msg/Obstacle.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__OBSTACLE__BUILDER_HPP_
#define CEV_MSGS__MSG__DETAIL__OBSTACLE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cev_msgs/msg/detail/obstacle__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cev_msgs
{

namespace msg
{

namespace builder
{

class Init_Obstacle_max_radius
{
public:
  explicit Init_Obstacle_max_radius(::cev_msgs::msg::Obstacle & msg)
  : msg_(msg)
  {}
  ::cev_msgs::msg::Obstacle max_radius(::cev_msgs::msg::Obstacle::_max_radius_type arg)
  {
    msg_.max_radius = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cev_msgs::msg::Obstacle msg_;
};

class Init_Obstacle_z
{
public:
  explicit Init_Obstacle_z(::cev_msgs::msg::Obstacle & msg)
  : msg_(msg)
  {}
  Init_Obstacle_max_radius z(::cev_msgs::msg::Obstacle::_z_type arg)
  {
    msg_.z = std::move(arg);
    return Init_Obstacle_max_radius(msg_);
  }

private:
  ::cev_msgs::msg::Obstacle msg_;
};

class Init_Obstacle_y
{
public:
  explicit Init_Obstacle_y(::cev_msgs::msg::Obstacle & msg)
  : msg_(msg)
  {}
  Init_Obstacle_z y(::cev_msgs::msg::Obstacle::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_Obstacle_z(msg_);
  }

private:
  ::cev_msgs::msg::Obstacle msg_;
};

class Init_Obstacle_x
{
public:
  Init_Obstacle_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Obstacle_y x(::cev_msgs::msg::Obstacle::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_Obstacle_y(msg_);
  }

private:
  ::cev_msgs::msg::Obstacle msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cev_msgs::msg::Obstacle>()
{
  return cev_msgs::msg::builder::Init_Obstacle_x();
}

}  // namespace cev_msgs

#endif  // CEV_MSGS__MSG__DETAIL__OBSTACLE__BUILDER_HPP_
