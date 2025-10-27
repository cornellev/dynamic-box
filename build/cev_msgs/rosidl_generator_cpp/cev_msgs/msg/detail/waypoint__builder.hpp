// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cev_msgs:msg/Waypoint.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__WAYPOINT__BUILDER_HPP_
#define CEV_MSGS__MSG__DETAIL__WAYPOINT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cev_msgs/msg/detail/waypoint__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cev_msgs
{

namespace msg
{

namespace builder
{

class Init_Waypoint_theta
{
public:
  explicit Init_Waypoint_theta(::cev_msgs::msg::Waypoint & msg)
  : msg_(msg)
  {}
  ::cev_msgs::msg::Waypoint theta(::cev_msgs::msg::Waypoint::_theta_type arg)
  {
    msg_.theta = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cev_msgs::msg::Waypoint msg_;
};

class Init_Waypoint_tau
{
public:
  explicit Init_Waypoint_tau(::cev_msgs::msg::Waypoint & msg)
  : msg_(msg)
  {}
  Init_Waypoint_theta tau(::cev_msgs::msg::Waypoint::_tau_type arg)
  {
    msg_.tau = std::move(arg);
    return Init_Waypoint_theta(msg_);
  }

private:
  ::cev_msgs::msg::Waypoint msg_;
};

class Init_Waypoint_v
{
public:
  explicit Init_Waypoint_v(::cev_msgs::msg::Waypoint & msg)
  : msg_(msg)
  {}
  Init_Waypoint_tau v(::cev_msgs::msg::Waypoint::_v_type arg)
  {
    msg_.v = std::move(arg);
    return Init_Waypoint_tau(msg_);
  }

private:
  ::cev_msgs::msg::Waypoint msg_;
};

class Init_Waypoint_y
{
public:
  explicit Init_Waypoint_y(::cev_msgs::msg::Waypoint & msg)
  : msg_(msg)
  {}
  Init_Waypoint_v y(::cev_msgs::msg::Waypoint::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_Waypoint_v(msg_);
  }

private:
  ::cev_msgs::msg::Waypoint msg_;
};

class Init_Waypoint_x
{
public:
  Init_Waypoint_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Waypoint_y x(::cev_msgs::msg::Waypoint::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_Waypoint_y(msg_);
  }

private:
  ::cev_msgs::msg::Waypoint msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cev_msgs::msg::Waypoint>()
{
  return cev_msgs::msg::builder::Init_Waypoint_x();
}

}  // namespace cev_msgs

#endif  // CEV_MSGS__MSG__DETAIL__WAYPOINT__BUILDER_HPP_
