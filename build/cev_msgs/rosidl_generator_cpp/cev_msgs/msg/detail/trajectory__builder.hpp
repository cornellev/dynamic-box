// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cev_msgs:msg/Trajectory.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__TRAJECTORY__BUILDER_HPP_
#define CEV_MSGS__MSG__DETAIL__TRAJECTORY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cev_msgs/msg/detail/trajectory__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cev_msgs
{

namespace msg
{

namespace builder
{

class Init_Trajectory_timestep
{
public:
  explicit Init_Trajectory_timestep(::cev_msgs::msg::Trajectory & msg)
  : msg_(msg)
  {}
  ::cev_msgs::msg::Trajectory timestep(::cev_msgs::msg::Trajectory::_timestep_type arg)
  {
    msg_.timestep = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cev_msgs::msg::Trajectory msg_;
};

class Init_Trajectory_waypoints
{
public:
  explicit Init_Trajectory_waypoints(::cev_msgs::msg::Trajectory & msg)
  : msg_(msg)
  {}
  Init_Trajectory_timestep waypoints(::cev_msgs::msg::Trajectory::_waypoints_type arg)
  {
    msg_.waypoints = std::move(arg);
    return Init_Trajectory_timestep(msg_);
  }

private:
  ::cev_msgs::msg::Trajectory msg_;
};

class Init_Trajectory_header
{
public:
  Init_Trajectory_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Trajectory_waypoints header(::cev_msgs::msg::Trajectory::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_Trajectory_waypoints(msg_);
  }

private:
  ::cev_msgs::msg::Trajectory msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cev_msgs::msg::Trajectory>()
{
  return cev_msgs::msg::builder::Init_Trajectory_header();
}

}  // namespace cev_msgs

#endif  // CEV_MSGS__MSG__DETAIL__TRAJECTORY__BUILDER_HPP_
