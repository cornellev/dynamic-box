// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cev_msgs:msg/SensorCollect.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__BUILDER_HPP_
#define CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cev_msgs/msg/detail/sensor_collect__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cev_msgs
{

namespace msg
{

namespace builder
{

class Init_SensorCollect_steering_angle
{
public:
  explicit Init_SensorCollect_steering_angle(::cev_msgs::msg::SensorCollect & msg)
  : msg_(msg)
  {}
  ::cev_msgs::msg::SensorCollect steering_angle(::cev_msgs::msg::SensorCollect::_steering_angle_type arg)
  {
    msg_.steering_angle = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cev_msgs::msg::SensorCollect msg_;
};

class Init_SensorCollect_velocity
{
public:
  explicit Init_SensorCollect_velocity(::cev_msgs::msg::SensorCollect & msg)
  : msg_(msg)
  {}
  Init_SensorCollect_steering_angle velocity(::cev_msgs::msg::SensorCollect::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_SensorCollect_steering_angle(msg_);
  }

private:
  ::cev_msgs::msg::SensorCollect msg_;
};

class Init_SensorCollect_timestamp
{
public:
  Init_SensorCollect_timestamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SensorCollect_velocity timestamp(::cev_msgs::msg::SensorCollect::_timestamp_type arg)
  {
    msg_.timestamp = std::move(arg);
    return Init_SensorCollect_velocity(msg_);
  }

private:
  ::cev_msgs::msg::SensorCollect msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cev_msgs::msg::SensorCollect>()
{
  return cev_msgs::msg::builder::Init_SensorCollect_timestamp();
}

}  // namespace cev_msgs

#endif  // CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__BUILDER_HPP_
