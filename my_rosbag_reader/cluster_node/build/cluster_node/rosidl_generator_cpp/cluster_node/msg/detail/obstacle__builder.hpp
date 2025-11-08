// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cluster_node:msg/Obstacle.idl
// generated code does not contain a copyright notice

#ifndef CLUSTER_NODE__MSG__DETAIL__OBSTACLE__BUILDER_HPP_
#define CLUSTER_NODE__MSG__DETAIL__OBSTACLE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cluster_node/msg/detail/obstacle__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cluster_node
{

namespace msg
{

namespace builder
{

class Init_Obstacle_blocking
{
public:
  explicit Init_Obstacle_blocking(::cluster_node::msg::Obstacle & msg)
  : msg_(msg)
  {}
  ::cluster_node::msg::Obstacle blocking(::cluster_node::msg::Obstacle::_blocking_type arg)
  {
    msg_.blocking = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cluster_node::msg::Obstacle msg_;
};

class Init_Obstacle_z_max
{
public:
  explicit Init_Obstacle_z_max(::cluster_node::msg::Obstacle & msg)
  : msg_(msg)
  {}
  Init_Obstacle_blocking z_max(::cluster_node::msg::Obstacle::_z_max_type arg)
  {
    msg_.z_max = std::move(arg);
    return Init_Obstacle_blocking(msg_);
  }

private:
  ::cluster_node::msg::Obstacle msg_;
};

class Init_Obstacle_z_min
{
public:
  explicit Init_Obstacle_z_min(::cluster_node::msg::Obstacle & msg)
  : msg_(msg)
  {}
  Init_Obstacle_z_max z_min(::cluster_node::msg::Obstacle::_z_min_type arg)
  {
    msg_.z_min = std::move(arg);
    return Init_Obstacle_z_max(msg_);
  }

private:
  ::cluster_node::msg::Obstacle msg_;
};

class Init_Obstacle_width
{
public:
  explicit Init_Obstacle_width(::cluster_node::msg::Obstacle & msg)
  : msg_(msg)
  {}
  Init_Obstacle_z_min width(::cluster_node::msg::Obstacle::_width_type arg)
  {
    msg_.width = std::move(arg);
    return Init_Obstacle_z_min(msg_);
  }

private:
  ::cluster_node::msg::Obstacle msg_;
};

class Init_Obstacle_length
{
public:
  explicit Init_Obstacle_length(::cluster_node::msg::Obstacle & msg)
  : msg_(msg)
  {}
  Init_Obstacle_width length(::cluster_node::msg::Obstacle::_length_type arg)
  {
    msg_.length = std::move(arg);
    return Init_Obstacle_width(msg_);
  }

private:
  ::cluster_node::msg::Obstacle msg_;
};

class Init_Obstacle_pose
{
public:
  explicit Init_Obstacle_pose(::cluster_node::msg::Obstacle & msg)
  : msg_(msg)
  {}
  Init_Obstacle_length pose(::cluster_node::msg::Obstacle::_pose_type arg)
  {
    msg_.pose = std::move(arg);
    return Init_Obstacle_length(msg_);
  }

private:
  ::cluster_node::msg::Obstacle msg_;
};

class Init_Obstacle_id
{
public:
  Init_Obstacle_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Obstacle_pose id(::cluster_node::msg::Obstacle::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_Obstacle_pose(msg_);
  }

private:
  ::cluster_node::msg::Obstacle msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cluster_node::msg::Obstacle>()
{
  return cluster_node::msg::builder::Init_Obstacle_id();
}

}  // namespace cluster_node

#endif  // CLUSTER_NODE__MSG__DETAIL__OBSTACLE__BUILDER_HPP_
