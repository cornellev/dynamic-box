// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cluster_node:msg/ObstacleArray.idl
// generated code does not contain a copyright notice

#ifndef CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__BUILDER_HPP_
#define CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cluster_node/msg/detail/obstacle_array__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cluster_node
{

namespace msg
{

namespace builder
{

class Init_ObstacleArray_obstacles
{
public:
  explicit Init_ObstacleArray_obstacles(::cluster_node::msg::ObstacleArray & msg)
  : msg_(msg)
  {}
  ::cluster_node::msg::ObstacleArray obstacles(::cluster_node::msg::ObstacleArray::_obstacles_type arg)
  {
    msg_.obstacles = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cluster_node::msg::ObstacleArray msg_;
};

class Init_ObstacleArray_header
{
public:
  Init_ObstacleArray_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ObstacleArray_obstacles header(::cluster_node::msg::ObstacleArray::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_ObstacleArray_obstacles(msg_);
  }

private:
  ::cluster_node::msg::ObstacleArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cluster_node::msg::ObstacleArray>()
{
  return cluster_node::msg::builder::Init_ObstacleArray_header();
}

}  // namespace cluster_node

#endif  // CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__BUILDER_HPP_
