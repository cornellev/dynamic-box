// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cluster_node:msg/Obstacle.idl
// generated code does not contain a copyright notice

#ifndef CLUSTER_NODE__MSG__DETAIL__OBSTACLE__STRUCT_HPP_
#define CLUSTER_NODE__MSG__DETAIL__OBSTACLE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'pose'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__cluster_node__msg__Obstacle __attribute__((deprecated))
#else
# define DEPRECATED__cluster_node__msg__Obstacle __declspec(deprecated)
#endif

namespace cluster_node
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Obstacle_
{
  using Type = Obstacle_<ContainerAllocator>;

  explicit Obstacle_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : pose(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0l;
      this->length = 0.0f;
      this->width = 0.0f;
      this->z_min = 0.0f;
      this->z_max = 0.0f;
      this->blocking = false;
    }
  }

  explicit Obstacle_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : pose(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0l;
      this->length = 0.0f;
      this->width = 0.0f;
      this->z_min = 0.0f;
      this->z_max = 0.0f;
      this->blocking = false;
    }
  }

  // field types and members
  using _id_type =
    int32_t;
  _id_type id;
  using _pose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _pose_type pose;
  using _length_type =
    float;
  _length_type length;
  using _width_type =
    float;
  _width_type width;
  using _z_min_type =
    float;
  _z_min_type z_min;
  using _z_max_type =
    float;
  _z_max_type z_max;
  using _blocking_type =
    bool;
  _blocking_type blocking;

  // setters for named parameter idiom
  Type & set__id(
    const int32_t & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__pose(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->pose = _arg;
    return *this;
  }
  Type & set__length(
    const float & _arg)
  {
    this->length = _arg;
    return *this;
  }
  Type & set__width(
    const float & _arg)
  {
    this->width = _arg;
    return *this;
  }
  Type & set__z_min(
    const float & _arg)
  {
    this->z_min = _arg;
    return *this;
  }
  Type & set__z_max(
    const float & _arg)
  {
    this->z_max = _arg;
    return *this;
  }
  Type & set__blocking(
    const bool & _arg)
  {
    this->blocking = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cluster_node::msg::Obstacle_<ContainerAllocator> *;
  using ConstRawPtr =
    const cluster_node::msg::Obstacle_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cluster_node::msg::Obstacle_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cluster_node::msg::Obstacle_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cluster_node::msg::Obstacle_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cluster_node::msg::Obstacle_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cluster_node::msg::Obstacle_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cluster_node::msg::Obstacle_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cluster_node::msg::Obstacle_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cluster_node::msg::Obstacle_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cluster_node__msg__Obstacle
    std::shared_ptr<cluster_node::msg::Obstacle_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cluster_node__msg__Obstacle
    std::shared_ptr<cluster_node::msg::Obstacle_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Obstacle_ & other) const
  {
    if (this->id != other.id) {
      return false;
    }
    if (this->pose != other.pose) {
      return false;
    }
    if (this->length != other.length) {
      return false;
    }
    if (this->width != other.width) {
      return false;
    }
    if (this->z_min != other.z_min) {
      return false;
    }
    if (this->z_max != other.z_max) {
      return false;
    }
    if (this->blocking != other.blocking) {
      return false;
    }
    return true;
  }
  bool operator!=(const Obstacle_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Obstacle_

// alias to use template instance with default allocator
using Obstacle =
  cluster_node::msg::Obstacle_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace cluster_node

#endif  // CLUSTER_NODE__MSG__DETAIL__OBSTACLE__STRUCT_HPP_
