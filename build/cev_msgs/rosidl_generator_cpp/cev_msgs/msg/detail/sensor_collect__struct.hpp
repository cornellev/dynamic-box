// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cev_msgs:msg/SensorCollect.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__STRUCT_HPP_
#define CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__cev_msgs__msg__SensorCollect __attribute__((deprecated))
#else
# define DEPRECATED__cev_msgs__msg__SensorCollect __declspec(deprecated)
#endif

namespace cev_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SensorCollect_
{
  using Type = SensorCollect_<ContainerAllocator>;

  explicit SensorCollect_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0.0;
      this->velocity = 0.0f;
      this->steering_angle = 0.0f;
    }
  }

  explicit SensorCollect_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0.0;
      this->velocity = 0.0f;
      this->steering_angle = 0.0f;
    }
  }

  // field types and members
  using _timestamp_type =
    double;
  _timestamp_type timestamp;
  using _velocity_type =
    float;
  _velocity_type velocity;
  using _steering_angle_type =
    float;
  _steering_angle_type steering_angle;

  // setters for named parameter idiom
  Type & set__timestamp(
    const double & _arg)
  {
    this->timestamp = _arg;
    return *this;
  }
  Type & set__velocity(
    const float & _arg)
  {
    this->velocity = _arg;
    return *this;
  }
  Type & set__steering_angle(
    const float & _arg)
  {
    this->steering_angle = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cev_msgs::msg::SensorCollect_<ContainerAllocator> *;
  using ConstRawPtr =
    const cev_msgs::msg::SensorCollect_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cev_msgs::msg::SensorCollect_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cev_msgs::msg::SensorCollect_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cev_msgs::msg::SensorCollect_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cev_msgs::msg::SensorCollect_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cev_msgs::msg::SensorCollect_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cev_msgs::msg::SensorCollect_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cev_msgs::msg::SensorCollect_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cev_msgs::msg::SensorCollect_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cev_msgs__msg__SensorCollect
    std::shared_ptr<cev_msgs::msg::SensorCollect_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cev_msgs__msg__SensorCollect
    std::shared_ptr<cev_msgs::msg::SensorCollect_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SensorCollect_ & other) const
  {
    if (this->timestamp != other.timestamp) {
      return false;
    }
    if (this->velocity != other.velocity) {
      return false;
    }
    if (this->steering_angle != other.steering_angle) {
      return false;
    }
    return true;
  }
  bool operator!=(const SensorCollect_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SensorCollect_

// alias to use template instance with default allocator
using SensorCollect =
  cev_msgs::msg::SensorCollect_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace cev_msgs

#endif  // CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__STRUCT_HPP_
