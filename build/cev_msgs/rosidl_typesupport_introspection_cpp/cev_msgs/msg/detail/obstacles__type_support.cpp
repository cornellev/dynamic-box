// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from cev_msgs:msg/Obstacles.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "cev_msgs/msg/detail/obstacles__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace cev_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void Obstacles_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) cev_msgs::msg::Obstacles(_init);
}

void Obstacles_fini_function(void * message_memory)
{
  auto typed_message = static_cast<cev_msgs::msg::Obstacles *>(message_memory);
  typed_message->~Obstacles();
}

size_t size_function__Obstacles__obstacles(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<cev_msgs::msg::Obstacle> *>(untyped_member);
  return member->size();
}

const void * get_const_function__Obstacles__obstacles(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<cev_msgs::msg::Obstacle> *>(untyped_member);
  return &member[index];
}

void * get_function__Obstacles__obstacles(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<cev_msgs::msg::Obstacle> *>(untyped_member);
  return &member[index];
}

void fetch_function__Obstacles__obstacles(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const cev_msgs::msg::Obstacle *>(
    get_const_function__Obstacles__obstacles(untyped_member, index));
  auto & value = *reinterpret_cast<cev_msgs::msg::Obstacle *>(untyped_value);
  value = item;
}

void assign_function__Obstacles__obstacles(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<cev_msgs::msg::Obstacle *>(
    get_function__Obstacles__obstacles(untyped_member, index));
  const auto & value = *reinterpret_cast<const cev_msgs::msg::Obstacle *>(untyped_value);
  item = value;
}

void resize_function__Obstacles__obstacles(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<cev_msgs::msg::Obstacle> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Obstacles_message_member_array[1] = {
  {
    "obstacles",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<cev_msgs::msg::Obstacle>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cev_msgs::msg::Obstacles, obstacles),  // bytes offset in struct
    nullptr,  // default value
    size_function__Obstacles__obstacles,  // size() function pointer
    get_const_function__Obstacles__obstacles,  // get_const(index) function pointer
    get_function__Obstacles__obstacles,  // get(index) function pointer
    fetch_function__Obstacles__obstacles,  // fetch(index, &value) function pointer
    assign_function__Obstacles__obstacles,  // assign(index, value) function pointer
    resize_function__Obstacles__obstacles  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Obstacles_message_members = {
  "cev_msgs::msg",  // message namespace
  "Obstacles",  // message name
  1,  // number of fields
  sizeof(cev_msgs::msg::Obstacles),
  Obstacles_message_member_array,  // message members
  Obstacles_init_function,  // function to initialize message memory (memory has to be allocated)
  Obstacles_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Obstacles_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Obstacles_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace cev_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<cev_msgs::msg::Obstacles>()
{
  return &::cev_msgs::msg::rosidl_typesupport_introspection_cpp::Obstacles_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, cev_msgs, msg, Obstacles)() {
  return &::cev_msgs::msg::rosidl_typesupport_introspection_cpp::Obstacles_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
