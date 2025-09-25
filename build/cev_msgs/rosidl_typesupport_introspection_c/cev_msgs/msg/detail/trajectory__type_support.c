// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from cev_msgs:msg/Trajectory.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "cev_msgs/msg/detail/trajectory__rosidl_typesupport_introspection_c.h"
#include "cev_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "cev_msgs/msg/detail/trajectory__functions.h"
#include "cev_msgs/msg/detail/trajectory__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `waypoints`
#include "cev_msgs/msg/waypoint.h"
// Member `waypoints`
#include "cev_msgs/msg/detail/waypoint__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cev_msgs__msg__Trajectory__init(message_memory);
}

void cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_fini_function(void * message_memory)
{
  cev_msgs__msg__Trajectory__fini(message_memory);
}

size_t cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__size_function__Trajectory__waypoints(
  const void * untyped_member)
{
  const cev_msgs__msg__Waypoint__Sequence * member =
    (const cev_msgs__msg__Waypoint__Sequence *)(untyped_member);
  return member->size;
}

const void * cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__get_const_function__Trajectory__waypoints(
  const void * untyped_member, size_t index)
{
  const cev_msgs__msg__Waypoint__Sequence * member =
    (const cev_msgs__msg__Waypoint__Sequence *)(untyped_member);
  return &member->data[index];
}

void * cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__get_function__Trajectory__waypoints(
  void * untyped_member, size_t index)
{
  cev_msgs__msg__Waypoint__Sequence * member =
    (cev_msgs__msg__Waypoint__Sequence *)(untyped_member);
  return &member->data[index];
}

void cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__fetch_function__Trajectory__waypoints(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const cev_msgs__msg__Waypoint * item =
    ((const cev_msgs__msg__Waypoint *)
    cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__get_const_function__Trajectory__waypoints(untyped_member, index));
  cev_msgs__msg__Waypoint * value =
    (cev_msgs__msg__Waypoint *)(untyped_value);
  *value = *item;
}

void cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__assign_function__Trajectory__waypoints(
  void * untyped_member, size_t index, const void * untyped_value)
{
  cev_msgs__msg__Waypoint * item =
    ((cev_msgs__msg__Waypoint *)
    cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__get_function__Trajectory__waypoints(untyped_member, index));
  const cev_msgs__msg__Waypoint * value =
    (const cev_msgs__msg__Waypoint *)(untyped_value);
  *item = *value;
}

bool cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__resize_function__Trajectory__waypoints(
  void * untyped_member, size_t size)
{
  cev_msgs__msg__Waypoint__Sequence * member =
    (cev_msgs__msg__Waypoint__Sequence *)(untyped_member);
  cev_msgs__msg__Waypoint__Sequence__fini(member);
  return cev_msgs__msg__Waypoint__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_message_member_array[3] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cev_msgs__msg__Trajectory, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "waypoints",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cev_msgs__msg__Trajectory, waypoints),  // bytes offset in struct
    NULL,  // default value
    cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__size_function__Trajectory__waypoints,  // size() function pointer
    cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__get_const_function__Trajectory__waypoints,  // get_const(index) function pointer
    cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__get_function__Trajectory__waypoints,  // get(index) function pointer
    cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__fetch_function__Trajectory__waypoints,  // fetch(index, &value) function pointer
    cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__assign_function__Trajectory__waypoints,  // assign(index, value) function pointer
    cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__resize_function__Trajectory__waypoints  // resize(index) function pointer
  },
  {
    "timestep",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cev_msgs__msg__Trajectory, timestep),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_message_members = {
  "cev_msgs__msg",  // message namespace
  "Trajectory",  // message name
  3,  // number of fields
  sizeof(cev_msgs__msg__Trajectory),
  cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_message_member_array,  // message members
  cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_init_function,  // function to initialize message memory (memory has to be allocated)
  cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_message_type_support_handle = {
  0,
  &cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cev_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cev_msgs, msg, Trajectory)() {
  cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cev_msgs, msg, Waypoint)();
  if (!cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_message_type_support_handle.typesupport_identifier) {
    cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &cev_msgs__msg__Trajectory__rosidl_typesupport_introspection_c__Trajectory_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
