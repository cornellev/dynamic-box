// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from cev_msgs:msg/SensorCollect.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "cev_msgs/msg/detail/sensor_collect__rosidl_typesupport_introspection_c.h"
#include "cev_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "cev_msgs/msg/detail/sensor_collect__functions.h"
#include "cev_msgs/msg/detail/sensor_collect__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void cev_msgs__msg__SensorCollect__rosidl_typesupport_introspection_c__SensorCollect_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cev_msgs__msg__SensorCollect__init(message_memory);
}

void cev_msgs__msg__SensorCollect__rosidl_typesupport_introspection_c__SensorCollect_fini_function(void * message_memory)
{
  cev_msgs__msg__SensorCollect__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember cev_msgs__msg__SensorCollect__rosidl_typesupport_introspection_c__SensorCollect_message_member_array[3] = {
  {
    "timestamp",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cev_msgs__msg__SensorCollect, timestamp),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cev_msgs__msg__SensorCollect, velocity),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "steering_angle",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cev_msgs__msg__SensorCollect, steering_angle),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers cev_msgs__msg__SensorCollect__rosidl_typesupport_introspection_c__SensorCollect_message_members = {
  "cev_msgs__msg",  // message namespace
  "SensorCollect",  // message name
  3,  // number of fields
  sizeof(cev_msgs__msg__SensorCollect),
  cev_msgs__msg__SensorCollect__rosidl_typesupport_introspection_c__SensorCollect_message_member_array,  // message members
  cev_msgs__msg__SensorCollect__rosidl_typesupport_introspection_c__SensorCollect_init_function,  // function to initialize message memory (memory has to be allocated)
  cev_msgs__msg__SensorCollect__rosidl_typesupport_introspection_c__SensorCollect_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t cev_msgs__msg__SensorCollect__rosidl_typesupport_introspection_c__SensorCollect_message_type_support_handle = {
  0,
  &cev_msgs__msg__SensorCollect__rosidl_typesupport_introspection_c__SensorCollect_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cev_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cev_msgs, msg, SensorCollect)() {
  if (!cev_msgs__msg__SensorCollect__rosidl_typesupport_introspection_c__SensorCollect_message_type_support_handle.typesupport_identifier) {
    cev_msgs__msg__SensorCollect__rosidl_typesupport_introspection_c__SensorCollect_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &cev_msgs__msg__SensorCollect__rosidl_typesupport_introspection_c__SensorCollect_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
