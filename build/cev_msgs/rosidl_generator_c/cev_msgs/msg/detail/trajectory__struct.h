// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cev_msgs:msg/Trajectory.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__TRAJECTORY__STRUCT_H_
#define CEV_MSGS__MSG__DETAIL__TRAJECTORY__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'waypoints'
#include "cev_msgs/msg/detail/waypoint__struct.h"

/// Struct defined in msg/Trajectory in the package cev_msgs.
/**
  * Trajectory.msg
  *
  * Header     : header       [ Header ]
  * Waypoint[] : waypoints    [ Array of Waypoints ]
 */
typedef struct cev_msgs__msg__Trajectory
{
  std_msgs__msg__Header header;
  cev_msgs__msg__Waypoint__Sequence waypoints;
  float timestep;
} cev_msgs__msg__Trajectory;

// Struct for a sequence of cev_msgs__msg__Trajectory.
typedef struct cev_msgs__msg__Trajectory__Sequence
{
  cev_msgs__msg__Trajectory * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cev_msgs__msg__Trajectory__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CEV_MSGS__MSG__DETAIL__TRAJECTORY__STRUCT_H_
