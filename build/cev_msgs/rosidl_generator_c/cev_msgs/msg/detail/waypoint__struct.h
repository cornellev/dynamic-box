// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cev_msgs:msg/Waypoint.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__WAYPOINT__STRUCT_H_
#define CEV_MSGS__MSG__DETAIL__WAYPOINT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Waypoint in the package cev_msgs.
/**
  * Waypoint.msg
  *
  * float32 : x    [ x-coordinate in meters ]
  * float32 : y    [ y-coordinate in meters ]
  * float32 : v    [ velocity in m/s ]
 */
typedef struct cev_msgs__msg__Waypoint
{
  float x;
  float y;
  float v;
  float tau;
  float theta;
} cev_msgs__msg__Waypoint;

// Struct for a sequence of cev_msgs__msg__Waypoint.
typedef struct cev_msgs__msg__Waypoint__Sequence
{
  cev_msgs__msg__Waypoint * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cev_msgs__msg__Waypoint__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CEV_MSGS__MSG__DETAIL__WAYPOINT__STRUCT_H_
