// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cev_msgs:msg/Obstacle.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__OBSTACLE__STRUCT_H_
#define CEV_MSGS__MSG__DETAIL__OBSTACLE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Obstacle in the package cev_msgs.
/**
  * Obstacle.msg
  *
  * float64 : x
  * float64 : y
  * float64 : z
  * float64 : max_radius
 */
typedef struct cev_msgs__msg__Obstacle
{
  double x;
  double y;
  double z;
  double max_radius;
} cev_msgs__msg__Obstacle;

// Struct for a sequence of cev_msgs__msg__Obstacle.
typedef struct cev_msgs__msg__Obstacle__Sequence
{
  cev_msgs__msg__Obstacle * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cev_msgs__msg__Obstacle__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CEV_MSGS__MSG__DETAIL__OBSTACLE__STRUCT_H_
