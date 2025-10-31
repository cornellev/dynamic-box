// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cev_msgs:msg/Obstacles.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__OBSTACLES__STRUCT_H_
#define CEV_MSGS__MSG__DETAIL__OBSTACLES__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'obstacles'
#include "cev_msgs/msg/detail/obstacle__struct.h"

/// Struct defined in msg/Obstacles in the package cev_msgs.
/**
  * Obstacle.msg
  *
  * Obstacle[] : obstacles
 */
typedef struct cev_msgs__msg__Obstacles
{
  cev_msgs__msg__Obstacle__Sequence obstacles;
} cev_msgs__msg__Obstacles;

// Struct for a sequence of cev_msgs__msg__Obstacles.
typedef struct cev_msgs__msg__Obstacles__Sequence
{
  cev_msgs__msg__Obstacles * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cev_msgs__msg__Obstacles__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CEV_MSGS__MSG__DETAIL__OBSTACLES__STRUCT_H_
