// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cluster_node:msg/Obstacle.idl
// generated code does not contain a copyright notice

#ifndef CLUSTER_NODE__MSG__DETAIL__OBSTACLE__STRUCT_H_
#define CLUSTER_NODE__MSG__DETAIL__OBSTACLE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'pose'
#include "geometry_msgs/msg/detail/pose__struct.h"

/// Struct defined in msg/Obstacle in the package cluster_node.
typedef struct cluster_node__msg__Obstacle
{
  int32_t id;
  geometry_msgs__msg__Pose pose;
  float length;
  float width;
  float z_min;
  float z_max;
  bool blocking;
} cluster_node__msg__Obstacle;

// Struct for a sequence of cluster_node__msg__Obstacle.
typedef struct cluster_node__msg__Obstacle__Sequence
{
  cluster_node__msg__Obstacle * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cluster_node__msg__Obstacle__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CLUSTER_NODE__MSG__DETAIL__OBSTACLE__STRUCT_H_
