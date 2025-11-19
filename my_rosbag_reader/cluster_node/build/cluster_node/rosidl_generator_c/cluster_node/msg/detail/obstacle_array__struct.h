// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cluster_node:msg/ObstacleArray.idl
// generated code does not contain a copyright notice

#ifndef CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__STRUCT_H_
#define CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__STRUCT_H_

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
// Member 'obstacles'
#include "cluster_node/msg/detail/obstacle__struct.h"

/// Struct defined in msg/ObstacleArray in the package cluster_node.
typedef struct cluster_node__msg__ObstacleArray
{
  std_msgs__msg__Header header;
  cluster_node__msg__Obstacle__Sequence obstacles;
} cluster_node__msg__ObstacleArray;

// Struct for a sequence of cluster_node__msg__ObstacleArray.
typedef struct cluster_node__msg__ObstacleArray__Sequence
{
  cluster_node__msg__ObstacleArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cluster_node__msg__ObstacleArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__STRUCT_H_
