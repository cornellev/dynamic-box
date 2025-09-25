// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cev_msgs:msg/SensorCollect.idl
// generated code does not contain a copyright notice

#ifndef CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__STRUCT_H_
#define CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/SensorCollect in the package cev_msgs.
/**
  * SensorCollect.msg
  *
  * float64 : timestamp      [ Message time ]
  * float32 : velocity       [ Velocity in m/s]
  * float32 : steering_angle [ Steering angle in radians ]
 */
typedef struct cev_msgs__msg__SensorCollect
{
  double timestamp;
  float velocity;
  float steering_angle;
} cev_msgs__msg__SensorCollect;

// Struct for a sequence of cev_msgs__msg__SensorCollect.
typedef struct cev_msgs__msg__SensorCollect__Sequence
{
  cev_msgs__msg__SensorCollect * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cev_msgs__msg__SensorCollect__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CEV_MSGS__MSG__DETAIL__SENSOR_COLLECT__STRUCT_H_
