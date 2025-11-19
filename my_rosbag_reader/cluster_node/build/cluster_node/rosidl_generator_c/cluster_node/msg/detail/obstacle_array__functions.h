// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from cluster_node:msg/ObstacleArray.idl
// generated code does not contain a copyright notice

#ifndef CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__FUNCTIONS_H_
#define CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "cluster_node/msg/rosidl_generator_c__visibility_control.h"

#include "cluster_node/msg/detail/obstacle_array__struct.h"

/// Initialize msg/ObstacleArray message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * cluster_node__msg__ObstacleArray
 * )) before or use
 * cluster_node__msg__ObstacleArray__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_cluster_node
bool
cluster_node__msg__ObstacleArray__init(cluster_node__msg__ObstacleArray * msg);

/// Finalize msg/ObstacleArray message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cluster_node
void
cluster_node__msg__ObstacleArray__fini(cluster_node__msg__ObstacleArray * msg);

/// Create msg/ObstacleArray message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * cluster_node__msg__ObstacleArray__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cluster_node
cluster_node__msg__ObstacleArray *
cluster_node__msg__ObstacleArray__create();

/// Destroy msg/ObstacleArray message.
/**
 * It calls
 * cluster_node__msg__ObstacleArray__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cluster_node
void
cluster_node__msg__ObstacleArray__destroy(cluster_node__msg__ObstacleArray * msg);

/// Check for msg/ObstacleArray message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cluster_node
bool
cluster_node__msg__ObstacleArray__are_equal(const cluster_node__msg__ObstacleArray * lhs, const cluster_node__msg__ObstacleArray * rhs);

/// Copy a msg/ObstacleArray message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_cluster_node
bool
cluster_node__msg__ObstacleArray__copy(
  const cluster_node__msg__ObstacleArray * input,
  cluster_node__msg__ObstacleArray * output);

/// Initialize array of msg/ObstacleArray messages.
/**
 * It allocates the memory for the number of elements and calls
 * cluster_node__msg__ObstacleArray__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_cluster_node
bool
cluster_node__msg__ObstacleArray__Sequence__init(cluster_node__msg__ObstacleArray__Sequence * array, size_t size);

/// Finalize array of msg/ObstacleArray messages.
/**
 * It calls
 * cluster_node__msg__ObstacleArray__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cluster_node
void
cluster_node__msg__ObstacleArray__Sequence__fini(cluster_node__msg__ObstacleArray__Sequence * array);

/// Create array of msg/ObstacleArray messages.
/**
 * It allocates the memory for the array and calls
 * cluster_node__msg__ObstacleArray__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cluster_node
cluster_node__msg__ObstacleArray__Sequence *
cluster_node__msg__ObstacleArray__Sequence__create(size_t size);

/// Destroy array of msg/ObstacleArray messages.
/**
 * It calls
 * cluster_node__msg__ObstacleArray__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cluster_node
void
cluster_node__msg__ObstacleArray__Sequence__destroy(cluster_node__msg__ObstacleArray__Sequence * array);

/// Check for msg/ObstacleArray message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cluster_node
bool
cluster_node__msg__ObstacleArray__Sequence__are_equal(const cluster_node__msg__ObstacleArray__Sequence * lhs, const cluster_node__msg__ObstacleArray__Sequence * rhs);

/// Copy an array of msg/ObstacleArray messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_cluster_node
bool
cluster_node__msg__ObstacleArray__Sequence__copy(
  const cluster_node__msg__ObstacleArray__Sequence * input,
  cluster_node__msg__ObstacleArray__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // CLUSTER_NODE__MSG__DETAIL__OBSTACLE_ARRAY__FUNCTIONS_H_
