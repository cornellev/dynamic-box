// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from cluster_node:msg/Obstacle.idl
// generated code does not contain a copyright notice
#include "cluster_node/msg/detail/obstacle__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `pose`
#include "geometry_msgs/msg/detail/pose__functions.h"

bool
cluster_node__msg__Obstacle__init(cluster_node__msg__Obstacle * msg)
{
  if (!msg) {
    return false;
  }
  // id
  // pose
  if (!geometry_msgs__msg__Pose__init(&msg->pose)) {
    cluster_node__msg__Obstacle__fini(msg);
    return false;
  }
  // length
  // width
  // z_min
  // z_max
  // blocking
  return true;
}

void
cluster_node__msg__Obstacle__fini(cluster_node__msg__Obstacle * msg)
{
  if (!msg) {
    return;
  }
  // id
  // pose
  geometry_msgs__msg__Pose__fini(&msg->pose);
  // length
  // width
  // z_min
  // z_max
  // blocking
}

bool
cluster_node__msg__Obstacle__are_equal(const cluster_node__msg__Obstacle * lhs, const cluster_node__msg__Obstacle * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
    return false;
  }
  // pose
  if (!geometry_msgs__msg__Pose__are_equal(
      &(lhs->pose), &(rhs->pose)))
  {
    return false;
  }
  // length
  if (lhs->length != rhs->length) {
    return false;
  }
  // width
  if (lhs->width != rhs->width) {
    return false;
  }
  // z_min
  if (lhs->z_min != rhs->z_min) {
    return false;
  }
  // z_max
  if (lhs->z_max != rhs->z_max) {
    return false;
  }
  // blocking
  if (lhs->blocking != rhs->blocking) {
    return false;
  }
  return true;
}

bool
cluster_node__msg__Obstacle__copy(
  const cluster_node__msg__Obstacle * input,
  cluster_node__msg__Obstacle * output)
{
  if (!input || !output) {
    return false;
  }
  // id
  output->id = input->id;
  // pose
  if (!geometry_msgs__msg__Pose__copy(
      &(input->pose), &(output->pose)))
  {
    return false;
  }
  // length
  output->length = input->length;
  // width
  output->width = input->width;
  // z_min
  output->z_min = input->z_min;
  // z_max
  output->z_max = input->z_max;
  // blocking
  output->blocking = input->blocking;
  return true;
}

cluster_node__msg__Obstacle *
cluster_node__msg__Obstacle__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cluster_node__msg__Obstacle * msg = (cluster_node__msg__Obstacle *)allocator.allocate(sizeof(cluster_node__msg__Obstacle), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cluster_node__msg__Obstacle));
  bool success = cluster_node__msg__Obstacle__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cluster_node__msg__Obstacle__destroy(cluster_node__msg__Obstacle * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cluster_node__msg__Obstacle__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cluster_node__msg__Obstacle__Sequence__init(cluster_node__msg__Obstacle__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cluster_node__msg__Obstacle * data = NULL;

  if (size) {
    data = (cluster_node__msg__Obstacle *)allocator.zero_allocate(size, sizeof(cluster_node__msg__Obstacle), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cluster_node__msg__Obstacle__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cluster_node__msg__Obstacle__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
cluster_node__msg__Obstacle__Sequence__fini(cluster_node__msg__Obstacle__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      cluster_node__msg__Obstacle__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

cluster_node__msg__Obstacle__Sequence *
cluster_node__msg__Obstacle__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cluster_node__msg__Obstacle__Sequence * array = (cluster_node__msg__Obstacle__Sequence *)allocator.allocate(sizeof(cluster_node__msg__Obstacle__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cluster_node__msg__Obstacle__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cluster_node__msg__Obstacle__Sequence__destroy(cluster_node__msg__Obstacle__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cluster_node__msg__Obstacle__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cluster_node__msg__Obstacle__Sequence__are_equal(const cluster_node__msg__Obstacle__Sequence * lhs, const cluster_node__msg__Obstacle__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cluster_node__msg__Obstacle__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cluster_node__msg__Obstacle__Sequence__copy(
  const cluster_node__msg__Obstacle__Sequence * input,
  cluster_node__msg__Obstacle__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cluster_node__msg__Obstacle);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    cluster_node__msg__Obstacle * data =
      (cluster_node__msg__Obstacle *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cluster_node__msg__Obstacle__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          cluster_node__msg__Obstacle__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!cluster_node__msg__Obstacle__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
