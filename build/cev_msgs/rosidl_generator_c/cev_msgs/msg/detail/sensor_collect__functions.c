// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from cev_msgs:msg/SensorCollect.idl
// generated code does not contain a copyright notice
#include "cev_msgs/msg/detail/sensor_collect__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
cev_msgs__msg__SensorCollect__init(cev_msgs__msg__SensorCollect * msg)
{
  if (!msg) {
    return false;
  }
  // timestamp
  // velocity
  // steering_angle
  return true;
}

void
cev_msgs__msg__SensorCollect__fini(cev_msgs__msg__SensorCollect * msg)
{
  if (!msg) {
    return;
  }
  // timestamp
  // velocity
  // steering_angle
}

bool
cev_msgs__msg__SensorCollect__are_equal(const cev_msgs__msg__SensorCollect * lhs, const cev_msgs__msg__SensorCollect * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // timestamp
  if (lhs->timestamp != rhs->timestamp) {
    return false;
  }
  // velocity
  if (lhs->velocity != rhs->velocity) {
    return false;
  }
  // steering_angle
  if (lhs->steering_angle != rhs->steering_angle) {
    return false;
  }
  return true;
}

bool
cev_msgs__msg__SensorCollect__copy(
  const cev_msgs__msg__SensorCollect * input,
  cev_msgs__msg__SensorCollect * output)
{
  if (!input || !output) {
    return false;
  }
  // timestamp
  output->timestamp = input->timestamp;
  // velocity
  output->velocity = input->velocity;
  // steering_angle
  output->steering_angle = input->steering_angle;
  return true;
}

cev_msgs__msg__SensorCollect *
cev_msgs__msg__SensorCollect__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cev_msgs__msg__SensorCollect * msg = (cev_msgs__msg__SensorCollect *)allocator.allocate(sizeof(cev_msgs__msg__SensorCollect), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cev_msgs__msg__SensorCollect));
  bool success = cev_msgs__msg__SensorCollect__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cev_msgs__msg__SensorCollect__destroy(cev_msgs__msg__SensorCollect * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cev_msgs__msg__SensorCollect__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cev_msgs__msg__SensorCollect__Sequence__init(cev_msgs__msg__SensorCollect__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cev_msgs__msg__SensorCollect * data = NULL;

  if (size) {
    data = (cev_msgs__msg__SensorCollect *)allocator.zero_allocate(size, sizeof(cev_msgs__msg__SensorCollect), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cev_msgs__msg__SensorCollect__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cev_msgs__msg__SensorCollect__fini(&data[i - 1]);
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
cev_msgs__msg__SensorCollect__Sequence__fini(cev_msgs__msg__SensorCollect__Sequence * array)
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
      cev_msgs__msg__SensorCollect__fini(&array->data[i]);
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

cev_msgs__msg__SensorCollect__Sequence *
cev_msgs__msg__SensorCollect__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cev_msgs__msg__SensorCollect__Sequence * array = (cev_msgs__msg__SensorCollect__Sequence *)allocator.allocate(sizeof(cev_msgs__msg__SensorCollect__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cev_msgs__msg__SensorCollect__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cev_msgs__msg__SensorCollect__Sequence__destroy(cev_msgs__msg__SensorCollect__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cev_msgs__msg__SensorCollect__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cev_msgs__msg__SensorCollect__Sequence__are_equal(const cev_msgs__msg__SensorCollect__Sequence * lhs, const cev_msgs__msg__SensorCollect__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cev_msgs__msg__SensorCollect__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cev_msgs__msg__SensorCollect__Sequence__copy(
  const cev_msgs__msg__SensorCollect__Sequence * input,
  cev_msgs__msg__SensorCollect__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cev_msgs__msg__SensorCollect);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    cev_msgs__msg__SensorCollect * data =
      (cev_msgs__msg__SensorCollect *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cev_msgs__msg__SensorCollect__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          cev_msgs__msg__SensorCollect__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!cev_msgs__msg__SensorCollect__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
