// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from cev_msgs:msg/Obstacles.idl
// generated code does not contain a copyright notice
#include "cev_msgs/msg/detail/obstacles__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `obstacles`
#include "cev_msgs/msg/detail/obstacle__functions.h"

bool
cev_msgs__msg__Obstacles__init(cev_msgs__msg__Obstacles * msg)
{
  if (!msg) {
    return false;
  }
  // obstacles
  if (!cev_msgs__msg__Obstacle__Sequence__init(&msg->obstacles, 0)) {
    cev_msgs__msg__Obstacles__fini(msg);
    return false;
  }
  return true;
}

void
cev_msgs__msg__Obstacles__fini(cev_msgs__msg__Obstacles * msg)
{
  if (!msg) {
    return;
  }
  // obstacles
  cev_msgs__msg__Obstacle__Sequence__fini(&msg->obstacles);
}

bool
cev_msgs__msg__Obstacles__are_equal(const cev_msgs__msg__Obstacles * lhs, const cev_msgs__msg__Obstacles * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // obstacles
  if (!cev_msgs__msg__Obstacle__Sequence__are_equal(
      &(lhs->obstacles), &(rhs->obstacles)))
  {
    return false;
  }
  return true;
}

bool
cev_msgs__msg__Obstacles__copy(
  const cev_msgs__msg__Obstacles * input,
  cev_msgs__msg__Obstacles * output)
{
  if (!input || !output) {
    return false;
  }
  // obstacles
  if (!cev_msgs__msg__Obstacle__Sequence__copy(
      &(input->obstacles), &(output->obstacles)))
  {
    return false;
  }
  return true;
}

cev_msgs__msg__Obstacles *
cev_msgs__msg__Obstacles__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cev_msgs__msg__Obstacles * msg = (cev_msgs__msg__Obstacles *)allocator.allocate(sizeof(cev_msgs__msg__Obstacles), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cev_msgs__msg__Obstacles));
  bool success = cev_msgs__msg__Obstacles__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cev_msgs__msg__Obstacles__destroy(cev_msgs__msg__Obstacles * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cev_msgs__msg__Obstacles__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cev_msgs__msg__Obstacles__Sequence__init(cev_msgs__msg__Obstacles__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cev_msgs__msg__Obstacles * data = NULL;

  if (size) {
    data = (cev_msgs__msg__Obstacles *)allocator.zero_allocate(size, sizeof(cev_msgs__msg__Obstacles), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cev_msgs__msg__Obstacles__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cev_msgs__msg__Obstacles__fini(&data[i - 1]);
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
cev_msgs__msg__Obstacles__Sequence__fini(cev_msgs__msg__Obstacles__Sequence * array)
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
      cev_msgs__msg__Obstacles__fini(&array->data[i]);
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

cev_msgs__msg__Obstacles__Sequence *
cev_msgs__msg__Obstacles__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cev_msgs__msg__Obstacles__Sequence * array = (cev_msgs__msg__Obstacles__Sequence *)allocator.allocate(sizeof(cev_msgs__msg__Obstacles__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cev_msgs__msg__Obstacles__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cev_msgs__msg__Obstacles__Sequence__destroy(cev_msgs__msg__Obstacles__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cev_msgs__msg__Obstacles__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cev_msgs__msg__Obstacles__Sequence__are_equal(const cev_msgs__msg__Obstacles__Sequence * lhs, const cev_msgs__msg__Obstacles__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cev_msgs__msg__Obstacles__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cev_msgs__msg__Obstacles__Sequence__copy(
  const cev_msgs__msg__Obstacles__Sequence * input,
  cev_msgs__msg__Obstacles__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cev_msgs__msg__Obstacles);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    cev_msgs__msg__Obstacles * data =
      (cev_msgs__msg__Obstacles *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cev_msgs__msg__Obstacles__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          cev_msgs__msg__Obstacles__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!cev_msgs__msg__Obstacles__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
