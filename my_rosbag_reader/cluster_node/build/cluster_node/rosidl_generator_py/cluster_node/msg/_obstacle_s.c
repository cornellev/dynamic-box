// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from cluster_node:msg/Obstacle.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "cluster_node/msg/detail/obstacle__struct.h"
#include "cluster_node/msg/detail/obstacle__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool geometry_msgs__msg__pose__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * geometry_msgs__msg__pose__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool cluster_node__msg__obstacle__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[36];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("cluster_node.msg._obstacle.Obstacle", full_classname_dest, 35) == 0);
  }
  cluster_node__msg__Obstacle * ros_message = _ros_message;
  {  // id
    PyObject * field = PyObject_GetAttrString(_pymsg, "id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->id = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // pose
    PyObject * field = PyObject_GetAttrString(_pymsg, "pose");
    if (!field) {
      return false;
    }
    if (!geometry_msgs__msg__pose__convert_from_py(field, &ros_message->pose)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // length
    PyObject * field = PyObject_GetAttrString(_pymsg, "length");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->length = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // width
    PyObject * field = PyObject_GetAttrString(_pymsg, "width");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->width = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // z_min
    PyObject * field = PyObject_GetAttrString(_pymsg, "z_min");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->z_min = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // z_max
    PyObject * field = PyObject_GetAttrString(_pymsg, "z_max");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->z_max = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // blocking
    PyObject * field = PyObject_GetAttrString(_pymsg, "blocking");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->blocking = (Py_True == field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * cluster_node__msg__obstacle__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of Obstacle */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("cluster_node.msg._obstacle");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "Obstacle");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  cluster_node__msg__Obstacle * ros_message = (cluster_node__msg__Obstacle *)raw_ros_message;
  {  // id
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->id);
    {
      int rc = PyObject_SetAttrString(_pymessage, "id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pose
    PyObject * field = NULL;
    field = geometry_msgs__msg__pose__convert_to_py(&ros_message->pose);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "pose", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // length
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->length);
    {
      int rc = PyObject_SetAttrString(_pymessage, "length", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // width
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->width);
    {
      int rc = PyObject_SetAttrString(_pymessage, "width", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // z_min
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->z_min);
    {
      int rc = PyObject_SetAttrString(_pymessage, "z_min", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // z_max
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->z_max);
    {
      int rc = PyObject_SetAttrString(_pymessage, "z_max", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // blocking
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->blocking ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "blocking", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
