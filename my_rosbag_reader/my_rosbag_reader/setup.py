from setuptools import setup, Extension
import pybind11
import os

ext_modules = [
    Extension(
        "cluster_cpp",
        ["cluster.cpp"],
        include_dirs=[
            pybind11.get_include(), 
            pybind11.get_include(user=True),
            "/usr/include/eigen3",
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]

setup(
    name="cluster_cpp",
    version="0.0.1",
    ext_modules=ext_modules,
)