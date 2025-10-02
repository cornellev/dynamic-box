<<<<<<< HEAD
from setuptools import find_packages, setup, Extension
import pybind11
import os

package_name = 'my_rosbag_reader'

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

=======
from setuptools import find_packages, setup

package_name = 'my_rosbag_reader'

>>>>>>> 73852507092b44cc445d3e1af79b72b3fa076d0f
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
<<<<<<< HEAD
    ext_modules=ext_modules,
=======
>>>>>>> 73852507092b44cc445d3e1af79b72b3fa076d0f
    install_requires=['setuptools', 'rclpy', 'std_msgs', 'cev_msgs'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'listener = my_rosbag_reader.live:main',
            'rs_listener = my_rosbag_reader.surface:main'
        ],
    },
)