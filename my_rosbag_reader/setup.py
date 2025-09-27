from setuptools import find_packages, setup

package_name = 'my_rosbag_reader'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'std_msgs', 'cev_msgs'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'listener = my_rosbag_reader.live:main',
            'rs_listener = my_rosbag_reader.surface:main'
        ],
    },
)