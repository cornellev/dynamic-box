from setuptools import find_packages
from setuptools import setup

setup(
    name='cev_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('cev_msgs', 'cev_msgs.*')),
)
