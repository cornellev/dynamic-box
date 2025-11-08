from setuptools import find_packages
from setuptools import setup

setup(
    name='cluster_node',
    version='0.0.0',
    packages=find_packages(
        include=('cluster_node', 'cluster_node.*')),
)
