import os
from setuptools import setup, find_packages

setup(
    name='mozi',
    version='0.1',
    packages=find_packages(),
    description='A machine learning library build on top of Theano.',
    license='MIT',
    install_requires=['numpy>=1.5', 'theano'],
    package_data={
        '': ['*.txt', '*.rst', '*.cu', '*.cuh', '*.h'],
    },
)
