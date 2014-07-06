import os
from setuptools import setup, find_packages

NNdir = os.path.dirname(os.path.realpath(__file__))

if not os.getenv('PYNET_DATA_PATH'):
    os.environ['PYNET_DATA_PATH'] = NNdir + '/data'

if not os.getenv('PYNET_SAVE_PATH'):
    os.environ['PYNET_SAVE_PATH'] = NNdir + '/save'

os.environ['PYTHONPATH'] += NNdir + '/pynet'

print(os.environ['PYNET_DATA_PATH'])
print(os.environ['PYNET_SAVE_PATH'])
print(os.environ['PYTHONPATH'])

setup(
    name='pynet',
    version='0.1',
    packages=find_packages(),
    description='A machine learning library build on top of Theano.',
    license='Apache License',
    install_requires=['numpy>=1.5', 'theano'],
    package_data={
        '': ['*.txt', '*.rst', '*.cu', '*.cuh', '*.h'],
    },
)
