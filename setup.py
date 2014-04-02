import os
from setuptools import setup, find_packages


NNdir = os.path.dirname(os.path.realpath(__file__))

if not os.getenv('smartNN_DATA_PATH'):
    os.environ['smartNN_DATA_PATH'] = NNdir + '/data'

if not os.getenv('smartNN_SAVE_PATH'):
    os.environ['smartNN_SAVE_PATH'] = NNdir + '/save'

os.environ['PYTHONPATH'] += NNdir + '/smartNN'

print(os.environ['smartNN_DATA_PATH'])
print(os.environ['smartNN_SAVE_PATH'])
print(os.environ['PYTHONPATH'])

# setup(
#     name='smartNN',
#     version='0.1',
#     packages=find_packages(),
#     description='A machine learning library build on top of Theano.',
#     license='BSD 3-clause license',
#     install_requires=['numpy>=1.5', 'theano'],
#     package_data={
#         '': ['*.txt', '*.rst', '*.cu', '*.cuh', '*.h'],
#     },
# )
