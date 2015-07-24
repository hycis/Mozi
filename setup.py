import os
from setuptools import setup, find_packages

NNdir = os.path.dirname(os.path.realpath(__file__))

if not os.getenv('MOZI_DATA_PATH'):
    os.environ['MOZI_DATA_PATH'] = NNdir + '/data'

if not os.getenv('MOZI_SAVE_PATH'):
    os.environ['MOZI_SAVE_PATH'] = NNdir + '/save'

if not os.getenv('MOZI_DATABASE_PATH'):
    os.environ['MOZI_DATABASE_PATH'] = NNdir + '/database'

os.environ['PYTHONPATH'] += NNdir + '/mozi'

print(os.environ['MOZI_DATA_PATH'])
print(os.environ['MOZI_SAVE_PATH'])
print(os.environ['MOZI_DATABASE_PATH'])

setup(
    name='mozi',
    version='0.1',
    packages=find_packages(),
    description='A machine learning library build on top of Theano.',
    license='BSD',
    install_requires=['numpy>=1.5', 'theano', 'sqlite3>=2.6.0'],
    package_data={
        '': ['*.txt', '*.rst', '*.cu', '*.cuh', '*.h'],
    },
)
