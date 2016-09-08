from distutils.core import setup
from setuptools import find_packages


setup(
    name='mozi',
    version='2.0.3',
    author=u'Wu Zhen Zhou',
    author_email='hyciswu@gmail.com',
    install_requires=['numpy>=1.7.1', 'scipy>=0.11',
                      'six>=1.9.0', 'scikit-learn>=0.17', 'pandas>=0.17',
                      'matplotlib>=1.5', 'Theano>=0.8'],
    url='https://github.com/hycis/Mozi',
    license='The MIT License (MIT), see LICENCE',
    description='Deep learning package based on theano for building all kinds of models',
    long_description=open('README.md').read(),
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True
)
