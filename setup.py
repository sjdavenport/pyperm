from distutils import sysconfig
from setuptools import setup, Extension, find_packages
import os
import sys
import setuptools
from copy import deepcopy

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pyperm',
    install_requires=[
        'numpy',
        'sanssouci',
        'matplotlib',
        'sklearn',
        'scipy',
        'scikit-image',
        'nilearn'
    ],
    version = '0.0.2',
    license='MIT',
    author='Samuel DAVENPORT',
    download_url='https://github.com/sjdavenport/pyperm/',
    author_email='samuel.davenport@math.univ-toulouse.fr',
    url='https://github.com/sjdavenport/pyperm/',
    long_description=long_description,
    description='Python Toolbox of Functions for perofrming resampling in large scale datasets',
    keywords='Permutation, Bootstrap, fMRI, Post-hoc inference',
    packages=find_packages(),
    python_requires='>=3',
)
