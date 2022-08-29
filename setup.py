from distutils import sysconfig
from setuptools import setup, Extension, find_packages
import os
import sys
import setuptools
from copy import deepcopy

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pyrft',
    install_requires=[
        'numpy',
        'sanssouci',
        'matplotlib',
        'sklearn',
        'scipy',
        'scikit-image'
    ],
    version = '0.0.1',
    license='MIT',
    author='Samuel DAVENPORT',
    download_url='https://github.com/sjdavenport/pyrft/',
    author_email='samuel.davenport@math.univ-toulouse.fr',
    url='https://github.com/sjdavenport/pyrft/',
    long_description=long_description,
    description='Python Toolbox of Functions for Analysing Random fields',
    keywords='Random field Theory, fMRI, Post-hoc inference',
    packages=find_packages(),
    python_requires='>=3',
)
