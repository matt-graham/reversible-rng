"""
Set up script for Reversible Random Number Generator (revrng) package.

Builds Cython module from .pyx / .c files.
"""

__authors__ = 'Matt Graham'
__license__ = 'MIT'

from distutils.core import setup
from distutils.extension import Extension
import os
from Cython.Build import cythonize

ext_modules = [
    Extension('revrng.numpy_wrapper',
              [os.path.join('revrng', file_name) for file_name
               in ['numpy_wrapper.pyx', 'revrand.c']])
]

ext_modules = cythonize(ext_modules)

setup(
    name='revrng',
    description='Reversible Mersenne-Twister random number generator',
    author='Matt Graham',
    ext_modules=ext_modules,
    packages=['revrng']
)
