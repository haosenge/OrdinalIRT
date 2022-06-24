#cython: language_level=3
#cython: profile=True

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["truncnormal.pyx","alpha_sampler.pyx"]),
    include_dirs=[numpy.get_include()]
)