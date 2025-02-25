from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths relative to the base directory
include_dirs = [
    os.path.join(base_dir, 'includes'),  # Directory containing header files
    numpy.get_include()  # Directory for numpy headers
]

library_dirs = [
    os.path.join(base_dir, 'build')  # Directory containing the compiled library
]

ext_modules = [
    Extension(
        name="sq_avx",
        sources=["espressoSq.pyx"],  # Cython source file
        include_dirs=include_dirs,  # Include directories
        libraries=["espressoSq"],  # Link against your C++ library
        library_dirs=library_dirs,  # Path to the library files
        language="c++",  # Indicate that this is a C++ extension
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="sq_avx",
    ext_modules=cythonize(ext_modules),
)