from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os
import sys

extensions = [
    Extension(
        "espressoSq",
        sources=[
            "espressoSq.pyx",  # Your Cython file
            "approx_trig_avx2.cpp",
            "sq_avx.cpp"
        ],
        language="c++",
        extra_compile_args=["-std=c++14","-g",
            "-mavx2",
            "-O3",
            "-march=native"],  # Adjust based on your C++ standard
        extra_link_args=[],
    )
]

setup(
    name="espressoSq",  # Replace with the actual name of your module
    ext_modules=cythonize(extensions),
)