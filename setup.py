from setuptools import setup, Extension
from Cython.Build import cythonize
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler
from distutils.errors import CompileError, LinkError
import numpy
import os
import tempfile
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths relative to the base directory
include_dirs = [
    os.path.join(base_dir, 'includes'),  # Directory containing header files
    numpy.get_include()  # Directory for numpy headers
]

library_dirs = [
    os.path.join(base_dir, 'build')  # Directory containing the compiled library
]


def _detect_openmp_flag():
    """Return the compiler flag for OpenMP if supported, else None."""
    compiler = new_compiler()
    customize_compiler(compiler)
    flag = "/openmp" if compiler.compiler_type == "msvc" else "-fopenmp"
    tmpdir = tempfile.mkdtemp()
    try:
        test_source = os.path.join(tmpdir, "check_openmp.c")
        with open(test_source, "w") as f:
            f.write("#include <omp.h>\nint main() { return 0; }\n")
        objects = compiler.compile([test_source], extra_postargs=[flag])
        compiler.link_executable(objects, os.path.join(tmpdir, "check_openmp"), extra_postargs=[flag])
        return flag
    except (CompileError, LinkError):
        return None
    finally:
        shutil.rmtree(tmpdir)


openmp_flag = _detect_openmp_flag()
extra_compile_args = [openmp_flag] if openmp_flag else []
extra_link_args = [openmp_flag] if openmp_flag else []

if not openmp_flag:
    print("Warning: OpenMP runtime not detected; building without OpenMP flags")

ext_modules = [
    Extension(
        name="sq_avx",
        sources=["espressoSq.pyx"],  # Cython source file
        include_dirs=include_dirs,  # Include directories
        libraries=["espressoSq"],  # Link against your C++ library
        library_dirs=library_dirs,  # Path to the library files
        language="c++",  # Indicate that this is a C++ extension
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="sq_avx",
    ext_modules=cythonize(ext_modules),
)
