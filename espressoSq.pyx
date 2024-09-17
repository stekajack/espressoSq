# distutils: language = c++
from libcpp.vector cimport vector
cimport cython

cdef extern from "sq_avx.hpp":
    vector[vector[double]] calculate_structure_factor_cpp "calculate_structure_factor" (vector[vector[double]], int, double, long unsigned int, long unsigned int)

# Define a Python wrapper function    
@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_structure_factor(vector[vector[double]] particle_positions, int order, double box_len,int M, int N):
    return calculate_structure_factor_cpp(particle_positions, order, box_len, M, N)
