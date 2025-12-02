# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool

cimport cython

cdef extern from "sq_avx.hpp":
    vector[vector[double]] calculate_structure_factor_cpp "calculate_structure_factor" (vector[vector[double]], int, double, long unsigned int, long unsigned int, vector[bool])

# Define a Python wrapper function    
@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_structure_factor(vector[vector[double]] particle_positions, int order, double box_len, int orientations_per_wavevector, int subsample_wavevectors, vector[bool] axis_mask=[True,True,True]):
    """
    Calculate S(q) for the provided particle positions.

    Parameters
    ----------
    particle_positions : vector[vector[double]]
        Folded particle coordinates.
    order : int
        Maximum wavevector order.
    box_len : float
        Simulation box length (assuming cubic box).
    orientations_per_wavevector : int
        Number of (i,j,k) orientations sampled for each |q|.
    subsample_wavevectors : int
        Number of logarithmically distributed |q| magnitudes to evaluate.
    axis_mask : vector[bool], optional
        Axis selector for directional structure factors.
    """
    return calculate_structure_factor_cpp(particle_positions, order, box_len, orientations_per_wavevector, subsample_wavevectors, axis_mask)

