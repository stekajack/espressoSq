#include <vector>

/**
 * @brief Calculates the structure factor of a particle system.
 *
 * This function computes the structure factor for a system of particles given their positions
 * within a defined simulation box. The structure factor is a measure of the arrangement of
 * particles in a system and can be used to analyze the degree of order or disorder.
 *
 * @param particle_positions An array where each subvector contains the position of a particle in 3D space (x, y, z).
 * @param order The order of the structure factor to compute. Typically, this is related to the max possible resolution discernable by the scattering.
 * @param box_len The length of the simulation box along each dimension. Assumes a cubic box.
 * @param M The number of wavevectors to use in scattering in the discretized wavevector space.
 * @param N The number of qs to calculate.
 * @param nthreads Number of OpenMP threads to use per call (>=1).
 *
 * @return A 2D array containing (q, S(q)).
 *
 * @note The function assumes that the particle positions are in folded coordinates.
 *
 * @warning Make sure you understand what you are doing, the code does not check for logical mistakes.
 */
std::vector<std::vector<double>> calculate_structure_factor(const std::vector<std::vector<double>> &particle_positions, long unsigned int order, double box_len, long unsigned int M, long unsigned int N, int nthreads = 1);
