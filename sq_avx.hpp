#ifndef SQ_PORT_AVX
#define SQ_PORT_AVX

#include <vector>

// Function prototypes for approximate trigonometric functions
std::vector<std::vector<double>> calculate_structure_factor(const std::vector<std::vector<double>> &particle_positions, long unsigned int order, double box_len, long unsigned int M, long unsigned int N);

std::vector<std::vector<double>> calculate_structure_factor_avx(const std::vector<std::vector<double>> &particle_positions, long unsigned int order, double box_len, long unsigned int M, long unsigned int N);

#endif // SQ_PORT_AVX