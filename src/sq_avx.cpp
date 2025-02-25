/**
 * @file structure_factor_simd.cpp
 * @brief This file contains functions for calculating the structure factor of particle positions in a simulation box.
 *
 * It includes both SIMD optimized functions using AVX2 for x86 and NEON for ARM, as well a fallback implementation. The primary goal is to compute the structure factor by accumulating cosine and sine sums over wavevector-particle interactions.
 *
 * Key Features:
 *  - SIMD Optimized particle loop: Uses vectorized operations to accelerate the computation of cosine and sine sums.
 *  - Logarithmic Distribution of Wavevector Indices: Ensures efficient coverage of wavevector space.
 *  - Index Combination Generator: Generates all valid combinations of `(i, j, k)` such that `i^2 + j^2 + k^2 = n`.
 *  - Fallback Non-SIMD Implementation: Provides a standard scalar version of the structure factor calculation for platforms without SIMD support.
 * Conditional Compilation:
 *  - If the macro `SUPPORTS_SIMD` is defined, the SIMD implementation is used, otherwise, the fallback scalar version is employed.
 *
 * @note This file relies on two external headers:
 *  - `simd_intrinsics.hpp`: Provides architecture-agnostic SIMD intrinsics (AVX2 for x86, NEON for ARM).
 *  - `approx_trig_avx2.hpp`: Provides SIMD-optimized auxilary functions  for use in the kernel.
 *
 * @see simd_intrinsics.hpp
 * @see approx_trig_avx2.hpp
 * *
 * @author Deniz Mostarac
 * @version 1.0
 * @date 03.06.2024
 */

#ifdef SUPPORTS_SIMD
#include "simd_intrinsics.hpp"
#include "approx_trig_avx2.hpp"
#endif
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cassert>
#include <omp.h> // Include OpenMP header

#ifdef SUPPORTS_SIMD
/**
 * @brief Particle loop kernel using SIMD.
 *
 * @param particle_positions_x X-coordinates of particle positions.
 * @param particle_positions_y Y-coordinates of particle positions.
 * @param particle_positions_z Z-coordinates of particle positions.
 * @param num_particles Number of particles.
 * @param twoPI_L Scaling factor 2π/L, where L is the box length.
 * @param comb Array containing the wave vector components (comb_x, comb_y, comb_z).
 * @param C_sum Accumulated sum of cosine values (output).
 * @param S_sum Accumulated sum of sine values (output).
 */
void kernel_avx2(const double *particle_positions_x, const double *particle_positions_y, const double *particle_positions_z, int num_particles, double twoPI_L, const double *comb, simd_double_t &C_sum, simd_double_t &S_sum)
{
    assert(num_particles % SIMD_VECTOR_WIDTH == 0);
    // Load constants into SIMD vectors
    simd_double_t twoPI_L_vec = simd_set1_pd(twoPI_L);
    simd_double_t comb_x = simd_set1_pd(comb[0]);
    simd_double_t comb_y = simd_set1_pd(comb[1]);
    simd_double_t comb_z = simd_set1_pd(comb[2]);
    // Loop through particle positions in chunks of SIMD_VECTOR_WIDTH
    for (int i = 0; i < num_particles; i += SIMD_VECTOR_WIDTH)
    {
        simd_double_t pos_x = simd_load_pd(&particle_positions_x[i]);
        simd_double_t pos_y = simd_load_pd(&particle_positions_y[i]);
        simd_double_t pos_z = simd_load_pd(&particle_positions_z[i]);
        // Calculate scalar product of wave vector and particle positions
        simd_double_t scalar_product = simd_mul_pd(comb_x, pos_x);
        scalar_product = simd_add_pd(scalar_product, simd_mul_pd(comb_y, pos_y));
        scalar_product = simd_add_pd(scalar_product, simd_mul_pd(comb_z, pos_z));
        // Multiply by scaling factor 2π/L
        scalar_product = simd_mul_pd(scalar_product, twoPI_L_vec);
        // Approximate cosine and sine of the scalar product
        simd_double_t cos_val = cos_approx_simd(scalar_product);
        simd_double_t sin_val = sin_approx_simd(scalar_product);
        // Accumulate results
        C_sum = simd_add_pd(C_sum, cos_val);
        S_sum = simd_add_pd(S_sum, sin_val);
    }
}
#endif

/**
 * @brief Generate a list of qs distributed logarithmically.
 *
 * @param order_sq Square of the maximum order of the wave vector.
 * @param N Number of desired indices.
 * @return std::vector<int> List of wavevector indices.
 */
std::vector<int> get_wavevector_indices_logdist(int order_sq, int N)
{
    std::vector<int> distances;
    std::vector<int> indices(order_sq);
    int set_length = order_sq / N;
    std::generate(indices.begin(), indices.end(), [n = 0]() mutable
                  { return n++; });

    for (int x : indices)
    {
        if (std::exp(x) < order_sq)
        {
            distances.push_back(std::round(std::exp(x)));
        }
        else
        {
            break;
        }
    }
    distances.push_back(std::round(order_sq));

    while (distances.size() < set_length)
    {
        int cnt = distances.size();
        std::vector<int> new_distances;
        for (size_t i = 0; i < distances.size() - 1; ++i)
        {
            if (cnt >= set_length)
            {
                break;
            }
            int midpoint = (distances[i] + distances[i + 1]) / 2;
            if (std::find(distances.begin(), distances.end(), midpoint) == distances.end())
            {
                new_distances.push_back(midpoint);
                cnt++;
            }
        }
        distances.insert(distances.end(), new_distances.begin(), new_distances.end());
        std::sort(distances.begin(), distances.end());
    }
    return distances;
}
/**
 * @brief Generate all possible index combinations of (i, j, k) such that i^2 + j^2 + k^2 = n.
 *
 * @param n The target sum of squares.
 * @param max_no_combinations Maximum number of combinations allowed.
 * @param rng_hnld Random number generator handle.
 * @return std::vector<std::vector<int>> List of index combinations.
 */
std::vector<std::vector<int>> get_index_combinations(int n, int max_no_combinations, std::mt19937 rng_hnld)
{
    std::vector<std::vector<int>> combinations;

    int nsq = static_cast<int>(std::ceil(std::sqrt(n)));
    // Generate all possible (i, j, k) combinations such that i^2 + j^2 + k^2 == n
    for (int i = 0; i <= nsq; i++)
    {
        for (int j = -nsq; j <= nsq; j++)
        {
            for (int k = -nsq; k <= nsq; k++)
            {
                if (i * i + j * j + k * k == n)
                {
                    combinations.push_back({i, j, k});
                }
            }
        }
    }

    // Check if the number of combinations exceeds M
    if (combinations.size() > max_no_combinations)
    {
        // Randomly select a subset of combinations
        std::shuffle(combinations.begin(), combinations.end(), rng_hnld);
        combinations.resize(max_no_combinations);
    }
    return combinations;
}

#ifdef SUPPORTS_SIMD
/**
 * @brief Compute the structure factor for a set of particle positions (SIMD version)
 *
 * @param particle_positions Vector of particle positions.
 * @param order Maximum order of wave vector.
 * @param box_len Length of the simulation box.
 * @param M Maximum number of combinations.
 * @param N Number of wave vector indices.
 * @return std::vector<std::vector<double>> Wavevectors and their corresponding intensities.
 */
std::vector<std::vector<double>> calculate_structure_factor(const std::vector<std::vector<double>> &particle_positions, long unsigned int order, double box_len, long unsigned int M, long unsigned int N)
{
    // Constants for calculations
    double C_sum = 0.0, S_sum = 0.0;
    long unsigned int order_sq = order * order;
    auto const twoPI_L = 2.0 * M_PI / box_len;
    std::vector<double> ff(2 * order_sq + 1);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> indices = get_wavevector_indices_logdist(order_sq, N);

    auto effective_size = (particle_positions.size() % SIMD_VECTOR_WIDTH != 0) ? (particle_positions.size() / SIMD_VECTOR_WIDTH) * SIMD_VECTOR_WIDTH : particle_positions.size();

    // Define the aligned array
    alignas(64) double particle_positions_x[effective_size];
    alignas(64) double particle_positions_y[effective_size];
    alignas(64) double particle_positions_z[effective_size];

    // Copy data from particle_positions to aligned C-style array
    for (size_t i = 0; i < effective_size; ++i)
    {
        particle_positions_x[i] = particle_positions[i][0];
        particle_positions_y[i] = particle_positions[i][1];
        particle_positions_z[i] = particle_positions[i][2];
    }
    int nthreads = std::max(omp_get_num_procs() - 2, 1);
    #pragma omp parallel for num_threads(nthreads) private(C_sum, S_sum)    
    for (int n : indices)
    {
        // Generate all combinations of i, j, k corresponding to n
        auto combinations = get_index_combinations(n, M, gen);
        for (const auto &comb : combinations)
        {
            alignas(64) double comb_novec[3] = {static_cast<double>(comb[0]), static_cast<double>(comb[1]), static_cast<double>(comb[2])};

            simd_double_t C_sum_novec = simd_setzero_pd();
            simd_double_t S_sum_novec = simd_setzero_pd();

            kernel_avx2(particle_positions_x, particle_positions_y, particle_positions_z, effective_size, twoPI_L, comb_novec, C_sum_novec, S_sum_novec);
            alignas(64) double C_sum_store[SIMD_VECTOR_WIDTH];
            alignas(64) double S_sum_store[SIMD_VECTOR_WIDTH];
            simd_store_pd(C_sum_store, C_sum_novec);
            simd_store_pd(S_sum_store, S_sum_novec);
            C_sum = C_sum_store[0] + C_sum_store[1] + C_sum_store[2] + C_sum_store[3];
            S_sum = S_sum_novec[0] + S_sum_novec[1] + S_sum_novec[2] + S_sum_novec[3];
            // fallback for parts that dont fit (if part_number%SIMD_VECTOR_WIDTH!=0)
            for (int i = effective_size; i < particle_positions.size(); i++)
            {
                const auto &pos = particle_positions[i];
                double scalar_product = comb[0] * pos[0] + comb[1] * pos[1] + comb[2] * pos[2];
                auto const qr = twoPI_L * scalar_product;
                C_sum += cos(qr);
                S_sum += sin(qr);
            }

            ff[2 * n - 2] += C_sum * C_sum + S_sum * S_sum;
            ff[2 * n - 1]++;
        }
    }
    //  Normalize the structure factor
    long n_particles = particle_positions.size();
    int length = 0;
    for (std::size_t qi = 0; qi < order_sq; qi++)
    {
        if (ff[2 * qi + 1] != 0)
        {
            ff[2 * qi] /= static_cast<double>(n_particles) * ff[2 * qi + 1];
            length++;
        }
    }

    std::vector<double> wavevectors(length);
    std::vector<double> intensities(length);

    int cnt = 0;
    for (std::size_t i = 0; i < order_sq; i++)
    {
        if (ff[2 * i + 1] != 0)
        {
            wavevectors[cnt] = twoPI_L * sqrt(static_cast<long>(i + 1));
            intensities[cnt] = ff[2 * i];
            cnt++;
        }
    }
    return {std::move(wavevectors), std::move(intensities)};
}

#else
/**
 * @brief Fallback function for calculating structure factor without SIMD support.
 *
 * @param particle_positions Vector of particle positions.
 * @param order Maximum order of wave vector.
 * @param box_len Length of the simulation box.
 * @param M Maximum number of combinations.
 * @param N Number of wave vector indices.
 * @return std::vector<std::vector<double>> Wavevectors and their corresponding intensities.
 */
std::vector<std::vector<double>> calculate_structure_factor(const std::vector<std::vector<double>> &particle_positions, long unsigned int order, double box_len, long unsigned int M, long unsigned int N)
{
    // Constants for calculations
    double C_sum = 0.0, S_sum = 0.0;
    long unsigned int order_sq = order * order;
    auto const twoPI_L = 2.0 * M_PI / box_len;
    std::vector<double> ff(2 * order_sq + 1);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> indices = get_wavevector_indices_logdist(order_sq, N);
    int nthreads = std::max(omp_get_num_procs() - 2, 1);
    #pragma omp parallel for num_threads(nthreads) private(C_sum, S_sum)
    for (int n : indices)
    {
        // Generate all combinations of i, j, k corresponding to n
        auto combinations = get_index_combinations(n, M, gen);

        for (const auto &comb : combinations)
        {
            C_sum = 0.0;
            S_sum = 0.0;
            for (const auto &pos : particle_positions)
            {
                double scalar_product = comb[0] * pos[0] + comb[1] * pos[1] + comb[2] * pos[2];
                auto const qr = twoPI_L * scalar_product;
                C_sum += cos(qr);
                S_sum += sin(qr);
            }
            ff[2 * n - 2] += C_sum * C_sum + S_sum * S_sum;
            ff[2 * n - 1]++;
        }
    }
    //  Normalize the structure factor
    long n_particles = particle_positions.size();

    int length = 0;
    for (std::size_t qi = 0; qi < order_sq; qi++)
    {
        if (ff[2 * qi + 1] != 0)
        {
            ff[2 * qi] /= static_cast<double>(n_particles) * ff[2 * qi + 1];
            length++;
        }
    }
    // Prepare wavevector and intensity output
    std::vector<double> wavevectors(length);
    std::vector<double> intensities(length);

    int cnt = 0;
    for (std::size_t i = 0; i < order_sq; i++)
    {
        if (ff[2 * i + 1] != 0)
        {
            wavevectors[cnt] = twoPI_L * sqrt(static_cast<long>(i + 1));
            intensities[cnt] = ff[2 * i];
            cnt++;
        }
    }
    return {std::move(wavevectors), std::move(intensities)};
}

#endif