#ifdef SUPPORTS_AVX2
#include "approx_trig_avx2.hpp"
#ifdef ARCH_X86
#include <immintrin.h> // AVX intrinsics
#endif
#endif
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cassert>

#ifdef SUPPORTS_AVX2
void kernel_avx2(const double *particle_positions_x, const double *particle_positions_y, const double *particle_positions_z, int num_particles, double twoPI_L, const double *comb, simd_double_t &C_sum, simd_double_t &S_sum)
{
    assert(num_particles % 4 == 0);

    simd_double_t twoPI_L_vec = simd_set1_pd(twoPI_L);
    simd_double_t comb_x = simd_set1_pd(comb[0]);
    simd_double_t comb_y = simd_set1_pd(comb[1]);
    simd_double_t comb_z = simd_set1_pd(comb[2]);

    for (int i = 0; i < num_particles; i += 4)
    {
        simd_double_t pos_x = simd_load_pd(&particle_positions_x[i]);
        simd_double_t pos_y = simd_load_pd(&particle_positions_y[i]);
        simd_double_t pos_z = simd_load_pd(&particle_positions_z[i]);

        simd_double_t scalar_product = simd_mul_pd(comb_x, pos_x);
        scalar_product = simd_add_pd(scalar_product, simd_mul_pd(comb_y, pos_y));
        scalar_product = simd_add_pd(scalar_product, simd_mul_pd(comb_z, pos_z));

        scalar_product = simd_mul_pd(scalar_product, twoPI_L_vec);

        simd_double_t cos_val = cos_approx_avx2(scalar_product);
        simd_double_t sin_val = sin_approx_avx2(scalar_product);

        C_sum = simd_add_pd(C_sum, cos_val);
        S_sum = simd_add_pd(S_sum, sin_val);
    }
}
#endif

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

std::vector<std::vector<int>> get_index_combinations(int n, int max_no_combinations, std::mt19937 rng_hnld)
{
    std::vector<std::vector<int>> combinations;

    int nsq = static_cast<int>(std::ceil(std::sqrt(n)));

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

#ifdef SUPPORTS_AVX2

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

    auto effective_size = (particle_positions.size() % 4 != 0) ? (particle_positions.size() / 4) * 4 : particle_positions.size();

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
            alignas(64) double C_sum_store[4];
            alignas(64) double S_sum_store[4];
            simd_store_pd(C_sum_store, C_sum_novec);
            simd_store_pd(S_sum_store, S_sum_novec);
            C_sum = C_sum_store[0] + C_sum_store[1] + C_sum_store[2] + C_sum_store[3];
            S_sum = S_sum_novec[0] + S_sum_novec[1] + S_sum_novec[2] + S_sum_novec[3];

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

#endif