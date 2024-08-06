#include <immintrin.h> // AVX intrinsics
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "approx_trig_avx2.hpp"
#include <iostream>
#include <cassert>
#include <iomanip> // For formatting

void kernel_avx2(const double *particle_positions_x, const double *particle_positions_y, const double *particle_positions_z, int num_particles, double twoPI_L, const double *comb, __m256d &C_sum, __m256d &S_sum)
{
    assert(num_particles % 4 == 0);

    __m256d twoPI_L_vec = _mm256_set1_pd(twoPI_L);
    __m256d comb_x = _mm256_set1_pd(comb[0]);
    __m256d comb_y = _mm256_set1_pd(comb[1]);
    __m256d comb_z = _mm256_set1_pd(comb[2]);

    for (int i = 0; i < num_particles; i += 4)
    {
        __m256d pos_x = _mm256_load_pd(&particle_positions_x[i]);
        __m256d pos_y = _mm256_load_pd(&particle_positions_y[i]);
        __m256d pos_z = _mm256_load_pd(&particle_positions_z[i]);

        __m256d scalar_product = _mm256_mul_pd(comb_x, pos_x);
        scalar_product = _mm256_add_pd(scalar_product, _mm256_mul_pd(comb_y, pos_y));
        scalar_product = _mm256_add_pd(scalar_product, _mm256_mul_pd(comb_z, pos_z));

        scalar_product = _mm256_mul_pd(scalar_product, twoPI_L_vec);

        __m256d cos_val = cos_approx_avx2(scalar_product);
        __m256d sin_val = sin_approx_avx2(scalar_product);

        C_sum = _mm256_add_pd(C_sum, cos_val);
        S_sum = _mm256_add_pd(S_sum, sin_val);
    }
}

void printVector(const std::vector<double> &vec, const std::string &name)
{
    std::cout << name << ": ";
    for (const auto &value : vec)
    {
        std::cout << std::fixed << std::setprecision(6) << value << " "; // Set precision for readability
    }
    std::cout << std::endl;
}

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

std::vector<std::vector<double>> calculate_structure_factor_avx(const std::vector<std::vector<double>> &particle_positions, long unsigned int order, double box_len, long unsigned int M, long unsigned int N)
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

            __m256d C_sum_novec = _mm256_setzero_pd();
            __m256d S_sum_novec = _mm256_setzero_pd();

            kernel_avx2(particle_positions_x, particle_positions_y, particle_positions_z, effective_size, twoPI_L, comb_novec, C_sum_novec, S_sum_novec);
            alignas(64) double C_sum_store[4];
            alignas(64) double S_sum_store[4];
            _mm256_store_pd(C_sum_store, C_sum_novec);
            _mm256_store_pd(S_sum_store, S_sum_novec);
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

// int main()
// {
// // Define the parameters for the test
// const unsigned int num_particles = 9999;
// const unsigned int order = 100;
// const double box_len = 10.0;
// const unsigned int M = 100;
// const unsigned int N = 100;

// // Create random particle positions
// std::vector<std::vector<double>> particle_positions(num_particles, std::vector<double>(3));
// std::random_device rd;
// std::mt19937 gen(rd());
// std::uniform_real_distribution<> dis(0.0, box_len);

// for (auto &pos : particle_positions)
// {
//     pos[0] = dis(gen);
//     pos[1] = dis(gen);
//     pos[2] = dis(gen);
// }

// // Measure the execution time of the structure_factor function
// auto start = std::chrono::high_resolution_clock::now();

// std::vector<std::vector<double>> result = structure_factor_avx2(particle_positions, order, box_len, M, N);

//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;

//     std::cout << " structure_factor_avx2 Elapsed time: " << elapsed.count() << " seconds" << std::endl;
//     // if (result.size() >= 2)
//     // {
//     //     // printVector(result[0], "Wavevectors");
//     //     printVector(result[1], "Intensities");
//     // }
//     // else
//     // {
//     //     std::cout << "Error: Unexpected result format." << std::endl;
//     // }

//     start = std::chrono::high_resolution_clock::now();

//     result = structure_factor(particle_positions, order, box_len, M, N);

//     end = std::chrono::high_resolution_clock::now();
//     elapsed = end - start;

//     std::cout << " structure_factor Elapsed time: " << elapsed.count() << " seconds" << std::endl;
//     // if (result.size() >= 2)
//     // {
//     //     // printVector(result[0], "Wavevectors");
//     //     printVector(result[1], "Intensities");
//     // }
//     // else
//     // {
//     //     std::cout << "Error: Unexpected result format." << std::endl;
//     // }

//     return 0;
// }