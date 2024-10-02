/**
 * @file approx_trig_avx2.cpp
 * @brief AVX2 SIMD Intrinsics for Trigonometric Approximations and Modular Arithmetic.
 *
 * This file contains the implementation auxiliary functions implemented using AVX2 SIMD intrinsics
 * for high-performance trigonometric approximations (cosine and sine) and modular
 * arithmetic operations. The code leverages AVX2 instructions to perform operations
 * on four double-precision floating-point numbers in parallel, offering significant
 * performance improvements over scalar implementations.
 *
 * These functions are part of the espressoSq library, intended to be used for
 * optimizing mathematical computations in the context of structure factor calculations. The trigonometric functions
 * use Taylor series expansions to approximate sine and cosine, and the modular
 * arithmetic function provides a vectorized version of fmod.
 *
 * Architecture requirements:
 * - AVX2 (Advanced Vector Extensions 2) support is required to use these functions.
 *
 * @author Deniz Mostarac
 * @version 1.0
 * @date 03.06.2024
 */

#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include "simd_intrinsics.hpp"
/**
 * @brief Precomputed inverse factorial constants for use in Taylor series expansions.
 */
const simd_double_t FACTORIAL_INV_2 = simd_set1_pd(1.0 / 2.0);
const simd_double_t FACTORIAL_INV_4 = simd_set1_pd(1.0 / 24.0);
const simd_double_t FACTORIAL_INV_6 = simd_set1_pd(1.0 / 720.0);
const simd_double_t FACTORIAL_INV_8 = simd_set1_pd(1.0 / 40320.0);

const simd_double_t FACTORIAL_INV_3 = simd_set1_pd(1.0 / 6.0);
const simd_double_t FACTORIAL_INV_5 = simd_set1_pd(1.0 / 120.0);
const simd_double_t FACTORIAL_INV_7 = simd_set1_pd(1.0 / 5040.0);
const simd_double_t FACTORIAL_INV_9 = simd_set1_pd(1.0 / 362880.0);

const simd_double_t ONE = simd_set1_pd(1.0);
const simd_double_t ZERO = simd_set1_pd(0.0);
const simd_double_t NEG_ONE = simd_set1_pd(-1.0);
const simd_double_t PI = simd_set1_pd(M_PI);
const simd_double_t TWO_PI = simd_set1_pd(2 * M_PI);
const simd_double_t PI_HALF = simd_set1_pd(M_PI / 2);
const simd_double_t THREE_PI_HALF = simd_set1_pd(3 * M_PI / 2);

/**
 * @brief Computes the floating-point remainder of a division using AVX2.
 *
 * This function computes the remainder of dividing numer by denom using
 * AVX2 vectorized instructions, similar to the standard `fmod` function in C++.
 *
 * @param numer Numerator (AVX2 vector of doubles).
 * @param denom Denominator (AVX2 vector of doubles).
 * @return simd_double_t The floating-point remainder (numer % denom).
 */
simd_double_t avx2_fmod(simd_double_t numer, simd_double_t denom)
{
    // Compute quotient (numer / denom)
    simd_double_t quotient = simd_div_pd(numer, denom);

    // Truncate quotient towards zero
    simd_double_t truncated_quotient = simd_round_pd(quotient);

    // Compute product of truncated quotient and denominator
    simd_double_t product = simd_mul_pd(truncated_quotient, denom);

    // Compute remainder (numer - product)
    return simd_sub_pd(numer, product);
}

/**
 * @brief SIMD approximate cos() function compatible with AVX2.
 *
 * This function uses a 4th order Taylor series expansion to approximate the cosine function.
 * It operates on four double-precision values in parallel using AVX2 instructions.
 *
 * @param xx Input angle(s) in radians (as an AVX2 vector of doubles).
 * @return simd_double_t The cosine approximation of the input values.
 */
simd_double_t cos_approx_avx2(simd_double_t xx)
{
    simd_double_t x = avx2_fmod(xx, TWO_PI);
    simd_double_t mask_neg = simd_cmp_pd(x, simd_set1_pd(0.0));
    x = simd_blendv_pd(x, simd_add_pd(x, TWO_PI), mask_neg);
    simd_double_t fact = ONE;

    // Select condition for 0 <= x < PI/2
    simd_double_t mask_0 = simd_cmp_pd(x, PI_HALF);

    // Select condition for PI/2 <= x < PI
    simd_double_t mask_1 = simd_andnot_pd(mask_0, simd_cmp_pd(x, PI));

    // Select condition for PI <= x < 3*PI/2
    simd_double_t mask_2 = simd_andnot_pd(simd_or_pd(mask_0, mask_1), simd_cmp_pd(x, THREE_PI_HALF));

    // Select condition for 3*PI/2 <= x < 2*PI
    simd_double_t mask_3 = simd_andnot_pd(simd_or_pd(simd_or_pd(mask_0, mask_1), mask_2), simd_cmp_pd(x, TWO_PI));

    // x = M_PI - x for PI/2 <= x < PI
    simd_double_t x1 = simd_sub_pd(PI, x);
    x = simd_blendv_pd(x, x1, mask_1);

    // x = x - M_PI for PI <= x < 3*PI/2
    simd_double_t x2 = simd_sub_pd(x, PI);
    x = simd_blendv_pd(x, x2, mask_2);
    // x = 2*M_PI - x for 3*PI/2 <= x < 2*PI
    simd_double_t x3 = simd_sub_pd(TWO_PI, x);
    x = simd_blendv_pd(x, x3, mask_3);

    fact = simd_blendv_pd(fact, NEG_ONE, mask_1);
    fact = simd_blendv_pd(fact, NEG_ONE, mask_2);

    simd_double_t x_squared = simd_mul_pd(x, x);
    simd_double_t x_squared_squared = simd_mul_pd(x_squared, x_squared);
    simd_double_t x_squared_cubed = simd_mul_pd(x_squared_squared, x_squared);
    simd_double_t x_squared_quad = simd_mul_pd(x_squared_cubed, x_squared);

    simd_double_t term1 = simd_mul_pd(x_squared, FACTORIAL_INV_2);
    simd_double_t term2 = simd_mul_pd(x_squared_squared, FACTORIAL_INV_4);
    simd_double_t term3 = simd_mul_pd(x_squared_cubed, FACTORIAL_INV_6);
    simd_double_t term4 = simd_mul_pd(x_squared_quad, FACTORIAL_INV_8);

    simd_double_t result = simd_sub_pd(ONE, term1);
    result = simd_add_pd(result, term2);
    result = simd_sub_pd(result, term3);
    result = simd_add_pd(result, term4);
    result = simd_mul_pd(result, fact);

    return result;
}

/**
 * @brief SIMD approximate sin() function compatible with AVX2.
 *
 * This function uses a 4th order Taylor series expansion to approximate the cosine function.
 * It operates on four double-precision values in parallel using AVX2 instructions.
 *
 * @param xx Input angle(s) in radians (as an AVX2 vector of doubles).
 * @return simd_double_t The cosine approximation of the input values.
 */
simd_double_t sin_approx_avx2(simd_double_t xx)
{
    simd_double_t x = avx2_fmod(xx, TWO_PI);
    simd_double_t mask_neg = simd_cmp_pd(x, simd_set1_pd(0.0));
    x = simd_blendv_pd(x, simd_add_pd(x, TWO_PI), mask_neg);
    simd_double_t fact = ONE;

    // Select condition for 0 <= x < PI/2
    simd_double_t mask_0 = simd_cmp_pd(x, PI_HALF);

    // Select condition for PI/2 <= x < PI
    simd_double_t mask_1 = simd_andnot_pd(mask_0, simd_cmp_pd(x, PI));

    // Select condition for PI <= x < 3*PI/2
    simd_double_t mask_2 = simd_andnot_pd(simd_or_pd(mask_0, mask_1), simd_cmp_pd(x, THREE_PI_HALF));

    // Select condition for 3*PI/2 <= x < 2*PI
    simd_double_t mask_3 = simd_andnot_pd(simd_or_pd(simd_or_pd(mask_0, mask_1), mask_2), simd_cmp_pd(x, TWO_PI));

    // x = M_PI - x for PI/2 <= x < PI
    simd_double_t x1 = simd_sub_pd(PI, x);
    x = simd_blendv_pd(x, x1, mask_1);

    // x = x - M_PI for PI <= x < 3*PI/2
    simd_double_t x2 = simd_sub_pd(x, PI);
    x = simd_blendv_pd(x, x2, mask_2);

    // x = 2*M_PI - x for 3*PI/2 <= x < 2*PI
    simd_double_t x3 = simd_sub_pd(TWO_PI, x);
    x = simd_blendv_pd(x, x3, mask_3);

    fact = simd_blendv_pd(fact, NEG_ONE, mask_2);
    fact = simd_blendv_pd(fact, NEG_ONE, mask_3);

    simd_double_t x_squared = simd_mul_pd(x, x);
    simd_double_t x_pow3 = simd_mul_pd(x, x_squared);
    simd_double_t x_pow5 = simd_mul_pd(x_pow3, x_squared);
    simd_double_t x_pow7 = simd_mul_pd(x_pow5, x_squared);
    simd_double_t x_pow9 = simd_mul_pd(x_pow7, x_squared);

    simd_double_t term2 = simd_mul_pd(x_pow3, FACTORIAL_INV_3);
    simd_double_t term3 = simd_mul_pd(x_pow5, FACTORIAL_INV_5);
    simd_double_t term4 = simd_mul_pd(x_pow7, FACTORIAL_INV_7);
    simd_double_t term5 = simd_mul_pd(x_pow9, FACTORIAL_INV_9);

    simd_double_t result = simd_sub_pd(x, term2);
    result = simd_add_pd(result, term3);
    result = simd_sub_pd(result, term4);
    result = simd_add_pd(result, term5);
    result = simd_mul_pd(result, fact);

    return result;
}