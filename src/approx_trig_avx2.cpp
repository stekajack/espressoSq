#include <iostream>
#include <cmath>
#ifdef ARCH_X86
#include <immintrin.h>
#endif
#include <chrono>
#include <vector>

const __m256d FACTORIAL_INV_2 = _mm256_set1_pd(1.0 / 2.0);
const __m256d FACTORIAL_INV_4 = _mm256_set1_pd(1.0 / 24.0);
const __m256d FACTORIAL_INV_6 = _mm256_set1_pd(1.0 / 720.0);
const __m256d FACTORIAL_INV_8 = _mm256_set1_pd(1.0 / 40320.0);

const __m256d FACTORIAL_INV_3 = _mm256_set1_pd(1.0 / 6.0);
const __m256d FACTORIAL_INV_5 = _mm256_set1_pd(1.0 / 120.0);
const __m256d FACTORIAL_INV_7 = _mm256_set1_pd(1.0 / 5040.0);
const __m256d FACTORIAL_INV_9 = _mm256_set1_pd(1.0 / 362880.0);

const __m256d ONE = _mm256_set1_pd(1.0);
const __m256d ZERO = _mm256_set1_pd(0.0);
const __m256d NEG_ONE = _mm256_set1_pd(-1.0);
const __m256d PI = _mm256_set1_pd(M_PI);
const __m256d TWO_PI = _mm256_set1_pd(2 * M_PI);
const __m256d PI_HALF = _mm256_set1_pd(M_PI / 2);
const __m256d THREE_PI_HALF = _mm256_set1_pd(3 * M_PI / 2);

__m256d avx2_fmod(__m256d numer, __m256d denom)
{
    // Compute quotient (numer / denom)
    __m256d quotient = _mm256_div_pd(numer, denom);

    // Truncate quotient towards zero
    __m256d truncated_quotient = _mm256_round_pd(quotient, _MM_FROUND_TO_ZERO);

    // Compute product of truncated quotient and denominator
    __m256d product = _mm256_mul_pd(truncated_quotient, denom);

    // Compute remainder (numer - product)
    return _mm256_sub_pd(numer, product);
}

__m256d cos_approx_avx2(__m256d xx)
{
    __m256d x = avx2_fmod(xx, TWO_PI);
    __m256d mask_neg = _mm256_cmp_pd(x, _mm256_set1_pd(0.0), _CMP_LT_OQ);
    x = _mm256_blendv_pd(x, _mm256_add_pd(x, TWO_PI), mask_neg);
    __m256d fact = ONE;

    // Select condition for 0 <= x < PI/2
    __m256d mask_0 = _mm256_cmp_pd(x, PI_HALF, _CMP_LT_OQ);

    // Select condition for PI/2 <= x < PI
    __m256d mask_1 = _mm256_andnot_pd(mask_0, _mm256_cmp_pd(x, PI, _CMP_LT_OQ));

    // Select condition for PI <= x < 3*PI/2
    __m256d mask_2 = _mm256_andnot_pd(_mm256_or_pd(mask_0, mask_1), _mm256_cmp_pd(x, THREE_PI_HALF, _CMP_LT_OQ));

    // Select condition for 3*PI/2 <= x < 2*PI
    __m256d mask_3 = _mm256_andnot_pd(_mm256_or_pd(_mm256_or_pd(mask_0, mask_1), mask_2), _mm256_cmp_pd(x, TWO_PI, _CMP_LT_OQ));

    // x = M_PI - x for PI/2 <= x < PI
    __m256d x1 = _mm256_sub_pd(PI, x);
    x = _mm256_blendv_pd(x, x1, mask_1);

    // x = x - M_PI for PI <= x < 3*PI/2
    __m256d x2 = _mm256_sub_pd(x, PI);
    x = _mm256_blendv_pd(x, x2, mask_2);
    // x = 2*M_PI - x for 3*PI/2 <= x < 2*PI
    __m256d x3 = _mm256_sub_pd(TWO_PI, x);
    x = _mm256_blendv_pd(x, x3, mask_3);

    fact = _mm256_blendv_pd(fact, NEG_ONE, mask_1);
    fact = _mm256_blendv_pd(fact, NEG_ONE, mask_2);

    __m256d x_squared = _mm256_mul_pd(x, x);
    __m256d x_squared_squared = _mm256_mul_pd(x_squared, x_squared);
    __m256d x_squared_cubed = _mm256_mul_pd(x_squared_squared, x_squared);
    __m256d x_squared_quad = _mm256_mul_pd(x_squared_cubed, x_squared);

    __m256d term1 = _mm256_mul_pd(x_squared, FACTORIAL_INV_2);
    __m256d term2 = _mm256_mul_pd(x_squared_squared, FACTORIAL_INV_4);
    __m256d term3 = _mm256_mul_pd(x_squared_cubed, FACTORIAL_INV_6);
    __m256d term4 = _mm256_mul_pd(x_squared_quad, FACTORIAL_INV_8);

    __m256d result = _mm256_sub_pd(ONE, term1);
    result = _mm256_add_pd(result, term2);
    result = _mm256_sub_pd(result, term3);
    result = _mm256_add_pd(result, term4);
    result = _mm256_mul_pd(result, fact);

    return result;
}

__m256d sin_approx_avx2(__m256d xx)
{
    __m256d x = avx2_fmod(xx, TWO_PI);
    __m256d mask_neg = _mm256_cmp_pd(x, _mm256_set1_pd(0.0), _CMP_LT_OQ);
    x = _mm256_blendv_pd(x, _mm256_add_pd(x, TWO_PI), mask_neg);
    __m256d fact = ONE;

    // Select condition for 0 <= x < PI/2
    __m256d mask_0 = _mm256_cmp_pd(x, PI_HALF, _CMP_LT_OQ);

    // Select condition for PI/2 <= x < PI
    __m256d mask_1 = _mm256_andnot_pd(mask_0, _mm256_cmp_pd(x, PI, _CMP_LT_OQ));

    // Select condition for PI <= x < 3*PI/2
    __m256d mask_2 = _mm256_andnot_pd(_mm256_or_pd(mask_0, mask_1), _mm256_cmp_pd(x, THREE_PI_HALF, _CMP_LT_OQ));

    // Select condition for 3*PI/2 <= x < 2*PI
    __m256d mask_3 = _mm256_andnot_pd(_mm256_or_pd(_mm256_or_pd(mask_0, mask_1), mask_2), _mm256_cmp_pd(x, TWO_PI, _CMP_LT_OQ));

    // x = M_PI - x for PI/2 <= x < PI
    __m256d x1 = _mm256_sub_pd(PI, x);
    x = _mm256_blendv_pd(x, x1, mask_1);

    // x = x - M_PI for PI <= x < 3*PI/2
    __m256d x2 = _mm256_sub_pd(x, PI);
    x = _mm256_blendv_pd(x, x2, mask_2);

    // x = 2*M_PI - x for 3*PI/2 <= x < 2*PI
    __m256d x3 = _mm256_sub_pd(TWO_PI, x);
    x = _mm256_blendv_pd(x, x3, mask_3);

    fact = _mm256_blendv_pd(fact, NEG_ONE, mask_2);
    fact = _mm256_blendv_pd(fact, NEG_ONE, mask_3);

    __m256d x_squared = _mm256_mul_pd(x, x);
    __m256d x_pow3 = _mm256_mul_pd(x, x_squared);
    __m256d x_pow5 = _mm256_mul_pd(x_pow3, x_squared);
    __m256d x_pow7 = _mm256_mul_pd(x_pow5, x_squared);
    __m256d x_pow9 = _mm256_mul_pd(x_pow7, x_squared);

    __m256d term2 = _mm256_mul_pd(x_pow3, FACTORIAL_INV_3);
    __m256d term3 = _mm256_mul_pd(x_pow5, FACTORIAL_INV_5);
    __m256d term4 = _mm256_mul_pd(x_pow7, FACTORIAL_INV_7);
    __m256d term5 = _mm256_mul_pd(x_pow9, FACTORIAL_INV_9);

    __m256d result = _mm256_sub_pd(x, term2);
    result = _mm256_add_pd(result, term3);
    result = _mm256_sub_pd(result, term4);
    result = _mm256_add_pd(result, term5);
    result = _mm256_mul_pd(result, fact);

    return result;
}