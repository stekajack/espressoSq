#ifndef APPROX_TRIG_AVX2_H
#define APPROX_TRIG_AVX2_H

#include <immintrin.h>

// Function prototypes for approximate trigonometric functions
__m256d cos_approx_avx2(__m256d xx);
__m256d sin_approx_avx2(__m256d xx);

#endif // APPROX_TRIG_AVX2_H