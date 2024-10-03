#pragma once

#ifdef SUPPORTS_AVX2
#include <immintrin.h>
// SIMD vector width (number of doubles in a vector)
#define SIMD_VECTOR_WIDTH 4
// Define 256-bit SIMD type for x86 (AVX2)
typedef __m256d simd_double_t;

// Alias AVX2 intrinsics for x86
#define simd_mul_pd _mm256_mul_pd
#define simd_add_pd _mm256_add_pd
#define simd_sub_pd _mm256_sub_pd
#define simd_set1_pd _mm256_set1_pd
#define simd_cmp_pd(x, y) _mm256_cmp_pd(x, y, _CMP_LT_OQ)
#define simd_blendv_pd _mm256_blendv_pd
#define simd_load_pd _mm256_loadu_pd
#define simd_div_pd _mm256_div_pd
// Alias for rounding/truncating towards zero
#define simd_round_pd(x) _mm256_round_pd(x, _MM_FROUND_TO_ZERO)
#define simd_andnot_pd _mm256_andnot_pd
#define simd_or_pd _mm256_or_pd
#define simd_setzero_pd _mm256_setzero_pd
#define simd_store_pd _mm256_store_pd

#elif defined(SUPPORTS_NEON)
#include <arm_neon.h>

// SIMD vector width (number of doubles in a vector)
#define SIMD_VECTOR_WIDTH 2
// Define 128-bit SIMD type for ARM (NEON)
typedef float64x2_t simd_double_t;

// Alias NEON intrinsics for ARM
#define simd_mul_pd vmulq_f64    // NEON multiplication for two 128-bit vectors (float64x2_t)
#define simd_add_pd vaddq_f64    // NEON addition for two 128-bit vectors (float64x2_t)
#define simd_sub_pd vsubq_f64    // NEON subtraction for two 128-bit vectors (float64x2_t)
#define simd_set1_pd vdupq_n_f64 // NEON broadcast a scalar to a 128-bit vector
#define simd_cmp_pd(x, y) vreinterpretq_f64_u64(vcltq_f64(x, y))
#define simd_blendv_pd(mask, v1, v2) vbslq_f64(vreinterpretq_u64_f64(mask), v1, v2)
#define simd_load_pd vld1q_f64  // NEON load 128-bit vector (2 doubles)
#define simd_store_pd vst1q_f64 // NEON store 128-bit vector (2 doubles)
#define simd_div_pd vdivq_f64   // NEON division for two 128-bit vectors (float64x2_t)
// Alias for rounding/truncating towards zero
#define simd_round_pd(x) vcvtq_f64_s64(vcvtq_s64_f64(x))
#define simd_andnot_pd(v1, v2) vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(v1), vreinterpretq_u64_f64(v2)))
#define simd_or_pd(v1, v2) vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(v1), vreinterpretq_u64_f64(v2)))
#define simd_setzero_pd vdupq_n_f64(0.0)

#endif