#pragma once

#ifdef ARCH_X86
#include <immintrin.h>

// SIMD type for 256-bit AVX2 vectors (4 doubles)
typedef __m256d simd_double_t;
typedef __m256i simd_mask_t; // Integer mask type for AVX2 (256-bit)

// SIMD vector width (number of doubles in a vector)
#define SIMD_VECTOR_WIDTH 4

// AVX2 intrinsics
#define simd_mul_pd _mm256_mul_pd                         // Multiply two 256-bit SIMD vectors
#define simd_add_pd _mm256_add_pd                         // Add two 256-bit SIMD vectors
#define simd_sub_pd _mm256_sub_pd                         // Subtract two 256-bit SIMD vectors
#define simd_set1_pd _mm256_set1_pd                       // Broadcast a scalar to all elements of a 256-bit SIMD vector
#define simd_cmp_pd(x, y) _mm256_cmp_pd(x, y, _CMP_LT_OQ) // Compare less than (ordered)
#define simd_blendv_pd _mm256_blendv_pd                   // Conditional blend of two vectors based on a mask
#define simd_load_pd _mm256_load_pd                       // Load 256-bit SIMD vector from memory
#define simd_store_pd _mm256_storeu_pd                    // Store 256-bit SIMD vector to memory
#define simd_div_pd _mm256_div_pd                         // Divide two 256-bit SIMD vectors
#define simd_setzero_pd _mm256_setzero_pd
#define simd_store_pd _mm256_store_pd

// Unified truncation logic for rounding towards zero
inline __m256d simd_round_pd_to_zero(__m256d v)
{
    return _mm256_cvtepi64_pd(_mm256_cvttpd_epi64(v));
}
#define simd_round_pd simd_round_pd_to_zero // Alias for rounding/truncating towards zero

#define simd_andnot_pd _mm256_andnot_pd // Bitwise AND NOT for 256-bit SIMD vectors
#define simd_or_pd _mm256_or_pd         // Bitwise OR for 256-bit SIMD vectors

// Helpers for reinterpretation between float and integer types
#define simd_reinterpret_f64_to_mask(v) _mm256_castpd_si256(v) // Reinterpret __m256d as __m256i
#define simd_reinterpret_mask_to_f64(v) _mm256_castsi256_pd(v) // Reinterpret __m256i as __m256d

#elif defined(ARCH_ARM)
#include <arm_neon.h>

// SIMD type for 128-bit NEON vectors (2 doubles)
typedef float64x2_t simd_double_t;
typedef uint64x2_t simd_mask_t; // Integer mask type for NEON (128-bit)

// SIMD vector width (number of doubles in a vector)
#define SIMD_VECTOR_WIDTH 2

// NEON intrinsics
#define simd_mul_pd vmulq_f64                                                       // Multiply two 128-bit SIMD vectors
#define simd_add_pd vaddq_f64                                                       // Add two 128-bit SIMD vectors
#define simd_sub_pd vsubq_f64                                                       // Subtract two 128-bit SIMD vectors
#define simd_set1_pd vdupq_n_f64                                                    // Broadcast a scalar to all elements of a 128-bit SIMD vector
#define simd_cmp_pd(x, y) vreinterpretq_f64_u64(vcltq_f64(x, y))                    // Compare less than (ordered) and reinterpret as float64x2_t
#define simd_blendv_pd(mask, v1, v2) vbslq_f64(vreinterpretq_u64_f64(mask), v1, v2) // Conditional blend using integer mask
#define simd_load_pd vld1q_f64                                                      // Load 128-bit SIMD vector from memory
#define simd_store_pd vst1q_f64                                                     // Store 128-bit SIMD vector to memory
#define simd_div_pd vdivq_f64                                                       // Divide two 128-bit SIMD vectors

// Unified truncation logic for rounding towards zero
inline float64x2_t simd_round_pd_to_zero(float64x2_t v)
{
    return vcvtq_f64_s64(vcvtq_s64_f64(v));
}
#define simd_round_pd simd_round_pd_to_zero                                         // Alias for rounding/truncating towards zero

#define simd_andnot_pd(v1, v2) vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(v1), vreinterpretq_u64_f64(v2))) // Bitwise AND NOT for 128-bit SIMD vectors
#define simd_or_pd(v1, v2) vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(v1), vreinterpretq_u64_f64(v2)))     // Bitwise OR for 128-bit SIMD vectors

// Helpers for reinterpretation between float and integer types
#define simd_reinterpret_f64_to_mask(v) vreinterpretq_u64_f64(v)                                                      // Reinterpret float64x2_t as uint64x2_t
#define simd_reinterpret_mask_to_f64(v) vreinterpretq_f64_u64(v)                                                      // Reinterpret uint64x2_t as float64x2_t
#define simd_setzero_pd vdupq_n_f64(0.0)
#define simd_store_pd vst1q_f64
#endif