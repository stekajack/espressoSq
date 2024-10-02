#pragma once
#include "simd_intrinsics.hpp"

// Function prototypes for approximate trigonometric functions
simd_double_t cos_approx_simd(simd_double_t xx);
simd_double_t sin_approx_simd(simd_double_t xx);