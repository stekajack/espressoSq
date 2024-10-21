// Cross-platform code for scalar, AVX2, Neon

enum class Arch
{
    SCALAR,
    AVX2,
    NEON
};

template <Arch arch = Arch::SCALAR, typename T = double>
struct SIMD
{
    using value_type = double;
    static constexpr auto SIZE = 1u;
    static constexpr auto mul(double const a, double const b) { return a * b; }
    static constexpr auto add(double const a, double const b) { return a + b; }
    static constexpr auto load(double const *a) { return *a; }
    static constexpr auto set1(double const a) { return a; }
};

#ifdef __AVX2__
#include <immintrin.h>
template <>
struct SIMD<Arch::AVX2, double>
{
    using value_type = __m256d;
    static constexpr auto SIZE = sizeof(value_type) / sizeof(double);
    static constexpr auto mul = _mm256_mul_pd;
    static constexpr auto add = _mm256_add_pd;
    static constexpr auto load = _mm256_loadu_pd;
    static constexpr auto set1 = _mm256_set1_pd;
};
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
template <>
struct SIMD<Arch::NEON, double>
{
    using value_type = float64x2_t;
    static constexpr auto SIZE = sizeof(value_type) / sizeof(double);
    static constexpr auto mul = vmulq_f64;
    static constexpr auto add = vaddq_f64;
    static constexpr auto load = vld1q_f64;
    static constexpr auto set1 = vdupq_n_f64;
};
#endif

template <typename simd>
void kernel(double const *const particle_positions_x,
            double const *const particle_positions_y,
            double const *const particle_positions_z,
            unsigned int num_particles,
            double const *const comb,
            typename simd::value_type &C_sum)
{
    auto const comb_x = simd::set1(comb[0]);
    auto const comb_y = simd::set1(comb[1]);
    auto const comb_z = simd::set1(comb[2]);
    for (unsigned int i = 0u; i < num_particles; i += simd::SIZE)
    {
        auto const pos_x = simd::load(&particle_positions_x[i]);
        auto const pos_y = simd::load(&particle_positions_y[i]);
        auto const pos_z = simd::load(&particle_positions_z[i]);
        auto scalar_product = simd::mul(comb_x, pos_x);
        scalar_product = simd::add(scalar_product, simd::mul(comb_y, pos_y));
        scalar_product = simd::add(scalar_product, simd::mul(comb_z, pos_z));
        C_sum = simd::add(C_sum, scalar_product);
    }
}

#ifdef __AVX2__
auto constexpr kernel_avx2 = kernel<SIMD<Arch::AVX2, double>>;
// force creation of a symbol
template void kernel<SIMD<Arch::AVX2, double>>(
    double const *const,
    double const *const,
    double const *const,
    unsigned int,
    double const *const,
    typename SIMD<Arch::AVX2, double>::value_type &);
#endif
#ifdef __ARM_NEON
template void kernel<SIMD<Arch::NEON, double>>(
    double const *const,
    double const *const,
    double const *const,
    unsigned int,
    double const *const,
    typename SIMD<Arch::NEON, double>::value_type &);
#endif