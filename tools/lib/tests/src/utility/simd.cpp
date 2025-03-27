#include <algorithm>
#include <catch2/catch_all.hpp>
#include <immintrin.h>
#include <voxcom/utility/simd.h>

TEST_CASE("_mm256_max_epu64_avx2", "[SIMD]")
{
    const auto values0 = GENERATE(take(10, chunk(4, random<uint64_t>(1, std::numeric_limits<uint64_t>::max()))));
    const auto values1 = GENERATE(take(10, chunk(4, random<uint64_t>(1, std::numeric_limits<uint64_t>::max()))));

    const auto values0_AVX2 = _mm256_loadu_si256((const __m256i*)values0.data());
    const auto values1_AVX2 = _mm256_loadu_si256((const __m256i*)values1.data());
    const auto result_AVX2 = _mm256_max_epu64_avx2(values0_AVX2, values1_AVX2);
    std::array<uint64_t, 4> result;
    _mm256_storeu_si256((__m256i*)result.data(), result_AVX2);

    for (uint32_t i = 0; i < 4; ++i) {
        const auto expected = std::max(values0[i], values1[i]);
        const auto got = result[i];
        CAPTURE(values0[i], values1[i], got, expected);
        REQUIRE(got == expected);
    }
}
