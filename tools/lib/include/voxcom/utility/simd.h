#pragma once
#include <immintrin.h>

// https://stackoverflow.com/questions/54394350/simd-implement-mm256-max-epu64-and-mm256-min-epu64
inline __m256i _mm256_blendv_epi64_avx2(__m256i a, __m256i b, __m256i mask)
{
    return _mm256_castpd_si256(_mm256_blendv_pd(
        _mm256_castsi256_pd(a), _mm256_castsi256_pd(b), _mm256_castsi256_pd(mask)));
}
inline __m256i _mm256_max_epu64_avx2(__m256i a, __m256i b)
{
    __m256i opposite_sign = _mm256_xor_si256(a, b);
    __m256i mask = _mm256_cmpgt_epi64(a, b);
    return _mm256_blendv_epi64_avx2(b, a, _mm256_xor_si256(mask, opposite_sign));
}

#if ENABLE_AVX512
#define _my_mm256_max_epu64(a, b) _mm256_max_epu64(a, b)
#else
#define _my_mm256_max_epu64(a, b) _mm256_max_epu64_avx2(a, b)
#endif