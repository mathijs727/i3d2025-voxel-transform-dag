#include "voxcom/core/cuckoo_hash_table.h"
#include "voxcom/core/radix_sort.h"
#include "voxcom/utility/hash.h"
#include "voxcom/utility/maths.h"
#include "voxcom/utility/simd.h"
#include "voxcom/utility/size_of.h"
#include "voxcom/utility/template_magic.h"
#include "voxcom/voxel/large_sub_grid.h"
#include "voxcom/voxel/transform_dag.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <emmintrin.h>
#include <execution>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <tmmintrin.h>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <voxcom/utility/error_handling.h>
#include <voxcom/utility/fmt_glm.h>

#include "voxcom/utility/disable_all_warnings.h"
DISABLE_WARNINGS_PUSH()
#include <absl/container/flat_hash_map.h>
#include <fmt/chrono.h>
#include <robin_hood.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
DISABLE_WARNINGS_POP()

#define HASH_TYPE uint64_t
struct MyHash {
    HOST_DEVICE HASH_TYPE operator()(HASH_TYPE v) const
    {
        // Copy from PCG RNG
        // https://www.shadertoy.com/view/XlGcRh
        const uint32_t state = uint32_t(v) * 747796405lu + 2891336453lu;
        const uint32_t word = ((state >> ((state >> 28lu) + 4lu)) ^ state) * 277803737lu;
        return (word >> 22u) ^ word;
    }
};
#define HASH_FUNCTION std::identity
#define USE_CUCKOO_HASHING 0
#define SORT_BY_VOXEL_COUNT 1
#define ENABLE_ASSERTIONS 0

static constexpr auto execution_policy = std::execution::par_unseq;

namespace voxcom {

// 3D dense voxel grid with size known at compile time. Will be used to store hashes of neighboring voxels.
template <size_t Resolution>
using HashGrid = TypedLargeSubGrid<HASH_TYPE, Resolution>;
template <size_t Resolution>
using HashGrid2D = TypedLargeSubGrid2D<HASH_TYPE, Resolution>;

template <int Resolution>
static LargeSubGrid<Resolution> translateSubGrid(const LargeSubGrid<Resolution>& subGrid, const glm::ivec3& translation)
{
    const glm::ivec3 lowerBound { 0 };
    const glm::ivec3 upperBound { (int)Resolution };
    LargeSubGrid<Resolution> out {};
    for (int z = 0; z < Resolution; ++z) {
        for (int y = 0; y < Resolution; ++y) {
            for (int x = 0; x < Resolution; ++x) {
                const glm::ivec3 outVox { x, y, z };
                const glm::ivec3 inVox = outVox - translation;
                const bool outOfBounds = glm::any(glm::lessThan(inVox, lowerBound)) || glm::any(glm::greaterThanEqual(inVox, upperBound));
                if (!outOfBounds && subGrid.get(inVox))
                    out.set(outVox);
            }
        }
    }
    return out;
}
template <unsigned Resolution>
static LargeSubGrid<Resolution> applySymmetryToSubGrid(const LargeSubGrid<Resolution>& subGrid, const glm::bvec3& symmetry)
{
    LargeSubGrid<Resolution> out {};
    for (unsigned z = 0; z < Resolution; ++z) {
        for (unsigned y = 0; y < Resolution; ++y) {
            for (unsigned x = 0; x < Resolution; ++x) {
                const glm::uvec3 inVox { x, y, z };
                if (subGrid.get(inVox)) {
                    auto outVox = inVox;
                    if (symmetry.x)
                        outVox.x = Resolution - 1 - outVox.x;
                    if (symmetry.y)
                        outVox.y = Resolution - 1 - outVox.y;
                    if (symmetry.z)
                        outVox.z = Resolution - 1 - outVox.z;
                    out.set(outVox);
                }
            }
        }
    }
    return out;
}

template <unsigned Resolution>
static LargeSubGrid<Resolution> permuteAxisOfSubGrid(const LargeSubGrid<Resolution>& subGrid, const glm::u8vec3& permutation)
{
    LargeSubGrid<Resolution> out {};
    for (unsigned z = 0; z < Resolution; ++z) {
        for (unsigned y = 0; y < Resolution; ++y) {
            for (unsigned x = 0; x < Resolution; ++x) {
                const glm::uvec3 inVox { x, y, z };
                if (subGrid.get(inVox)) {
                    const glm::uvec3 outVox { inVox[permutation[0]], inVox[permutation[1]], inVox[permutation[2]] };
                    out.set(outVox);
                }
            }
        }
    }
    return out;
}

template <unsigned Resolution>
static void permuteAxisOfSubGrid(const HashGrid<Resolution>& inSubGrid, const glm::u8vec3& permutation, HashGrid<Resolution>& outSubGrid)
{
#if 1
    constexpr int Level = std::bit_width(Resolution - 1);
    constexpr uint32_t Mask = (1u << Level) - 1u;
    constexpr uint32_t ShiftX = 0;
    constexpr uint32_t ShiftY = Level;
    constexpr uint32_t ShiftZ = Level + Level;

    constexpr uint32_t Resolution3 = Resolution * Resolution * Resolution;
    for (uint32_t voxelIdx = 0; voxelIdx < Resolution3; ++voxelIdx) {
        const glm::uvec3 inVox {
            (voxelIdx >> ShiftX) & Mask,
            (voxelIdx >> ShiftY) & Mask,
            (voxelIdx >> ShiftZ) & Mask
        };
        const glm::uvec3 outVox { inVox[permutation[0]], inVox[permutation[1]], inVox[permutation[2]] };
        outSubGrid.set(outVox, inSubGrid.get(inVox));
    }
#else
    for (unsigned z = 0; z < Resolution; ++z) {
        for (unsigned y = 0; y < Resolution; ++y) {
            for (unsigned x = 0; x < Resolution; ++x) {
                const glm::uvec3 inVox { x, y, z };
                const glm::uvec3 outVox { inVox[permutation[0]], inVox[permutation[1]], inVox[permutation[2]] };
                outSubGrid.set(outVox, inSubGrid.get(inVox));
            }
        }
    }
#endif
}

#if ENABLE_AVX512
static inline __m512i hash_combine64(__m512i seed, __m512i key)
{
    const auto shiftLeft6 = _mm512_slli_epi64(seed, 6);
    const auto shiftRight2 = _mm512_srli_epi64(seed, 2);
    const auto combined = _mm512_add_epi64(_mm512_add_epi64(_mm512_add_epi64(key, _mm512_set1_epi64(0x9e3779b9)), shiftLeft6), shiftRight2);
    return _mm512_xor_si512(seed, combined);
}
#endif

static inline __m256i hash_combine64(__m256i seed, __m256i key)
{
    const auto shiftLeft6 = _mm256_slli_epi64(seed, 6);
    const auto shiftRight2 = _mm256_srli_epi64(seed, 2);
    const auto combined = _mm256_add_epi64(_mm256_add_epi64(_mm256_add_epi64(key, _mm256_set1_epi64x(0x9e3779b9)), shiftLeft6), shiftRight2);
    return _mm256_xor_si256(seed, combined);
}
static inline __m256i simd_negate64(__m256i lhs)
{
    return _mm256_xor_si256(lhs, _mm256_cmpeq_epi64(lhs, lhs)); // cmmpeq always returns 0xFFFFFFFF
}
static inline __m256i simd_neq_epi64(__m256i lhs, __m256i rhs)
{
    return simd_negate64(_mm256_cmpeq_epi64(lhs, rhs));
}

static inline __m256i hash_combine32(__m256i seed, __m256i key)
{
    const auto shiftLeft6 = _mm256_slli_epi32(seed, 6);
    const auto shiftRight2 = _mm256_srli_epi32(seed, 2);
    const auto combined = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(key, _mm256_set1_epi32(0x9e3779b9)), shiftLeft6), shiftRight2);
    return _mm256_xor_si256(seed, combined);
}
static inline __m256i simd_negate32(__m256i lhs)
{
    return _mm256_xor_si256(lhs, _mm256_cmpeq_epi32(lhs, lhs)); // cmmpeq always returns 0xFFFFFFFF
}
static inline __m256i simd_neq_epi32(__m256i lhs, __m256i rhs)
{
    return simd_negate32(_mm256_cmpeq_epi32(lhs, rhs));
}

template <bool PositiveAxis, int Resolution>
static void computeGridHashInPlaceX(HashGrid<Resolution>& hashGrid)
{
    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    for (int z = 0; z < Resolution; ++z) {
        for (int y = 0; y < Resolution; ++y) {
            HASH_TYPE combinedHash = 0;
            for (int x = start; x != end; x += step) {
                const glm::uvec3 p { x, y, z };
                const HASH_TYPE hash = hashGrid.get(p);
                const HASH_TYPE started = ((hash | combinedHash) == 0 ? 0 : std::numeric_limits<HASH_TYPE>::max());
                hash_combine<HASH_TYPE, HASH_FUNCTION>(combinedHash, hash);
                combinedHash = std::max(combinedHash, (HASH_TYPE)1);
                combinedHash &= started;
                hashGrid.set(p, combinedHash);
            }
        }
    }
}
template <bool PositiveAxis, int Resolution>
static void computeGridHashInPlaceY(HashGrid<Resolution>& hashGrid)
{
    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    for (int z = 0; z < Resolution; ++z) {
        for (int x = 0; x < Resolution; ++x) {
            HASH_TYPE combinedHash = 0;
            for (int y = start; y != end; y += step) {
                const glm::uvec3 p { x, y, z };
                const HASH_TYPE hash = hashGrid.get(p);
                const HASH_TYPE started = ((hash | combinedHash) == 0 ? 0 : std::numeric_limits<HASH_TYPE>::max());
                hash_combine<HASH_TYPE, HASH_FUNCTION>(combinedHash, hash);
                combinedHash = std::max(combinedHash, (HASH_TYPE)1);
                combinedHash &= started;
                hashGrid.set(p, combinedHash);
            }
        }
    }
}
template <bool PositiveAxis, int Resolution>
static void computeGridHashInPlaceZ(HashGrid<Resolution>& hashGrid)
{
    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    for (int x = 0; x < Resolution; ++x) {
        for (int y = 0; y < Resolution; ++y) {
            HASH_TYPE combinedHash = 0;
            for (int z = start; z != end; z += step) {
                const glm::uvec3 p { x, y, z };
                const HASH_TYPE hash = hashGrid.get(p);
                const HASH_TYPE started = ((hash | combinedHash) == 0 ? 0 : std::numeric_limits<HASH_TYPE>::max());
                hash_combine<HASH_TYPE, HASH_FUNCTION>(combinedHash, hash);
                combinedHash = std::max(combinedHash, (HASH_TYPE)1);
                combinedHash &= started;
                hashGrid.set(p, combinedHash);
            }
        }
    }
}

template <bool PositiveAxis, int Resolution>
static void computeGridHashInPlaceX_AVX2(HashGrid<Resolution>& hashGrid)
{
    static_assert(std::is_same_v<HASH_FUNCTION, std::identity>);
    static_assert(HashGrid<Resolution>::strideX == 1);

    if constexpr ((std::is_same_v<HASH_TYPE, uint32_t> && Resolution < 8) || (std::is_same_v<HASH_TYPE, uint64_t> && Resolution < 4)) {
        return computeGridHashInPlaceX(hashGrid);
    }

    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;

    if constexpr (std::is_same_v<HASH_TYPE, uint64_t>) {
        const __m256i zero = _mm256_set1_epi64x(0);
        const __m256i one = _mm256_set1_epi64x(1);
        for (int z = 0; z < Resolution; ++z) {
            for (int y = 0; y < Resolution; y += 4) {
                HASH_TYPE* ptr = &hashGrid.voxels[start * hashGrid.strideX + y * hashGrid.strideY + z * hashGrid.strideZ];
                __m256i hash = zero;

                HASH_TYPE* addr0 = ptr + 0 * hashGrid.strideY;
                HASH_TYPE* addr1 = ptr + 1 * hashGrid.strideY;
                HASH_TYPE* addr2 = ptr + 2 * hashGrid.strideY;
                HASH_TYPE* addr3 = ptr + 3 * hashGrid.strideY;

                alignas(64) std::array<uint64_t, 4> data;
                for (int x = start; x != end; x += step) {
                    data[0] = *addr0;
                    data[1] = *addr1;
                    data[2] = *addr2;
                    data[3] = *addr3;

                    const __m256i v = _mm256_load_si256((const __m256i*)data.data());
                    const __m256i started = simd_neq_epi64(_mm256_or_si256(hash, v), zero);
                    hash = hash_combine64(hash, v);
                    hash = _my_mm256_max_epu64(hash, one);
                    hash = _mm256_and_si256(started, hash);
                    _mm256_store_si256((__m256i*)data.data(), hash);

                    *addr0 = data[0];
                    *addr1 = data[1];
                    *addr2 = data[2];
                    *addr3 = data[3];

                    addr0 += step * (intptr_t)hashGrid.strideX;
                    addr1 += step * (intptr_t)hashGrid.strideX;
                    addr2 += step * (intptr_t)hashGrid.strideX;
                    addr3 += step * (intptr_t)hashGrid.strideX;
                }
            }
        }
    }
}
template <bool PositiveAxis, int Resolution>
static void computeGridHashInPlaceY_AVX2(HashGrid<Resolution>& hashGrid)
{
    static_assert(std::is_same_v<HASH_FUNCTION, std::identity>);
    static_assert(HashGrid<Resolution>::strideX == 1);

    if constexpr ((std::is_same_v<HASH_TYPE, uint32_t> && Resolution < 8) || (std::is_same_v<HASH_TYPE, uint64_t> && Resolution < 4)) {
        return computeGridHashInPlaceY(hashGrid);
    }

    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;

    if constexpr (std::is_same_v<HASH_TYPE, uint64_t>) {
        const __m256i zero = _mm256_set1_epi64x(0);
        const __m256i one = _mm256_set1_epi64x(1);
        for (int z = 0; z < Resolution; ++z) {
            for (int x = 0; x < Resolution; x += 4) {
                HASH_TYPE* ptr = hashGrid.voxels.data() + x * hashGrid.strideX + start * hashGrid.strideY + z * hashGrid.strideZ;
                __m256i hash = zero;

                for (int y = start; y != end; y += step) {
                    const __m256i v = _mm256_load_si256((const __m256i*)ptr);
                    const __m256i started = simd_neq_epi64(_mm256_or_si256(hash, v), zero);
                    hash = hash_combine64(hash, v);
                    hash = _my_mm256_max_epu64(hash, one);
                    hash = _mm256_and_si256(started, hash);
                    _mm256_store_si256((__m256i*)ptr, hash);
                    ptr += step * (intptr_t)hashGrid.strideY;
                }
            }
        }
    } else if constexpr (std::is_same_v<HASH_TYPE, uint32_t>) {
        const __m256i zero = _mm256_set1_epi32(0);
        const __m256i one = _mm256_set1_epi32(1);
        for (int z = 0; z < Resolution; ++z) {
            for (int x = 0; x < Resolution; x += 8) {
                HASH_TYPE* ptr = hashGrid.voxels.data() + x * hashGrid.strideX + start * hashGrid.strideY + z * hashGrid.strideZ;
                __m256i hash = zero;

                for (int y = start; y != end; y += step) {
                    const __m256i v = _mm256_load_si256((const __m256i*)ptr);
                    const __m256i started = simd_neq_epi32(_mm256_or_si256(hash, v), zero);
                    hash = hash_combine32(hash, v);
                    hash = _mm256_max_epu32(hash, one);
                    hash = _mm256_and_si256(started, hash);
                    _mm256_store_si256((__m256i*)ptr, hash);
                    ptr += step * (intptr_t)hashGrid.strideY;
                }
            }
        }
    }
}
template <bool PositiveAxis, int Resolution>
static void computeGridHashInPlaceZ_AVX2(HashGrid<Resolution>& hashGrid)
{
    static_assert(std::is_same_v<HASH_FUNCTION, std::identity>);
    static_assert(HashGrid<Resolution>::strideX == 1);

    if constexpr ((std::is_same_v<HASH_TYPE, uint32_t> && Resolution < 8) || (std::is_same_v<HASH_TYPE, uint64_t> && Resolution < 4)) {
        return computeGridHashInPlaceZ(hashGrid);
    }

    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    if constexpr (std::is_same_v<HASH_TYPE, uint64_t>) {
        const __m256i zero = _mm256_set1_epi64x(0);
        const __m256i one = _mm256_set1_epi64x(1);
        for (int y = 0; y < Resolution; ++y) {
            for (int x = 0; x < Resolution; x += 4) {
                HASH_TYPE* ptr = hashGrid.voxels.data() + x * hashGrid.strideX + y * hashGrid.strideY + start * hashGrid.strideZ;
                __m256i hash = zero;

                for (int z = start; z != end; z += step) {
                    const __m256i v = _mm256_load_si256((const __m256i*)ptr);
                    const __m256i started = simd_neq_epi64(_mm256_or_si256(hash, v), zero);
                    hash = hash_combine64(hash, v);
                    hash = _my_mm256_max_epu64(hash, one);
                    hash = _mm256_and_si256(started, hash);
                    _mm256_store_si256((__m256i*)ptr, hash);
                    ptr += step * (intptr_t)hashGrid.strideZ;
                }
            }
        }
    } else if constexpr (std::is_same_v<HASH_TYPE, uint32_t>) {
        const __m256i zero = _mm256_set1_epi32(0);
        const __m256i one = _mm256_set1_epi32(1);
        for (int y = 0; y < Resolution; ++y) {
            for (int x = 0; x < Resolution; x += 8) {
                HASH_TYPE* ptr = hashGrid.voxels.data() + x * hashGrid.strideX + y * hashGrid.strideY + start * hashGrid.strideZ;
                __m256i hash = zero;

                for (int z = start; z != end; z += step) {
                    const __m256i v = _mm256_load_si256((const __m256i*)ptr);
                    const __m256i started = simd_neq_epi32(_mm256_or_si256(hash, v), zero);
                    hash = hash_combine32(hash, v);
                    hash = _mm256_max_epu32(hash, one);
                    hash = _mm256_and_si256(started, hash);
                    _mm256_store_si256((__m256i*)ptr, hash);
                    ptr += step * (intptr_t)hashGrid.strideZ;
                }
            }
        }
    }
}

#if ENABLE_AVX512
template <bool PositiveAxis, int Resolution>
static void computeGridHashInPlaceZ_AVX512(HashGrid<Resolution>& hashGrid)
{
    static_assert(std::is_same_v<HASH_FUNCTION, std::identity>);
    static_assert(HashGrid<Resolution>::strideX == 1);

    if constexpr ((std::is_same_v<HASH_TYPE, uint32_t> && Resolution < 16) || (std::is_same_v<HASH_TYPE, uint64_t> && Resolution < 8)) {
        return computeGridHashInPlaceZ_AVX2<PositiveAxis, Resolution>(hashGrid);
    }

    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    if constexpr (std::is_same_v<HASH_TYPE, uint64_t>) {
        const __m512i zero = _mm512_set1_epi64(0);
        const __m512i one = _mm512_set1_epi64(1);
        for (int y = 0; y < Resolution; ++y) {
            for (int x = 0; x < Resolution; x += 8) {
                HASH_TYPE* ptr = hashGrid.voxels.data() + x * hashGrid.strideX + y * hashGrid.strideY + start * hashGrid.strideZ;
                __m512i hash = zero;

                for (int z = start; z != end; z += step) {
                    const __m512i v = _mm512_load_si512((const __m512i*)ptr);
                    const __mmask8 started = _mm512_cmpneq_epu64_mask(_mm512_or_si512(hash, v), zero);
                    hash = hash_combine64(hash, v);
                    hash = _mm512_max_epu64(hash, one);
                    hash = _mm512_mask_blend_epi64(started, zero, hash);
                    _mm512_store_si512((__m512i*)ptr, hash);
                    ptr += step * (intptr_t)hashGrid.strideZ;
                }
            }
        }
    }
}
#endif

template <int Resolution, bool PositiveX, bool PositiveY, bool PositiveZ>
static void computeGridHashInPlace(HashGrid<Resolution>& hashGrid)
{
    computeGridHashInPlaceX<PositiveX, Resolution>(hashGrid);
    computeGridHashInPlaceY<PositiveY, Resolution>(hashGrid);
    computeGridHashInPlaceZ<PositiveZ, Resolution>(hashGrid);
}
template <int Resolution, bool PositiveX, bool PositiveY, bool PositiveZ>
static void computeGridHashInPlace_AVX2(HashGrid<Resolution>& hashGrid)
{
    computeGridHashInPlaceX<PositiveX, Resolution>(hashGrid);
    computeGridHashInPlaceY<PositiveY, Resolution>(hashGrid);
    computeGridHashInPlaceZ<PositiveZ, Resolution>(hashGrid);
}

template <int Resolution, bool PositiveX, bool PositiveY, bool PositiveZ>
static void computePermutedGridHashInPlace_AVX2(HashGrid<Resolution>& hashGrid, const glm::u8vec3& axisPermutation)
{
    const auto computeHashAlongAxis = [&]<bool Positive>(int axis, std::integral_constant<bool, Positive>) {
        if (axis == 0)
            computeGridHashInPlaceX<Positive, Resolution>(hashGrid);
        else if (axis == 1)
            computeGridHashInPlaceY<Positive, Resolution>(hashGrid);
        else
            computeGridHashInPlaceZ<Positive, Resolution>(hashGrid);
    };
    computeHashAlongAxis(axisPermutation.x, std::integral_constant<bool, PositiveX>());
    computeHashAlongAxis(axisPermutation.y, std::integral_constant<bool, PositiveY>());
    computeHashAlongAxis(axisPermutation.z, std::integral_constant<bool, PositiveZ>());
}

template <int Resolution>
void createHashGrid(const LargeSubGrid<Resolution>& inGrid, HashGrid<Resolution>& outHashGrid)
{
    for (uint32_t i = 0; i < outHashGrid.voxels.size(); ++i)
        outHashGrid.set(i, inGrid.get(i));
}
template <int Resolution>
void createHashGrid_AVX2(const LargeSubGrid<Resolution>& inGrid, HashGrid<Resolution>& outHashGrid)
{
    static_assert(inGrid.strideX == outHashGrid.strideX);
    static_assert(inGrid.strideY == outHashGrid.strideY);
    static_assert(inGrid.strideZ == outHashGrid.strideZ);
    static_assert(std::is_same_v<std::remove_all_extents_t<decltype(inGrid.voxels.storage)>, uint64_t>);

#if 0
    for (uint32_t i = 0; i < outHashGrid.voxels.size(); i++) {
        const uint64_t in = (inGrid.voxels.storage[i >> 6] >> (i & 63)) & 0b1;
        uint64_t* pOut = outHashGrid.voxels.data() + i;
        *pOut = in;
    }
#else
    if constexpr (std::is_same_v<HASH_TYPE, uint64_t>) {
        const __m256i indices = _mm256_set_epi32(0, 3, 0, 2, 0, 1, 0, 0);
        const __m256i ones = _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1);
        for (uint32_t i = 0; i < outHashGrid.voxels.size(); i += 4) {
            const uint64_t in = (inGrid.voxels.storage[i >> 6] >> (i & 63)) & 0b1111;
            uint64_t* pOut = outHashGrid.voxels.data() + i;

            const __m256i bits = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(in), indices), ones);
            _mm256_store_si256((__m256i*)pOut, bits);
        }
    } else if constexpr (std::is_same_v<HASH_TYPE, uint32_t>) {
        const __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        const __m256i ones = _mm256_set1_epi32(1);
        for (uint32_t i = 0; i < outHashGrid.voxels.size(); i += 8) {
            const uint32_t in = (inGrid.voxels.storage[i >> 6] >> (i & 63)) & 0xFF;
            uint32_t* pOut = outHashGrid.voxels.data() + i;

            const __m256i bits = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(in), indices), ones);
            _mm256_store_si256((__m256i*)pOut, bits);
        }
    }
#endif
}

// Work-around while waiting for fetch_min to be implemented (C++26).
static void atomicMin64(std::atomic_uint64_t& atomic, uint64_t value)
{
    uint64_t expected = atomic.load(std::memory_order_relaxed);
    while (value < expected && !atomic.compare_exchange_weak(expected, value, std::memory_order_relaxed, std::memory_order_relaxed))
        ;
}

static uint64_t encodeTranslation0(int translation)
{
    if (translation >= 0) {
        return ((uint64_t)(+translation) << 1) | (0b0);
    } else {
        return ((uint64_t)(-translation) << 1) | (0b1);
    }
}
static int decodeTranslation0(uint64_t encodedTranslation)
{
    const int sign = encodedTranslation & 0b1 ? -1 : +1;
    return sign * (int)(encodedTranslation >> 1);
}

template <uint32_t Level>
static TransformPointer decodeSortingShiftPointer64(uint64_t encodedShiftPointer)
{
    constexpr uint32_t numTranslationBits = Level + 1;
    constexpr uint64_t translationMask = (1llu << (3 * numTranslationBits)) - 1llu;
    constexpr uint64_t singleAxisTranslationMask = (1llu << numTranslationBits) - 1llu;
    constexpr uint64_t translationStart = 64 - 3 * numTranslationBits;
    constexpr uint64_t axisPermutationStart = translationStart - 3;
    constexpr uint64_t symmetryStart = axisPermutationStart - 3;
    constexpr uint64_t pointerMask = (1llu << symmetryStart) - 1llu;

    const uint64_t symmetryBits = (encodedShiftPointer >> symmetryStart) & 0b111;
    const uint64_t axisPermutationBits = (encodedShiftPointer >> axisPermutationStart) & 0b111;
    const uint64_t offsetBits = (encodedShiftPointer >> translationStart) & translationMask;

    glm::ivec3 translation;
    translation.x = decodeTranslation0((offsetBits >> (0 * numTranslationBits)) & singleAxisTranslationMask);
    translation.y = decodeTranslation0((offsetBits >> (1 * numTranslationBits)) & singleAxisTranslationMask);
    translation.z = decodeTranslation0((offsetBits >> (2 * numTranslationBits)) & singleAxisTranslationMask);

    TransformPointer out;
    out.ptr = encodedShiftPointer & pointerMask;
    out.setSymmetry(u64ToBvec3(symmetryBits));
    out.setAxisPermutation(TransformPointer::decodeAxisPermutation((uint32_t)axisPermutationBits));
    out.setTranslation(translation);
    return out;
}

template <uint32_t Level>
static uint64_t encodeSortingShiftPointer64(const TransformPointer& shiftPointer)
{
    constexpr uint32_t numTranslationBits = Level + 1;
    constexpr uint64_t translationStart = 64 - 3 * numTranslationBits;
    constexpr uint64_t axisPermutationStart = translationStart - 3;
    constexpr uint64_t symmetryStart = axisPermutationStart - 3;

    const auto translation = shiftPointer.getTranslation();
    // const glm::uvec3 offsetShift = glm::uvec3(shiftPointer.getTranslation() + glm::ivec3(1 << Level));
    //  const uint64_t offsetBits = (offsetShift.z << (2 * numTranslationBits)) | (offsetShift.y << (1 * numTranslationBits)) | offsetShift.x;
    const uint64_t offsetBits = (encodeTranslation0(translation.z) << (2 * numTranslationBits)) | (encodeTranslation0(translation.y) << (1 * numTranslationBits)) | encodeTranslation0(translation.x);
    const uint64_t axisPermutationBits = TransformPointer::encodeAxisPermutation(shiftPointer.getAxisPermutation());
    const uint64_t symmetryBits = bvec3ToU64(shiftPointer.getSymmetry());
    const uint64_t out = shiftPointer.ptr | (offsetBits << translationStart) | (axisPermutationBits << axisPermutationStart) | (symmetryBits << symmetryStart);

    const auto control = decodeSortingShiftPointer64<Level>(out);
    assert_always(control == shiftPointer);
    return out;
}

// Given a collection of items (EditSubGrid or EditNode) and their representation as a dense voxel grid,
//  find duplicates w.r.t. translation, symmetry and axis permutation. Returns the list of unique items and
//  the mapping from original indices into this new array (outPrevLevelMapping).
template <size_t Level, typename Item, typename GridGenerator>
std::vector<Item> findDuplicatesAmongFlattenedItems_cpp2(std::span<const Item> items, const GridGenerator& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config)
{
    using clock = std::chrono::high_resolution_clock;
    static constexpr size_t Resolution = 1u << Level;
    static constexpr size_t Resolution3 = Resolution * Resolution * Resolution;

#if USE_CUCKOO_HASHING
    using HashTable = CuckooHashTable<HASH_TYPE, std::pair<uint32_t, uint32_t>, 0>;
#else
    using HashTable = robin_hood::unordered_flat_map<HASH_TYPE, uint32_t, HASH_FUNCTION>;
    // using HashTable = absl::flat_hash_map<HASH_TYPE, uint32_t, HASH_FUNCTION>;
    //   using HashTable = std::unordered_map<HASH_TYPE, uint32_t>;
#endif

#if SORT_BY_VOXEL_COUNT
    std::vector<uint32_t> itemIndicesByVoxelCount(items.size());
    {
        spdlog::info("Sorting by voxel count");
        std::iota(std::begin(itemIndicesByVoxelCount), std::end(itemIndicesByVoxelCount), 0u);
        std::vector<uint16_t> itemVoxelCounts(items.size());
        std::transform(execution_policy, std::begin(itemIndicesByVoxelCount), std::end(itemIndicesByVoxelCount), std::begin(itemVoxelCounts),
            [&](uint32_t itemIdx) { return (uint16_t)grids[itemIdx].popcount(); });
        std::stable_sort(execution_policy, std::begin(itemIndicesByVoxelCount), std::end(itemIndicesByVoxelCount), [&](uint32_t lhs, uint32_t rhs) { return itemVoxelCounts[lhs] > itemVoxelCounts[rhs]; });
        // std::mt19937 rng {};
        // std::shuffle(std::begin(itemIndicesByVoxelCount), std::end(itemIndicesByVoxelCount), rng);
    }
#endif

    // Generate hashes for each voxel of the grid, storing what the grid looks like from that voxel of the grid.
    // This is repeated for all eight directions that we can look into (positive/negative x/y/z).
    // We search the previously generated hash table for grids that looked the same, as seen from any voxel.
    //
    // The search may lead to one grid having multiple parents from which it can be constructed.
    // We decide to store the one that appears first in the array of items/grids. This allows groups of grids which
    // can all be transformed into each-other, to all pick the same parent.
    const uint64_t encodedSentinel = TransformPointer::sentinel().encodeFixed64(Level);

    // Initialize the arrays storing for each item/grid, how it can be reconstructed from another grid.
    spdlog::info("Initializing parent pointers");
    std::vector<std::atomic_uint64_t> encodedParents(items.size()); // Pointers to other grids for which the transformation is not invertible.
    std::for_each(execution_policy, std::begin(encodedParents), std::end(encodedParents), [=](std::atomic_uint64_t& atomicU64) { atomicU64.store(encodedSentinel, std::memory_order::relaxed); });
    spdlog::info("encodedParents memory usage: {}MiB", sizeOfVector(encodedParents) >> 20);

    // Generate hashes of what each grid looks like from each of the 8 corner of the grid.
    constexpr size_t estimatedBytesPerBasePointer = 8 + 8 + 12; // 64-bit pointer, 64-bit range (shared between pointers), 64+32 bit hash table entry per range.
    constexpr size_t maxBasePointerBatchSizeInBytes = 32 * 1024llu * 1024llu * 1024llu;
    constexpr size_t maxBasePointerbatchSize = maxBasePointerBatchSizeInBytes / (estimatedBytesPerBasePointer * 8); // 8 pointers per item (one per direction).
    for (size_t basePointerBatchStart = 0; basePointerBatchStart < items.size(); basePointerBatchStart += maxBasePointerbatchSize) {
        std::vector<TransformPointer> pointers;
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        HashTable gridHashesLUT;

        const size_t basePointerBatchEnd = std::min(basePointerBatchStart + maxBasePointerbatchSize, items.size());
        const size_t basePointerBatchSize = basePointerBatchEnd - basePointerBatchStart;

        spdlog::info("Computing base hashes");
        {
            std::vector<uint32_t> indices(basePointerBatchSize);
            std::iota(std::begin(indices), std::end(indices), (uint32_t)basePointerBatchStart);
            std::vector<std::pair<HASH_TYPE, TransformPointer>> hashAndPointers(8 * basePointerBatchSize);
            const auto initializeLookUpTable = [&]<bool PositiveX, bool PositiveY, bool PositiveZ>(std::bool_constant<PositiveX>, std::bool_constant<PositiveY>, std::bool_constant<PositiveZ>) {
                const glm::ivec3 anchorPoint { PositiveX ? 0 : Resolution - 1, PositiveY ? 0 : Resolution - 1, PositiveZ ? 0 : Resolution - 1 };
                const glm::bvec3 direction { PositiveX, PositiveY, PositiveZ };

#if ENABLE_ASSERTIONS
                for (uint32_t i = 0; i < items.size(); ++i) {
                    const auto grid = grids[i];
                    HashGrid<Resolution> hashGrid, hashGrid2;
                    createHashGrid(grid, hashGrid);
                    createHashGrid_AVX2(grid, hashGrid2);
                    assert_always(hashGrid == hashGrid2);

                    computeGridHashInPlace<Resolution, true, true, true>(hashGrid);
                    computeGridHashInPlace_AVX2<Resolution, true, true, true>(hashGrid2);
                    assert_always(hashGrid == hashGrid2);

                    computeGridHashInPlace<Resolution, false, false, false>(hashGrid);
                    computeGridHashInPlace_AVX2<Resolution, false, false, false>(hashGrid2);
                    assert_always(hashGrid == hashGrid2);
                }
#endif

                // Parallel hash computation.
                const auto directionIdx = morton_encode32(glm::uvec3(direction));
                const auto t0 = clock::now();
                std::transform(execution_policy, std::begin(indices), std::end(indices), std::begin(hashAndPointers) + directionIdx * basePointerBatchSize,
                    [&](uint32_t itemIdx) {
                        HashGrid<Resolution> hashGrid;
                        grids.fillHashGrid(itemIdx, hashGrid);
                        computeGridHashInPlace_AVX2<Resolution, PositiveX, PositiveY, PositiveZ>(hashGrid);
                        return std::pair { hashGrid.get(anchorPoint), TransformPointer::create(itemIdx, direction) };
                    });
                const auto t1 = clock::now();
                spdlog::info("Computing hashAndPointers took {}", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0));
            };
            initializeLookUpTable(std::bool_constant<true>(), std::bool_constant<true>(), std::bool_constant<true>());
            initializeLookUpTable(std::bool_constant<true>(), std::bool_constant<true>(), std::bool_constant<false>());
            initializeLookUpTable(std::bool_constant<true>(), std::bool_constant<false>(), std::bool_constant<true>());
            initializeLookUpTable(std::bool_constant<true>(), std::bool_constant<false>(), std::bool_constant<false>());
            initializeLookUpTable(std::bool_constant<false>(), std::bool_constant<true>(), std::bool_constant<true>());
            initializeLookUpTable(std::bool_constant<false>(), std::bool_constant<true>(), std::bool_constant<false>());
            initializeLookUpTable(std::bool_constant<false>(), std::bool_constant<false>(), std::bool_constant<true>());
            initializeLookUpTable(std::bool_constant<false>(), std::bool_constant<false>(), std::bool_constant<false>());

            spdlog::info("Removing empty hashAndPointers");
            std::erase_if(hashAndPointers, [](const std::pair<HASH_TYPE, TransformPointer>& hashAndPointer) { return std::get<0>(hashAndPointer) == 0; });
            hashAndPointers.shrink_to_fit();
            spdlog::info("hashAndPointers memory usage: {}MiB", sizeOfVector(hashAndPointers) >> 20);

            spdlog::info("Sorting hashAndPointers");
            std::sort(execution_policy, std::begin(hashAndPointers), std::end(hashAndPointers), [](const auto& lhs, const auto& rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });

            spdlog::info("Constructing ranges");
            std::vector<HASH_TYPE> rangeHashes;
            {
                uint32_t start = 0;
                HASH_TYPE prevHash = std::get<0>(hashAndPointers[0]);
                for (uint32_t i = 0; i < hashAndPointers.size(); ++i) {
                    if (std::get<0>(hashAndPointers[i]) != prevHash) {
                        ranges.push_back({ start, i });
                        rangeHashes.push_back(prevHash);
                        prevHash = std::get<0>(hashAndPointers[i]);
                        start = i;
                    }
                }

                ranges.push_back({ start, (uint32_t)hashAndPointers.size() });
                rangeHashes.push_back(std::get<0>(hashAndPointers.back()));
            }
            pointers.resize(hashAndPointers.size());
            std::transform(std::execution::par, std::begin(hashAndPointers), std::end(hashAndPointers), std::begin(pointers),
                [&](const std::pair<HASH_TYPE, TransformPointer>& hashAndPointer) { return std::get<1>(hashAndPointer); });

            spdlog::info("Construct hash table over ranges");
#if USE_CUCKOO_HASHING
            gridHashesLUT = HashTable(rangeHashes, ranges);
#else
            for (uint32_t i = 0; i < rangeHashes.size(); ++i) {
                const auto hash = rangeHashes[i];
                gridHashesLUT[hash] = i;
            }
            gridHashesLUT.compact();
#endif
            spdlog::info("Minimum hash table memory usage: {}MiB", (rangeHashes.size() * (sizeof(HASH_TYPE) + sizeof(uint32_t))) >> 20);
        }
        spdlog::info("pointers memory usage: {}MiB", sizeOfVector(pointers) >> 20);
        spdlog::info("ranges memory usage: {}MiB", sizeOfVector(ranges) >> 20);

        const auto numVoxels = items.size() * Resolution3;
        const auto numPermutedVoxels = numVoxels * 8 * TransformPointer::NumUniqueAxisPermutations;
        spdlog::info("{} grids of {}^3={} voxels ({} voxels; {} permuted voxels; {} GiB)", items.size(), Resolution, Resolution3, numVoxels, numPermutedVoxels, (numPermutedVoxels * sizeof(HASH_TYPE)) >> 30);

        spdlog::info("Start search");
        const auto t0 = clock::now();
        const size_t maxSearchBatchSize = std::max(items.size() / 100, (size_t)100'000);
        for (size_t searchBatchStart = 0; searchBatchStart < items.size(); searchBatchStart += maxSearchBatchSize) {
            // For each item/grid.
            const size_t searchBatchEnd = std::min(searchBatchStart + maxSearchBatchSize, items.size());
            spdlog::info("{} / {} ({:.2f}%)", searchBatchStart, items.size(), double(searchBatchStart) / items.size() * 100.0);
            std::vector<uint32_t> itemIndices(searchBatchEnd - searchBatchStart);
            std::iota(std::begin(itemIndices), std::end(itemIndices), (uint32_t)searchBatchStart);
            const auto ta0 = clock::now();
            std::for_each(execution_policy, std::begin(itemIndices), std::end(itemIndices),
                [&](uint32_t itemIdx) {
                    const auto minEncodedShiftPointer = TransformPointer::create(itemIdx, glm::bvec3(false), glm::ivec3(0, 1, 2), glm::ivec3(1 - (1 << Level))).encodeFixed64(Level);
#if SORT_BY_VOXEL_COUNT
                    const auto grid = grids[itemIndicesByVoxelCount[itemIdx]];
#else
                    const auto grid = grids[itemIdx];
#endif
                    HashGrid<Resolution> initialHashGrid;
                    createHashGrid_AVX2<Resolution>(grid, initialHashGrid);
                    const auto searchDirection = [&]<bool PositiveX, bool PositiveY, bool PositiveZ>(std::integral_constant<bool, PositiveX>, std::integral_constant<bool, PositiveY>, std::integral_constant<bool, PositiveZ>) {
                        const glm::ivec3 anchorPoint { PositiveX ? 0 : Resolution - 1, PositiveY ? 0 : Resolution - 1, PositiveZ ? 0 : Resolution - 1 };
                        const glm::bvec3 direction { PositiveX, PositiveY, PositiveZ };
                        // For each way that we can permute the order of the axis.
                        for (uint8_t permutationID = 0; permutationID < TransformPointer::NumUniqueAxisPermutations; ++permutationID) {
                            // Reorder the axis of the grid accordingly, and then compute the hash for each voxel.
                            const auto axisPermutation = TransformPointer::decodeAxisPermutation(permutationID);
                            auto hashGrid = initialHashGrid;
                            computePermutedGridHashInPlace_AVX2<Resolution, PositiveX, PositiveY, PositiveZ>(hashGrid, axisPermutation);

                            // Loop over all voxels in the grid.
                            for (uint32_t hashVoxelIdx = 0; hashVoxelIdx < Resolution * Resolution * Resolution; ++hashVoxelIdx) {
                                const glm::ivec3 hashVoxel = morton_decode32<3>(hashVoxelIdx);
                                //  Get the hash of what the grid looks like from the selected voxel in the selected positive/negative x/y/z direction.
                                const auto hash = hashGrid.get(hashVoxel);
                                if (hash == 0) // A hash of 0 indicates that the grid, as seen from this voxel, is empty.
                                    continue;

                                // Find the list of original grids (seen from any of their 8 corners) that look the same as the current grid from this voxel.
#if USE_CUCKOO_HASHING
                                std::pair<uint32_t, uint32_t> beginEnd;
                                if (!gridHashesLUT.find(hash, beginEnd))
                                    continue;
                                const auto [begin, end] = beginEnd;
#else
                                const auto matchesIter = gridHashesLUT.find(hash);
                                if (matchesIter == std::end(gridHashesLUT))
                                    continue;
                                const auto [begin, end] = ranges[matchesIter->second];
#endif
                                const glm::ivec3 voxel = applyAxisPermutation(hashVoxel, axisPermutation);
                                for (auto gridHashIdx = begin; gridHashIdx < end; ++gridHashIdx) {
                                    const auto& matchingItem = pointers[gridHashIdx];
                                    const auto matchingItemIdx = matchingItem.ptr;
                                // Potential matching "child" grid which can be created by transforming the current grid.
                                // auto& potentialChild = encodedParents[matchingItemIdx];
#if !SORT_BY_VOXEL_COUNT
                                    if (matchingItemIdx == itemIdx) // Don't match to self.
                                        continue;
#endif

                                    uint64_t currentBestPointer = encodedParents[matchingItemIdx].load(std::memory_order::relaxed);
                                    if (minEncodedShiftPointer >= currentBestPointer)
                                        continue;

                                    // Compute the transformation required to turn the current grid into the matching "child" grid.
                                    const auto symmetry = direction ^ matchingItem.getSymmetry();
                                    const auto translation = glm::mix(anchorPoint - voxel, voxel - anchorPoint, symmetry);
                                    const auto shiftPointer = TransformPointer::create(itemIdx, symmetry, axisPermutation, translation);

                                    // Only consider the types of transformations that the user requested.
                                    if (!config.axisPermutation && shiftPointer.hasAxisPermutation())
                                        continue;
                                    if (!config.symmetry && shiftPointer.hasSymmetry())
                                        continue;
                                    if (!config.translation && shiftPointer.hasTranslation())
                                        continue;

                                    //  Skip if we already know this pointer won't be selected (only keeps the first match in the array).
                                    const auto encodedShiftPointer = shiftPointer.encodeFixed64(Level);
                                    if (encodedShiftPointer >= currentBestPointer)
                                        continue;

                                    // While a matching hash gives a very high likely hood of the two grids matching, there is a chance
                                    // that two different grids produce the same hash code (this is called a "hash collision").
                                    // To ensure correctness we transform the current grid and check whether it is indeed equivalent
                                    // to the matching "child" grid. This step is expensive but is required to prevent rare failures.
                                    assert_always(itemIdx < items.size());
                                    const auto permutedGrid = permuteAxisOfSubGrid<Resolution>(grid, shiftPointer.getAxisPermutation());
                                    const auto permutedSymmetryGrid = applySymmetryToSubGrid<Resolution>(permutedGrid, shiftPointer.getSymmetry()); // <=== THIS IS EXPENSIVE.
                                    const auto permutedSymmetryShiftedGrid = translateSubGrid<Resolution>(permutedSymmetryGrid, shiftPointer.getTranslation()); // <=== THIS IS EXPENSIVE.

                                    if (grids[(uint32_t)matchingItemIdx] != permutedSymmetryShiftedGrid)
                                        continue;

                                    atomicMin64(encodedParents[matchingItemIdx], encodedShiftPointer);
                                } // for all potential matches
                            } // voxelIdx
                        } // axis permutation
                    };
                    searchDirection(std::bool_constant<true>(), std::bool_constant<true>(), std::bool_constant<true>());
                    searchDirection(std::bool_constant<true>(), std::bool_constant<true>(), std::bool_constant<false>());
                    searchDirection(std::bool_constant<true>(), std::bool_constant<false>(), std::bool_constant<true>());
                    searchDirection(std::bool_constant<true>(), std::bool_constant<false>(), std::bool_constant<false>());
                    searchDirection(std::bool_constant<false>(), std::bool_constant<true>(), std::bool_constant<true>());
                    searchDirection(std::bool_constant<false>(), std::bool_constant<true>(), std::bool_constant<false>());
                    searchDirection(std::bool_constant<false>(), std::bool_constant<false>(), std::bool_constant<true>());
                    searchDirection(std::bool_constant<false>(), std::bool_constant<false>(), std::bool_constant<false>());
                });
            const auto ta1 = clock::now();
            const auto batchSize = searchBatchEnd - searchBatchStart;
            const auto numBatchPermutedVoxels = batchSize * Resolution3 * 8 * TransformPointer::NumUniqueAxisPermutations;
            const auto batchSearchTime = std::chrono::duration<double>(ta1 - ta0).count();
            spdlog::info("Search batch of {} grids took {:.2f}s ({:.2f} MiB/s)",
                batchSize,
                batchSearchTime,
                double(numBatchPermutedVoxels * sizeof(HASH_TYPE)) / batchSearchTime / 1024 / 1024);

#if !USE_CUCKOO_HASHING
            // Compute for each range of pointers with the same hash, whether any future encodedParent may improve over the current value.
            // If not then we remove the range of pointers from the hash table.
            const auto nextSearchBatchStart = searchBatchEnd + 1;
            const auto nextSearchBatchMinEncodedTransformPointer = TransformPointer::create(nextSearchBatchStart, glm::bvec3(false), glm::ivec3(0, 1, 2), glm::ivec3(1 - (1 << Level))).encodeFixed64(Level);

            // For each item, we maintain the parent with the *lowest* numeric pointer value.
            // We iterate over the ranges of items with the same hash, removing items who's encodedParent value exceeds
            // the lowest pointer achievable in the remaining batches. This should not have any impact on the final result,
            // it is purely here to achieve a performance speed-up.
            std::transform(execution_policy, std::begin(ranges), std::end(ranges), std::begin(ranges),
                [&](std::pair<uint32_t, uint32_t> range) {
                    const auto [rangeBegin, rangeEnd] = range;
                    uint32_t newRangeEnd = rangeBegin;
                    for (uint32_t i = rangeBegin; i < rangeEnd; ++i) {
                        const auto matchingItemPointer = pointers[i];
                        const auto encodedParent = encodedParents[matchingItemPointer.ptr].load(std::memory_order::relaxed);
                        if (encodedParent > nextSearchBatchMinEncodedTransformPointer)
                            pointers[newRangeEnd++] = matchingItemPointer;
                    }
                    return std::pair { rangeBegin, newRangeEnd };
                });

            for (auto iter = std::begin(gridHashesLUT); iter != std::end(gridHashesLUT); ++iter) {
                const auto [rangeBegin, rangeEnd] = ranges[iter->second];
                if (rangeBegin == rangeEnd)
                    gridHashesLUT.erase(iter);
            }
#endif
            const auto ta2 = clock::now();
            spdlog::info("Hash table update took {}", std::chrono::duration_cast<std::chrono::milliseconds>(ta2 - ta1));
        }
        const auto t1 = clock::now();
        spdlog::info("Matching took {}", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0));
    }

    const auto decodeShiftPointer = [&](const std::atomic_uint64_t& encodedShiftPointer) {
        if (encodedShiftPointer.load() == encodedSentinel) {
            return TransformPointer::sentinel();
        } else {
#if SORT_BY_VOXEL_COUNT
            auto out = TransformPointer::decodeFixed64(encodedShiftPointer.load(), Level);
            out.ptr = itemIndicesByVoxelCount[out.ptr];
            return out;
#else
            return TransformPointer::decodeFixed64(encodedShiftPointer.load(), Level);
#endif
        }
    };
    std::vector<TransformPointer> parents(items.size());
    std::transform(execution_policy, std::begin(encodedParents), std::end(encodedParents), std::begin(parents), decodeShiftPointer);
    encodedParents.clear();
    spdlog::info("parents memory usage: {}MiB", sizeOfVector(parents) >> 20);

    // Determine for each item whether it will be inserted into the final DAG.
    spdlog::info("Process results");
    std::vector<bool> isUnique(items.size(), false);
    // We now know for every grid whether there is (one) other grid from which it can be reconstructed.
    // When nodes can be reconstructed from each-other (the transformation is invertible) then two nodes
    // could point to each-other and create a circle. A similar problem is that we may create chains of
    // transformations: node A is reconstructed from node B, which is reconstructed from node C. We want
    // to limit ourselves to a single level of indirection in order to keep traversal of the DAG efficient.
    // The following code breaks these chains & cycles by indicating which nodes must be stored (isUnique),
    // and which ones are allowed to be reconstructed (!isUnique).
    size_t numUnique = 0, numUniqueInvariant = 0, numDoubleIndirect = 0;
#if 1
    for (size_t itemIdx = 0; itemIdx < items.size(); ++itemIdx) {
        const auto parent = parents[itemIdx];
        if (parent == TransformPointer::sentinel()) {
            isUnique[itemIdx] = true;
        } else {
            // assert_always(parent.ptr != itemIdx);
            isUnique[parent.ptr] = true;
        }
    }
    numUnique = std::count(std::begin(isUnique), std::end(isUnique), true);
#else
    // Determine for each item whether it will be inserted into the final DAG.
    for (size_t itemIdx = 0; itemIdx < items.size(); ++itemIdx) {
        if (parents[itemIdx] == TransformPointer::sentinel()) {
            isUnique[itemIdx] = true;
            ++numUnique;
        }
    }
    // We currently don't support multiple indirections.
    for (size_t itemIdx = 0; itemIdx < items.size(); ++itemIdx) {
        if (parents[itemIdx] != TransformPointer::sentinel() && !isUnique[parents[itemIdx].ptr]) {
            isUnique[itemIdx] = true;
            ++numDoubleIndirect;
        }
    }
#endif
    spdlog::info("Reduced {} SVDAG to {} SVTDAG items ({} unique, {} unique invariant, {} double indirect)", items.size(), numUnique + numUniqueInvariant + numDoubleIndirect, numUnique, numUniqueInvariant, numDoubleIndirect);

    // Output all unique items and store the mapping from old to new indices (outPrevLevelMapping).
    outPrevLevelMapping.resize(items.size());
    std::fill(std::begin(outPrevLevelMapping), std::end(outPrevLevelMapping), TransformPointer::sentinel());
    std::vector<Item> out;
    for (size_t itemIdx = 0; itemIdx < items.size(); ++itemIdx) {
        if (isUnique[itemIdx]) {
            outPrevLevelMapping[itemIdx] = TransformPointer::create(out.size());
            out.push_back(items[itemIdx]);
        }
    }
    for (size_t itemIdx = 0; itemIdx < items.size(); ++itemIdx) {
        if (!isUnique[itemIdx]) {
            auto parent = parents[itemIdx];
            parent.ptr = outPrevLevelMapping[parent.ptr].ptr;
            outPrevLevelMapping[itemIdx] = parent;
        }
    }
    spdlog::info("out memory usage: {}MiB", sizeOfVector(out) >> 20);
    return out;
}

template std::vector<EditSubGrid<void>> findDuplicatesAmongFlattenedItems_cpp2<2, EditSubGrid<void>, FlatGridGenerator<2, EditStructure<void, TransformPointer>>>(std::span<const EditSubGrid<void>> items, const FlatGridGenerator<2, EditStructure<void, TransformPointer>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<EditNode<TransformPointer>> findDuplicatesAmongFlattenedItems_cpp2<3, EditNode<TransformPointer>, FlatGridGenerator<3, EditStructure<void, TransformPointer>>>(std::span<const EditNode<TransformPointer>> items, const FlatGridGenerator<3, EditStructure<void, TransformPointer>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<EditNode<TransformPointer>> findDuplicatesAmongFlattenedItems_cpp2<4, EditNode<TransformPointer>, FlatGridGenerator<4, EditStructure<void, TransformPointer>>>(std::span<const EditNode<TransformPointer>> items, const FlatGridGenerator<4, EditStructure<void, TransformPointer>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<EditNode<TransformPointer>> findDuplicatesAmongFlattenedItems_cpp2<5, EditNode<TransformPointer>, FlatGridGenerator<5, EditStructure<void, TransformPointer>>>(std::span<const EditNode<TransformPointer>> items, const FlatGridGenerator<5, EditStructure<void, TransformPointer>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);

template std::vector<EditSubGrid<void>> findDuplicatesAmongFlattenedItems_cpp2<2, EditSubGrid<void>, FlatGridGenerator<2, EditStructureOOC<void, TransformPointer>>>(std::span<const EditSubGrid<void>> items, const FlatGridGenerator<2, EditStructureOOC<void, TransformPointer>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<EditNode<TransformPointer>> findDuplicatesAmongFlattenedItems_cpp2<3, EditNode<TransformPointer>, FlatGridGenerator<3, EditStructureOOC<void, TransformPointer>>>(std::span<const EditNode<TransformPointer>> items, const FlatGridGenerator<3, EditStructureOOC<void, TransformPointer>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<EditNode<TransformPointer>> findDuplicatesAmongFlattenedItems_cpp2<4, EditNode<TransformPointer>, FlatGridGenerator<4, EditStructureOOC<void, TransformPointer>>>(std::span<const EditNode<TransformPointer>> items, const FlatGridGenerator<4, EditStructureOOC<void, TransformPointer>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<EditNode<TransformPointer>> findDuplicatesAmongFlattenedItems_cpp2<5, EditNode<TransformPointer>, FlatGridGenerator<5, EditStructureOOC<void, TransformPointer>>>(std::span<const EditNode<TransformPointer>> items, const FlatGridGenerator<5, EditStructureOOC<void, TransformPointer>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);

template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_cpp2<2, uint32_t, FlatGridGenerator<2, EditStructure<void, uint32_t>>>(std::span<const uint32_t> items, const FlatGridGenerator<2, EditStructure<void, uint32_t>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_cpp2<3, uint32_t, FlatGridGenerator<3, EditStructure<void, uint32_t>>>(std::span<const uint32_t> items, const FlatGridGenerator<3, EditStructure<void, uint32_t>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_cpp2<4, uint32_t, FlatGridGenerator<4, EditStructure<void, uint32_t>>>(std::span<const uint32_t> items, const FlatGridGenerator<4, EditStructure<void, uint32_t>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_cpp2<5, uint32_t, FlatGridGenerator<5, EditStructure<void, uint32_t>>>(std::span<const uint32_t> items, const FlatGridGenerator<5, EditStructure<void, uint32_t>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);

template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_cpp2<2, uint32_t, FlatGridGenerator<2, EditStructureOOC<void, uint32_t>>>(std::span<const uint32_t> items, const FlatGridGenerator<2, EditStructureOOC<void, uint32_t>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_cpp2<3, uint32_t, FlatGridGenerator<3, EditStructureOOC<void, uint32_t>>>(std::span<const uint32_t> items, const FlatGridGenerator<3, EditStructureOOC<void, uint32_t>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_cpp2<4, uint32_t, FlatGridGenerator<4, EditStructureOOC<void, uint32_t>>>(std::span<const uint32_t> items, const FlatGridGenerator<4, EditStructureOOC<void, uint32_t>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_cpp2<5, uint32_t, FlatGridGenerator<5, EditStructureOOC<void, uint32_t>>>(std::span<const uint32_t> items, const FlatGridGenerator<5, EditStructureOOC<void, uint32_t>>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);

template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_cpp2<2, uint32_t, StaticFlatGridGenerator<2>>(std::span<const uint32_t> items, const StaticFlatGridGenerator<2>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_cpp2<3, uint32_t, StaticFlatGridGenerator<3>>(std::span<const uint32_t> items, const StaticFlatGridGenerator<3>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_cpp2<4, uint32_t, StaticFlatGridGenerator<4>>(std::span<const uint32_t> items, const StaticFlatGridGenerator<4>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_cpp2<5, uint32_t, StaticFlatGridGenerator<5>>(std::span<const uint32_t> items, const StaticFlatGridGenerator<5>& grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
}
