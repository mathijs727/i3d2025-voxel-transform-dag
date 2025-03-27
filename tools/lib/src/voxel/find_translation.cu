#include "voxcom/utility/hash.h"
#include "voxcom/utility/maths.h"
#include "voxcom/voxel/large_sub_grid.h"
#include "voxcom/voxel/transform_dag.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
#include <execution>
#include <numeric>
#include <set>
#include <span>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <voxcom/utility/error_handling.h>
#include <voxcom/utility/fmt_glm.h>

#include "voxcom/utility/disable_all_warnings.h"
DISABLE_WARNINGS_PUSH()
#include <cooperative_groups.h>
#include <cuda.h>
#include <fmt/chrono.h>
#include <robin_hood.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <thrust/device_vector.h>
DISABLE_WARNINGS_POP()

#define PROCESS_PER_WARP 0
#define COMPARE_WITH_CPU 0
#define REDUCED_SHARED_MEMORY 1 // Process layer-by-layer to reduce memory usage from Resolution^3 to Resolution^2
#define SORT_BY_VOXEL_COUNT 1
#define ENABLE_ASSERTIONS 0
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
struct MyIdentityHash {
    HOST_DEVICE HASH_TYPE operator()(HASH_TYPE v) const
    {
        return v;
    }
};
#define HASH_FUNCTION MyHash

static constexpr auto execution_policy = std::execution::par;
namespace cg = cooperative_groups;
constexpr uint32_t threadsPerWarp = 32;

namespace voxcom {

template <int Resolution>
static HOST_DEVICE LargeSubGrid<Resolution> translateSubGrid(const LargeSubGrid<Resolution>& subGrid, const glm::ivec3& translation)
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
static HOST_DEVICE LargeSubGrid<Resolution> applySymmetryToSubGrid(const LargeSubGrid<Resolution>& subGrid, const glm::bvec3& symmetry)
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
static HOST_DEVICE LargeSubGrid<Resolution> permuteAxisOfSubGrid(const LargeSubGrid<Resolution>& subGrid, const glm::u8vec3& permutation)
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
// 3D dense voxel grid with size known at compile time. Will be used to store hashes of neighboring voxels.
template <size_t Resolution>
using HashGrid = TypedLargeSubGrid<HASH_TYPE, Resolution>;
template <size_t Resolution>
using HashGrid2D = TypedLargeSubGrid2D<HASH_TYPE, Resolution>;
template <size_t Resolution>
static HOST_DEVICE HashGrid<Resolution> permuteAxisOfSubGrid(const HashGrid<Resolution>& subGrid, const glm::u8vec3& permutation)
{
    HashGrid<Resolution> out {};
    for (unsigned z = 0; z < Resolution; ++z) {
        for (unsigned y = 0; y < Resolution; ++y) {
            for (unsigned x = 0; x < Resolution; ++x) {
                const glm::uvec3 inVox { x, y, z };
                const glm::uvec3 outVox { inVox[permutation[0]], inVox[permutation[1]], inVox[permutation[2]] };
                out.set(outVox, subGrid.get(inVox));
            }
        }
    }
    return out;
}

template <bool PositiveAxis, int Resolution>
static HOST_DEVICE void computeGridHashX(const HashGrid<Resolution>& inGrid, HashGrid<Resolution>& outGrid)
{
    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    for (int z = 0; z < Resolution; ++z) {
        for (int y = 0; y < Resolution; ++y) {
            bool started = false;
            HASH_TYPE hash = 0;
            for (int x = start; x != end; x += step) {
                const glm::uvec3 p { x, y, z };
                const HASH_TYPE v = inGrid.get(p);
                started |= (v != 0);
                hash_combine<HASH_TYPE, HASH_FUNCTION>(hash, v);
                hash = std::max(hash, (HASH_TYPE)1); // Prevent hash from accidentally becoming 0 for a non-empty input.
                hash *= started;
                outGrid.set(p, hash);
            }
        }
    }
}
template <bool PositiveAxis, int Resolution>
static HOST_DEVICE void computeGridHashY(const HashGrid<Resolution>& inGrid, HashGrid<Resolution>& outGrid)
{
    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    for (int z = 0; z < Resolution; ++z) {
        for (int x = 0; x < Resolution; ++x) {
            bool started = false;
            HASH_TYPE hash = 0;
            for (int y = start; y != end; y += step) {
                const glm::uvec3 p { x, y, z };
                const HASH_TYPE v = inGrid.get(p);
                started |= (v != 0);
                hash_combine<HASH_TYPE, HASH_FUNCTION>(hash, v);
                hash = std::max(hash, (HASH_TYPE)1); // Prevent hash from accidentally becoming 0 for a non-empty input.
                hash *= started;
                outGrid.set(p, hash);
            }
        }
    }
}
template <bool PositiveAxis, int Resolution>
static HOST_DEVICE void computeGridHashZ(const HashGrid<Resolution>& inGrid, HashGrid<Resolution>& outGrid)
{
    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    for (int x = 0; x < Resolution; ++x) {
        for (int y = 0; y < Resolution; ++y) {
            bool started = false;
            HASH_TYPE hash = 0;
            for (int z = start; z != end; z += step) {
                const glm::uvec3 p { x, y, z };
                const HASH_TYPE v = inGrid.get(p);
                started |= (v != 0);
                hash_combine<HASH_TYPE, HASH_FUNCTION>(hash, v);
                hash = std::max(hash, (HASH_TYPE)1); // Prevent hash from accidentally becoming 0 for a non-empty input.
                hash *= started;
                outGrid.set(p, hash);
            }
        }
    }
}
template <int Resolution, bool PositiveX, bool PositiveY, bool PositiveZ>
static HOST_DEVICE HashGrid<Resolution> computeGridHash(const LargeSubGrid<Resolution>& inGrid)
{
    HashGrid<Resolution> outHashGrid {};
    for (uint32_t i = 0; i < outHashGrid.voxels.size(); ++i)
        outHashGrid.set(i, inGrid.get(i));
    computeGridHashX<PositiveX>(outHashGrid, outHashGrid);
    computeGridHashY<PositiveY>(outHashGrid, outHashGrid);
    computeGridHashZ<PositiveZ>(outHashGrid, outHashGrid);
    return outHashGrid;
}

constexpr int constexpr_log2(int x)
{
    int out = -1;
    while (x) {
        x >>= 1;
        ++out;
    }
    return out;
}
template <bool PositiveAxis, int Resolution, typename Tile>
HOST_DEVICE void computeGridHashX_tile(const HashGrid<Resolution>& inGrid, HashGrid<Resolution>& outGrid, Tile&& tile)
{
    constexpr int Level = constexpr_log2(Resolution);
    constexpr int Resolution2 = Resolution * Resolution;
    constexpr uint32_t mask = (1u << Level) - 1u;
    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    for (int j = 0; j < Resolution * Resolution; j += tile.num_threads()) {
        const int i = j + tile.thread_rank();
        if (i >= Resolution2)
            continue;
        const int y = i & mask;
        const int z = (i >> Level) & mask;

        bool started = false;
        HASH_TYPE hash = 0;
        for (int x = start; x != end; x += step) {
            const glm::ivec3 p { x, y, z };
            const HASH_TYPE v = inGrid.get(p);
            started |= (v != 0);
            hash_combine<HASH_TYPE, HASH_FUNCTION>(hash, v);
            hash = std::max(hash, (HASH_TYPE)1); // Prevent hash from accidentally becoming 0 for a non-empty input.
            hash *= started;
            outGrid.set(p, hash);
        }
    }
}
template <bool PositiveAxis, int Resolution, typename Tile>
HOST_DEVICE void computeGridHashY_tile(const HashGrid<Resolution>& inGrid, HashGrid<Resolution>& outGrid, Tile&& tile)
{
    constexpr int Level = constexpr_log2(Resolution);
    constexpr int Resolution2 = Resolution * Resolution;
    constexpr uint32_t mask = (1u << Level) - 1u;
    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    for (int j = 0; j < Resolution * Resolution; j += tile.num_threads()) {
        const int i = j + tile.thread_rank();
        if (i >= Resolution2)
            continue;
        const int x = i & mask;
        const int z = (i >> Level) & mask;

        bool started = false;
        HASH_TYPE hash = 0;
        for (int y = start; y != end; y += step) {
            const glm::ivec3 p { x, y, z };
            const HASH_TYPE v = inGrid.get(p);
            started |= (v != 0);
            hash_combine<HASH_TYPE, HASH_FUNCTION>(hash, v);
            hash = std::max(hash, (HASH_TYPE)1); // Prevent hash from accidentally becoming 0 for a non-empty input.
            hash *= started;
            outGrid.set(p, hash);
        }
    }
}
template <bool PositiveAxis, int Resolution, typename Tile>
HOST_DEVICE void computeGridHashZ_tile(const HashGrid<Resolution>& inGrid, HashGrid<Resolution>& outGrid, Tile&& tile)
{
    constexpr int Level = constexpr_log2(Resolution);
    constexpr int Resolution2 = Resolution * Resolution;
    constexpr uint32_t mask = (1u << Level) - 1u;
    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    for (int j = 0; j < Resolution * Resolution; j += tile.num_threads()) {
        const int i = j + tile.thread_rank();
        if (i >= Resolution2)
            continue;
        const int x = i & mask;
        const int y = (i >> Level) & mask;

        bool started = false;
        HASH_TYPE hash = 0;
        for (int z = start; z != end; z += step) {
            const glm::ivec3 p { x, y, z };
            const HASH_TYPE v = inGrid.get(p);
            started |= (v != 0);
            hash_combine<HASH_TYPE, HASH_FUNCTION>(hash, v);
            hash = std::max(hash, (HASH_TYPE)1); // Prevent hash from accidentally becoming 0 for a non-empty input.
            hash *= started;
            outGrid.set(p, hash);
        }
    }
}
template <bool Positive0, bool Positive1, bool Positive2, int Resolution, typename Tile>
HOST_DEVICE void computeTransformedGridHash_tile(
    const LargeSubGrid<Resolution>& inGrid, const glm::ivec3& axisPermutation,
    HashGrid<Resolution>& outHashGrid, Tile&& tile)
{
    const auto threadRank = tile.thread_rank();
    constexpr uint32_t Resolution3 = Resolution * Resolution * Resolution;
    assert_always(Resolution3 % tile.num_threads() == 0);
    for (uint32_t i = 0; i < Resolution3; i += tile.num_threads()) {
        outHashGrid.set(i + threadRank, inGrid.get(i + threadRank));
    }

    tile.sync();

    if (axisPermutation.x == 0) {
        computeGridHashX_tile<Positive0>(outHashGrid, outHashGrid, tile);
    } else if (axisPermutation.x == 1) {
        computeGridHashY_tile<Positive0>(outHashGrid, outHashGrid, tile);
    } else {
        computeGridHashZ_tile<Positive0>(outHashGrid, outHashGrid, tile);
    }

    tile.sync();

    if (axisPermutation.y == 0) {
        computeGridHashX_tile<Positive1>(outHashGrid, outHashGrid, tile);
    } else if (axisPermutation.y == 1) {
        computeGridHashY_tile<Positive1>(outHashGrid, outHashGrid, tile);
    } else {
        computeGridHashZ_tile<Positive1>(outHashGrid, outHashGrid, tile);
    }

    tile.sync();

    if (axisPermutation.z == 0) {
        computeGridHashX_tile<Positive2>(outHashGrid, outHashGrid, tile);
    } else if (axisPermutation.z == 1) {
        computeGridHashY_tile<Positive2>(outHashGrid, outHashGrid, tile);
    } else {
        computeGridHashZ_tile<Positive2>(outHashGrid, outHashGrid, tile);
    }
}

template <bool PositiveAxis, int Resolution, typename Tile>
HOST_DEVICE void computeGridHashX_tile(HashGrid2D<Resolution>& inOutGrid, Tile&& tile)
{
    constexpr int Level = constexpr_log2(Resolution);
    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    for (int yBase = 0; yBase < Resolution; yBase += tile.num_threads()) {
        const int y = yBase + tile.thread_rank();
        if (y >= Resolution)
            continue;

        bool started = false;
        HASH_TYPE hash = 0;
        for (int x = start; x != end; x += step) {
            const glm::ivec2 p { x, y };
            const HASH_TYPE v = inOutGrid.get(p);
            started |= (v != 0);
            hash_combine<HASH_TYPE, HASH_FUNCTION>(hash, v);
            hash = std::max(hash, (HASH_TYPE)1); // Prevent hash from accidentally becoming 0 for a non-empty input.
            hash *= started;
            inOutGrid.set(p, hash);
        }
    }
}
template <bool PositiveAxis, int Resolution, typename Tile>
HOST_DEVICE void computeGridHashY_tile(HashGrid2D<Resolution>& inOutGrid, Tile&& tile)
{
    constexpr int Level = constexpr_log2(Resolution);
    constexpr int start = PositiveAxis ? Resolution - 1 : 0;
    constexpr int end = PositiveAxis ? -1 : Resolution;
    constexpr int step = PositiveAxis ? -1 : +1;
    for (int xBase = 0; xBase < Resolution; xBase += tile.num_threads()) {
        const int x = xBase + tile.thread_rank();
        if (x >= Resolution)
            continue;

        bool started = false;
        HASH_TYPE hash = 0;
        for (int y = start; y != end; y += step) {
            const glm::ivec2 p { x, y };
            const HASH_TYPE v = inOutGrid.get(p);
            started |= (v != 0);
            hash_combine<HASH_TYPE, HASH_FUNCTION>(hash, v);
            hash = std::max(hash, (HASH_TYPE)1); // Prevent hash from accidentally becoming 0 for a non-empty input.
            hash *= started;
            inOutGrid.set(p, hash);
        }
    }
}
template <int Resolution, typename Tile>
HOST_DEVICE void computeGridHashLayers_tile(const HashGrid2D<Resolution>& inCurrentGrid, HashGrid2D<Resolution>& inOutAccumulatorGrid, Tile&& tile)
{
    constexpr auto Resolution2 = Resolution * Resolution;
    for (int iBase = 0; iBase < Resolution2; iBase += tile.num_threads()) {
        const int i = iBase + tile.thread_rank();
        if (i >= Resolution2)
            continue;
        const auto vIn = inCurrentGrid.get(i);
        auto vAccumulator = inOutAccumulatorGrid.get(i);
        const bool started = (vAccumulator != 0 || vIn != 0);
        hash_combine<HASH_TYPE, HASH_FUNCTION>(vAccumulator, vIn);
        vAccumulator = std::max(vAccumulator, (HASH_TYPE)1); // Prevent hash from accidentally becoming 0 for a non-empty input.
        vAccumulator *= started;
        inOutAccumulatorGrid.set(i, vAccumulator);
    }
}
template <bool Positive0, bool Positive1, bool Positive2, int Resolution, typename Tile, typename F>
HOST_DEVICE void computeTransformedGridHash_tile(
    const LargeSubGrid<Resolution>& inGrid,
    HashGrid2D<Resolution>& hashGrid1, HashGrid2D<Resolution>& hashGrid2, const glm::ivec3& axisPermutation, Tile&& tile, F&& func)
{
    constexpr int Resolution2 = Resolution * Resolution;
    constexpr int Level = constexpr_log2(Resolution);
    constexpr int mask = (1u << Level) - 1u;

    constexpr int start = Positive2 ? Resolution - 1 : 0;
    constexpr int end = Positive2 ? -1 : Resolution;
    constexpr int step = Positive2 ? -1 : +1;

    // Result of std::hash<int>()(0) and std::hash<int>()(0) on MSVC.
    for (int iBase = 0; iBase < Resolution2; iBase += tile.num_threads()) {
        const int i = iBase + tile.thread_rank();
        if (i < Resolution2)
            hashGrid2.voxels.at(i) = 0;
    }
    tile.sync();

    if (axisPermutation.z == 2) {
        for (int z = start; z != end; z += step) {
            for (int iBase = 0; iBase < Resolution2; iBase += tile.num_threads()) {
                const int i = iBase + tile.thread_rank();
                if (i >= Resolution2)
                    continue;

                const int x = i & mask;
                const int y = (i >> Level) & mask;
                const glm::ivec3 voxel3D { x, y, z };
                const glm::ivec2 voxel2D { x, y };
                hashGrid1.set(voxel2D, inGrid.get(voxel3D));
            }
            tile.sync();

            if (axisPermutation.x == 0) {
                // x y z
                computeGridHashX_tile<Positive0>(hashGrid1, tile);
                tile.sync();
                computeGridHashY_tile<Positive1>(hashGrid1, tile);
            } else {
                // y x z
                computeGridHashY_tile<Positive0>(hashGrid1, tile);
                tile.sync();
                computeGridHashX_tile<Positive1>(hashGrid1, tile);
            }
            tile.sync();

            computeGridHashLayers_tile(hashGrid1, hashGrid2, tile);
            tile.sync();

            for (int iBase = 0; iBase < Resolution2; iBase += tile.num_threads()) {
                const int i = iBase + tile.thread_rank();
                if (i >= Resolution2)
                    continue;

                const int x = i & mask;
                const int y = (i >> Level) & mask;
                const glm::ivec3 voxel3D { x, y, z };
                const glm::ivec2 voxel2D { x, y };
                func(voxel3D, hashGrid2.get(voxel2D));
            }
        }
    } else if (axisPermutation.z == 1) {
        for (int y = start; y != end; y += step) {
            for (int iBase = 0; iBase < Resolution2; iBase += tile.num_threads()) {
                const int i = iBase + tile.thread_rank();
                if (i >= Resolution2)
                    continue;

                const int x = i & mask;
                const int z = (i >> Level) & mask;
                const glm::ivec3 voxel3D { x, y, z };
                const glm::ivec2 voxel2D { x, z };
                hashGrid1.set(voxel2D, inGrid.get(voxel3D));
            }
            tile.sync();

            if (axisPermutation.x == 0) {
                // x z y
                computeGridHashX_tile<Positive0>(hashGrid1, tile);
                tile.sync();
                computeGridHashY_tile<Positive1>(hashGrid1, tile);
            } else {
                // z x y
                computeGridHashY_tile<Positive0>(hashGrid1, tile);
                tile.sync();
                computeGridHashX_tile<Positive1>(hashGrid1, tile);
            }
            tile.sync();

            computeGridHashLayers_tile(hashGrid1, hashGrid2, tile);
            tile.sync();

            for (int iBase = 0; iBase < Resolution2; iBase += tile.num_threads()) {
                const int i = iBase + tile.thread_rank();
                if (i >= Resolution2)
                    continue;

                const int x = i & mask;
                const int z = (i >> Level) & mask;
                const glm::ivec3 voxel3D { x, y, z };
                const glm::ivec2 voxel2D { x, z };
                func(voxel3D, hashGrid2.get(voxel2D));
            }
        }
    } else {
        for (int x = start; x != end; x += step) {
            for (int iBase = 0; iBase < Resolution2; iBase += tile.num_threads()) {
                const int i = iBase + tile.thread_rank();
                if (i >= Resolution2)
                    continue;

                const int y = i & mask;
                const int z = (i >> Level) & mask;
                const glm::ivec3 voxel3D { x, y, z };
                const glm::ivec2 voxel2D { y, z };
                hashGrid1.set(voxel2D, inGrid.get(voxel3D));
            }
            tile.sync();

            if (axisPermutation.x == 1) {
                // y z x
                computeGridHashX_tile<Positive0>(hashGrid1, tile);
                tile.sync();
                computeGridHashY_tile<Positive1>(hashGrid1, tile);
            } else {
                // z y x
                computeGridHashY_tile<Positive0>(hashGrid1, tile);
                tile.sync();
                computeGridHashX_tile<Positive1>(hashGrid1, tile);
            }
            tile.sync();

            computeGridHashLayers_tile(hashGrid1, hashGrid2, tile);
            tile.sync();

            for (int iBase = 0; iBase < Resolution2; iBase += tile.num_threads()) {
                const int i = iBase + tile.thread_rank();
                if (i >= Resolution2)
                    continue;

                const int y = i & mask;
                const int z = (i >> Level) & mask;
                const glm::ivec3 voxel3D { x, y, z };
                const glm::ivec2 voxel2D { y, z };
                func(voxel3D, hashGrid2.get(voxel2D));
            }
        }
    }
}

HOST_DEVICE constexpr uint32_t toTransformID(const glm::bvec3& symmetry, const uint32_t permutationID)
{
    const uint32_t symmetryID = (((uint32_t)symmetry.x) << 0) | (((uint32_t)symmetry.y) << 1) | (((uint32_t)symmetry.z) << 2);
    return symmetryID | (permutationID << 3);
}

// Work-around while waiting for fetch_min to be implemented (C++26).
[[maybe_unused]] static void atomicMin64(std::atomic_uint64_t& atomic, uint64_t value)
{
    uint64_t expected = atomic.load(std::memory_order_relaxed);
    while (value < expected && !atomic.compare_exchange_weak(expected, value, std::memory_order_relaxed, std::memory_order_relaxed))
        ;
}

template <int Min, int Max, typename F>
static void templateForLoop(F f, int runTimeEnd)
{
    if (runTimeEnd == Min)
        f(std::integral_constant<int, Min>());
    if constexpr (Min != Max)
        templateForLoop<Min + 1, Max, F>(f, runTimeEnd);
}

HOST_DEVICE std::pair<uint32_t, uint32_t> decodeRangePair(uint64_t encodedPair)
{
    return { (uint32_t)encodedPair, (uint32_t)(encodedPair >> 32) };
}
HOST_DEVICE uint64_t encodeRangePair(const std::pair<uint32_t, uint32_t>& pair)
{
    return (uint64_t)pair.first | ((uint64_t)pair.second << 32);
}

// CUDA compatible (std::)lower_bound which returns an index instead of an iterator.
// https://en.cppreference.com/w/cpp/algorithm/lower_bound
template <typename T>
HOST_DEVICE uint32_t my_lower_bound(const T* elements, uint32_t numElements, const T& searchElement)
{
    uint32_t count = numElements, first = 0;
    while (count > 0) {
        uint32_t it = first;
        const uint32_t step = count / 2;
        it += step;
        if (elements[it] < searchElement) {
            first = ++it;
            count -= step + 1;
        } else
            count = step;
    }
    return first;
}
template <typename T>
HOST_DEVICE bool my_binary_search(const T* elements, uint32_t numElements, const T& searchElement, uint32_t& outIndex)
{
    outIndex = my_lower_bound(elements, numElements, searchElement);
    return (outIndex != numElements && elements[outIndex] == searchElement);
}

template <typename Key, typename Value>
class MapHost;
template <typename Key, typename Value>
class MapDevice {
public:
    HOST_DEVICE bool find(Key inSearchKey, Value& outValue)
    {
        uint32_t index;
        if (!my_binary_search(pKeys, numItems, inSearchKey, index))
            return false;
        outValue = pValues[index];
        return true;
    }

private:
    friend class MapHost<Key, Value>;
    const Key* pKeys;
    const Value* pValues;
    uint32_t numItems;
};
template <typename Key, typename Value>
class MapHost {
public:
    MapHost() = default;
    MapHost(std::span<const Key> keys, std::span<const Value> values)
        : m_keys(std::begin(keys), std::end(keys))
        , m_values(std::begin(values), std::end(values))
    {
        assert_always(keys.size() == values.size());
    }

    operator MapDevice<Key, Value>() const
    {
        MapDevice<Key, Value> out {};
        out.pKeys = thrust::raw_pointer_cast(m_keys.data());
        out.pValues = thrust::raw_pointer_cast(m_values.data());
        out.numItems = (uint32_t)m_keys.size();
        return out;
    }

private:
    thrust::device_vector<Key> m_keys;
    thrust::device_vector<Value> m_values;
};

namespace impl {
    template <typename Key, typename Value>
    struct CuckooBucket {
        static constexpr uint32_t cuckooBucketSize = 32;
        Key keys[cuckooBucketSize];
        Value values[cuckooBucketSize];
    };
    template <typename Value>
    struct CuckooBucket<uint64_t, Value> {
        static constexpr uint32_t cuckooBucketSize = 16;
        uint64_t keys[cuckooBucketSize];
        Value values[cuckooBucketSize];
    };
}

template <typename Key, typename Value, Key sentinel>
class CuckooMapHost;
template <typename Key, typename Value, Key sentinel>
class CuckooMapDevice {
public:
    DEVICE bool find(Key key, Value& value) const
    {
        const auto& bucket1 = pBuckets[hash1(key) % numBuckets];
        for (uint32_t slot = 0; slot < bucketSize; ++slot) {
            if (bucket1.keys[slot] == key) {
                value = bucket1.values[slot];
                return true;
            }
        }

        const auto& bucket2 = pBuckets[hash2(key) % numBuckets];
        for (uint32_t slot = 0; slot < bucketSize; ++slot) {
            if (bucket2.keys[slot] == key) {
                value = bucket2.values[slot];
                return true;
            }
        }

        return false;
    }

    template <typename Warp>
    DEVICE bool findForThreadAsWarp(Key key, Value& value, uint32_t outputLane, Warp warp) const
    {
        const auto threadRank = warp.thread_rank();
        if constexpr (std::is_same_v<Key, uint64_t> && bucketSize == 16) {
            const auto partialKey = uint32_t(key >> ((threadRank & 1) * 32));
            // const auto partialKey = uint32_t(key >> ((1 - (threadRank & 1)) * 32));
            const auto& bucket1 = pBuckets[hash1(key) % numBuckets];
            auto foundMask = warp.ballot(((const uint32_t*)bucket1.keys)[threadRank] == partialKey);
            foundMask = (foundMask & 0x5555'5555) & ((foundMask >> 1) & 0x5555'5555);
            if (foundMask) {
                const auto foundSlot = (__ffs(foundMask) - 1) >> 1;
                if (threadRank == outputLane)
                    value = bucket1.values[foundSlot];
                return true;
            }

            const auto& bucket2 = pBuckets[hash2(key) % numBuckets];
            foundMask = warp.ballot(((const uint32_t*)bucket2.keys)[threadRank] == partialKey);
            foundMask = (foundMask & 0x5555'5555) & ((foundMask >> 1) & 0x5555'5555);
            if (foundMask) {
                const auto foundSlot = (__ffs(foundMask) - 1) >> 1;
                if (threadRank == outputLane)
                    value = bucket2.values[foundSlot];
                return true;
            }
        } else {
            const auto& bucket1 = pBuckets[hash1(key) % numBuckets];
            auto foundMask = warp.ballot(threadRank < bucketSize && bucket1.keys[threadRank] == key);
            if (foundMask) {
                const auto foundSlot = __ffs(foundMask) - 1;
                if (threadRank == outputLane)
                    value = bucket1.values[foundSlot];
                return true;
            }

            const auto& bucket2 = pBuckets[hash2(key) % numBuckets];
            foundMask = warp.ballot(threadRank < bucketSize && bucket2.keys[threadRank] == key);
            if (foundMask) {
                const auto foundSlot = __ffs(foundMask) - 1;
                if (threadRank == outputLane)
                    value = bucket2.values[foundSlot];
                return true;
            }
        }

        return false;
    }

    template <typename Warp>
    DEVICE void insertKeyAsWarp(Key key, Warp warp)
    {
        const auto threadRank = warp.thread_rank();
        auto h1 = hash1(key);
        auto h2 = hash2(key);
        uint32_t iteration = 0;
        while (true) {
            if (++iteration > 1000)
                printf("Stuck %u", iteration);

            // Insert into the first empty slot encountered in the first bucket.
            const uint32_t bucketIdx = h1 % numBuckets;
            auto& bucket = pBuckets[bucketIdx];

            while (const uint32_t emptySlotMask = warp.ballot(threadRank < bucketSize && bucket.keys[threadRank] == sentinel)) {
                const auto selectedSlot = __ffs(emptySlotMask) - 1u;
                bool success = false;
                if (threadRank == selectedSlot)
                    success = atomicCAS(&bucket.keys[threadRank], sentinel, key) == sentinel;
                success = warp.shfl(success, selectedSlot);
                if (success)
                    return;
            }

            // When the bucket is full: insert by kicking out another item.
            const auto evictedSlot = h2 % bucketSize;
            if (threadRank == evictedSlot)
                key = atomicExch(&bucket.keys[evictedSlot], key);
            key = warp.shfl(key, evictedSlot);

            // Now find a new spot for the item we just kicked out.
            // Regularly switch h1 & h2 so we don't keep inserting into the same bucket.
            h1 = hash1(key);
            h2 = hash2(key);
            if ((h1 % numBuckets) == bucketIdx)
                std::swap(h1, h2);
        }
    }
    template <typename Warp>
    DEVICE void insertValueAsWarp(Key key, Value value, Warp warp)
    {
        const auto threadRank = warp.thread_rank();
        auto& bucket1 = pBuckets[hash1(key) % numBuckets];
        auto foundMask = warp.ballot(threadRank < bucketSize && bucket1.keys[threadRank] == key);
        if (foundMask) {
            const auto foundSlot = __ffs(foundMask) - 1;
            bucket1.values[foundSlot] = value;
            return;
        }

        auto& bucket2 = pBuckets[hash2(key) % numBuckets];
        foundMask = warp.ballot(threadRank < bucketSize && bucket2.keys[threadRank] == key);
        if (foundMask) {
            const auto foundSlot = __ffs(foundMask) - 1;
            bucket2.values[foundSlot] = value;
            return;
        }
    }

private:
    static HOST_DEVICE uint32_t hash1(Key k)
    {
        return 8121u * (uint32_t)k + 28411u;
    }
    static HOST_DEVICE uint32_t hash2(Key k)
    {
        return 1103515245u * (uint32_t)k + 12345u;
    }

private:
    friend class CuckooMapHost<Key, Value, sentinel>;
    static constexpr auto bucketSize = impl::CuckooBucket<Key, Value>::cuckooBucketSize;

    impl::CuckooBucket<Key, Value>* pBuckets;
    uint32_t numBuckets;
};
template <typename Key, typename Value, Key sentinel>
__global__ void cuckooInsertKeys_kernel(CuckooMapDevice<Key, Value, sentinel> cuckooMap, const Key* keys, size_t numItems)
{
    const auto threadBlock = cg::this_thread_block();
    const auto warp = cg::tiled_partition<threadsPerWarp>(threadBlock);
    const unsigned itemIdx = threadBlock.group_index().x * warp.meta_group_size() + warp.meta_group_rank();
    if (itemIdx < numItems)
        cuckooMap.insertKeyAsWarp(keys[itemIdx], warp);
}
template <typename Key, typename Value, Key sentinel>
__global__ void cuckooInsertValues_kernel(CuckooMapDevice<Key, Value, sentinel> cuckooMap, const Key* keys, const Value* values, size_t numItems)
{
    const auto threadBlock = cg::this_thread_block();
    const auto warp = cg::tiled_partition<threadsPerWarp>(threadBlock);
    const unsigned itemIdx = threadBlock.group_index().x * warp.meta_group_size() + warp.meta_group_rank();
    if (itemIdx < numItems)
        cuckooMap.insertValueAsWarp(keys[itemIdx], values[itemIdx], warp);
}

template <typename Key, typename Value, Key sentinel>
class CuckooMapHost {
public:
    CuckooMapHost() = default;
    CuckooMapHost(std::span<const Key> keys, std::span<const Value> values)
    {
        assert_always(keys.size() == values.size());
        constexpr int workGroupSize = 64;
        static_assert(workGroupSize % threadsPerWarp == 0);
        const int numWorkGroups = computeNumWorkGroups(keys.size() * threadsPerWarp, workGroupSize);

        constexpr auto loadFactor = 2;
        m_numBuckets = (loadFactor * (uint32_t)keys.size()) / bucketSize + 2;
        m_buckets_device.resize(m_numBuckets);
        const auto keys_device = thrust::device_vector<Key>(std::begin(keys), std::end(keys));
        const auto values_device = thrust::device_vector<Value>(std::begin(values), std::end(values));
        cuckooInsertKeys_kernel<Key, Value, sentinel><<<numWorkGroups, workGroupSize>>>(*this, thrust::raw_pointer_cast(keys_device.data()), keys.size());
        CUDA_CHECK_ERROR();
        cuckooInsertValues_kernel<Key, Value, sentinel><<<numWorkGroups, workGroupSize>>>(*this, thrust::raw_pointer_cast(keys_device.data()), thrust::raw_pointer_cast(values_device.data()), keys.size());
        CUDA_CHECK_ERROR();
    }

    operator CuckooMapDevice<Key, Value, sentinel>()
    {
        CuckooMapDevice<Key, Value, sentinel> out {};
        out.pBuckets = thrust::raw_pointer_cast(m_buckets_device.data());
        out.numBuckets = (uint32_t)m_numBuckets;
        return out;
    }
    operator CuckooMapDevice<Key, Value, sentinel>() const
    {
        CuckooMapDevice<Key, Value, sentinel> out {};
        out.pBuckets = thrust::raw_pointer_cast(m_buckets_device.data());
        out.numBuckets = (uint32_t)m_numBuckets;
        return out;
    }

    size_t memoryUsedInBytes() const
    {
        return m_buckets_device.size() * sizeof(Bucket);
    }

private:
    using DeviceType = CuckooMapDevice<Key, Value, sentinel>;
    using Bucket = impl::CuckooBucket<Key, Value>;
    static constexpr auto bucketSize = Bucket::cuckooBucketSize;

    uint32_t m_numBuckets;
    thrust::device_vector<Bucket> m_buckets_device;
};

// The hash table insertion code requires warps (32 threads) to cooperate.
// When processing 4x4x4 level a single "slice" is 4x4=16 threads.
// Thus, at that level, we need to process the entire grid at once.
namespace impl {
#if REDUCED_SHARED_MEMORY
    template <int Resolution>
    struct hash_grid_data {
        using type = std::pair<HashGrid2D<Resolution>, HashGrid2D<Resolution>>;
    };
    template <>
    struct hash_grid_data<4> {
        using type = HashGrid<4>;
    };
#else
    template <int Resolution>
    struct hash_grid_data {
        using type = HashGrid<Resolution>;
    };
#endif
    template <int Resolution>
    using hash_grid_data_t = typename hash_grid_data<Resolution>::type;
}

template <bool PositiveX, bool PositiveY, bool PositiveZ, int WorkGroupSize, int Resolution>
__global__ void findMatches_kernel(
    const LargeSubGrid<Resolution>* pInGrids, uint32_t numGrids, uint32_t gridOffset, // All grids stored in a binary format.
    const TransformPointer* pInPointers, // Pointers to the grids; one for each symmetry. Sorted by hash of the grids.
    // MapDevice<uint64_t, std::pair<uint32_t, uint32_t>> pointerRanges, // Look-up table: hash => range of pInPointers with the same hash.
    CuckooMapDevice<HASH_TYPE, std::pair<uint32_t, uint32_t>, 0> pointerRanges, // Look-up table: hash => range of pInPointers with the same hash.
    uint64_t* pOutParents, // Output: 64-bit encoded output pointers.
    TransformDAGConfig config) // Settings...
{
    const auto threadBlock = cg::this_thread_block();
    const auto warp = cg::tiled_partition<threadsPerWarp>(threadBlock);
    const auto threadRank = warp.thread_rank();
    // assert(warp.group_size() == 32);

#if PROCESS_PER_WARP
    const unsigned passItemIdx = threadBlock.group_index().x * warp.meta_group_size() + warp.meta_group_rank();
#else
    const unsigned passItemIdx = threadBlock.group_index().x;
#endif
    if (passItemIdx >= numGrids)
        return;

    const unsigned itemIdx = passItemIdx + gridOffset;

    constexpr int Level = constexpr_log2(Resolution);
    constexpr int Resolution3 = Resolution * Resolution * Resolution;
    assert_always(Resolution3 % warp.num_threads() == 0); // Required for the loop over voxels to be correct (out-of-bounds access otherwise).
    const glm::ivec3 anchorPoint { PositiveX ? 0 : Resolution - 1, PositiveY ? 0 : Resolution - 1, PositiveZ ? 0 : Resolution - 1 };
    const glm::bvec3 direction { PositiveX, PositiveY, PositiveZ };

#if PROCESS_PER_WARP
    __shared__ impl::hash_grid_data_t<Resolution> hashGridsDataShared[WorkGroupSize / threadsPerWarp];
    auto& hashGridsData = hashGridsDataShared[warp.meta_group_rank()];
#else
    __shared__ impl::hash_grid_data_t<Resolution> hashGridsData;
#endif

    // For each way that we can permute the order of the axis.
    const auto& grid = pInGrids[passItemIdx];
    for (uint8_t axisPermutationID = 0; axisPermutationID < TransformPointer::NumUniqueAxisPermutations; ++axisPermutationID) {
        const auto axisPermutation = TransformPointer::decodeAxisPermutation(axisPermutationID);

        const auto visitHashVoxel = [&](const glm::ivec3& hashVoxel, HASH_TYPE hash) {
            // Reorder the axis of the grid accordingly, and then compute the hash for each voxel.
            std::pair<uint32_t, uint32_t> pointersRange { 0, 0 };
#if 1
            uint32_t validHashMask = warp.ballot(hash != 0);
            for (uint32_t selectedLane = 0; selectedLane < warp.num_threads(); ++selectedLane) {
                // if (warp.thread_rank() > 15)
                if ((validHashMask >> selectedLane) & 0b1) {
                    const auto laneHash = warp.shfl(hash, selectedLane);
                    pointerRanges.findForThreadAsWarp(laneHash, pointersRange, selectedLane, warp);
                }
            }

#else
            if (hash != 0)
                pointerRanges.find(hash, pointersRange);
#endif
            if (pointersRange.first == pointersRange.second) // Empty range.
                return;

            const glm::ivec3 voxel = applyAxisPermutation(hashVoxel, axisPermutation);
            for (uint32_t i = pointersRange.first; i < pointersRange.second; ++i) {
                const auto& matchingItem = pInPointers[i];
                const auto matchingItemIdx = matchingItem.ptr;
                // (Potentially) matching grid: a grid that can be created by transforming the current grid.
                auto& potentialChild = pOutParents[matchingItemIdx];
#if !SORT_BY_VOXEL_COUNT
                if (matchingItemIdx == itemIdx) // Don't match to self.
                    continue;
#endif
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
                if (potentialChild <= encodedShiftPointer)
                    continue;

                atomicMin(&potentialChild, encodedShiftPointer);
            } // for all potential matches
        };

#if PROCESS_PER_WARP
#define PROCESS_UNIT warp
#else
#define PROCESS_UNIT threadBlock
#endif
        if constexpr (std::is_same_v<impl::hash_grid_data_t<Resolution>, HashGrid<Resolution>>) {
            // Process entire grid at once.
            computeTransformedGridHash_tile<PositiveX, PositiveY, PositiveZ>(grid, axisPermutation, hashGridsData, PROCESS_UNIT);
            PROCESS_UNIT.sync();

            // Loop over all voxels in the grid.
            for (uint32_t baseVoxelIdx = 0; baseVoxelIdx < Resolution3; baseVoxelIdx += PROCESS_UNIT.num_threads()) {
                //  Get the hash of what the grid looks like from the selected voxel in the selected positive/negative x/y/z direction.
                constexpr uint32_t mask = (1u << Level) - 1u;
                const uint32_t voxelIdx = baseVoxelIdx + PROCESS_UNIT.thread_rank();
                const glm::ivec3 hashVoxel { (voxelIdx >> (0 * Level)) & mask, (voxelIdx >> (1 * Level)) & mask, (voxelIdx >> (2 * Level)) & mask };
                const auto hash = hashGridsData.get(hashVoxel);
                visitHashVoxel(hashVoxel, hash);
            }
        } else {
            // Process plane-by-plane
            PROCESS_UNIT.sync(); // Make sure we're finished with hashGrid2D_2 before continuing to the next iteration.
            computeTransformedGridHash_tile<PositiveX, PositiveY, PositiveZ>(grid, hashGridsData.first, hashGridsData.second, axisPermutation, PROCESS_UNIT, visitHashVoxel);
        }
    } // axis permutation
}

template <typename Key, typename Value>
void sort_by_key(std::span<Key> keys, std::span<Value> values)
{
    assert_always(keys.size() == values.size());
    std::vector<uint32_t> indices(keys.size());
    std::iota(std::begin(indices), std::end(indices), 0);
    std::sort(execution_policy, std::begin(indices), std::end(indices), [&](uint32_t lhs, uint32_t rhs) { return keys[lhs] < keys[rhs]; });

    const auto applySort = [&indices]<typename T>(std::span<T> items) {
        std::vector<T> itemsCopy(items.size());
        std::copy(execution_policy, std::begin(items), std::end(items), std::begin(itemsCopy));
        std::transform(execution_policy, std::begin(indices), std::end(indices), std::begin(items), [&itemsCopy](uint32_t index) { return itemsCopy[index]; });
    };
    applySort(keys);
    applySort(values);
}

template <bool PositiveX, bool PositiveY, bool PositiveZ, int WorkGroupSize, int Resolution>
__global__ void computeGridHashCorner_kernel(
    const LargeSubGrid<Resolution>* pInGrids, uint32_t numGrids, // All grids stored in a binary format.
    uint64_t* pOutHashes)
{
    const auto threadBlock = cg::this_thread_block();
    const auto warp = cg::tiled_partition<threadsPerWarp>(threadBlock);
    const auto threadRank = warp.thread_rank();
    // assert(warp.group_size() == 32);

#if PROCESS_PER_WARP
    const unsigned itemIdx = threadBlock.group_index().x * warp.meta_group_size() + warp.meta_group_rank();
#else
    const unsigned itemIdx = threadBlock.group_index().x;
#endif
    if (itemIdx >= numGrids)
        return;

    constexpr int Level = constexpr_log2(Resolution);
    constexpr int Resolution3 = Resolution * Resolution * Resolution;
    assert_always(Resolution3 % warp.num_threads() == 0); // Required for the loop over voxels to be correct (out-of-bounds access otherwise).
    const glm::ivec3 anchorPoint { PositiveX ? 0 : Resolution - 1, PositiveY ? 0 : Resolution - 1, PositiveZ ? 0 : Resolution - 1 };
    //const glm::bvec3 direction { PositiveX, PositiveY, PositiveZ };

#if PROCESS_PER_WARP
    __shared__ impl::hash_grid_data_t<Resolution> hashGridsDataShared[WorkGroupSize / threadsPerWarp];
    auto& hashGridsData = hashGridsDataShared[warp.meta_group_rank()];
#else
    __shared__ impl::hash_grid_data_t<Resolution> hashGridsData;
#endif

    // For each way that we can permute the order of the axis.
    const auto& grid = pInGrids[itemIdx];

    const auto visitHashVoxel = [&](const glm::ivec3& voxel, HASH_TYPE hash) {
        if (voxel == anchorPoint)
            pOutHashes[itemIdx] = hash;
    };

#if PROCESS_PER_WARP
#define PROCESS_UNIT warp
#else
#define PROCESS_UNIT threadBlock
#endif
    if constexpr (std::is_same_v<impl::hash_grid_data_t<Resolution>, HashGrid<Resolution>>) {
        // Process entire grid at once.
        computeTransformedGridHash_tile<PositiveX, PositiveY, PositiveZ>(grid, glm::ivec3(0, 1, 2), hashGridsData, PROCESS_UNIT);
        PROCESS_UNIT.sync();

        // Loop over all voxels in the grid.
        for (uint32_t baseVoxelIdx = 0; baseVoxelIdx < Resolution3; baseVoxelIdx += PROCESS_UNIT.num_threads()) {
            //  Get the hash of what the grid looks like from the selected voxel in the selected positive/negative x/y/z direction.
            constexpr uint32_t mask = (1u << Level) - 1u;
            const uint32_t voxelIdx = baseVoxelIdx + PROCESS_UNIT.thread_rank();
            const glm::ivec3 hashVoxel { (voxelIdx >> (0 * Level)) & mask, (voxelIdx >> (1 * Level)) & mask, (voxelIdx >> (2 * Level)) & mask };
            const auto hash = hashGridsData.get(hashVoxel);
            visitHashVoxel(hashVoxel, hash);
        }
    } else {
        // Process plane-by-plane
        PROCESS_UNIT.sync(); // Make sure we're finished with hashGrid2D_2 before continuing to the next iteration.
        computeTransformedGridHash_tile<PositiveX, PositiveY, PositiveZ>(grid, hashGridsData.first, hashGridsData.second, glm::ivec3(0, 1, 2), PROCESS_UNIT, visitHashVoxel);
    }
}

template <size_t Resolution, bool PositiveX, bool PositiveY, bool PositiveZ>
struct ComputeCorner {
    glm::ivec3 anchorPoint;

    HOST_DEVICE HASH_TYPE operator()(const LargeSubGrid<Resolution>& subGrid) const
    {
        return computeGridHash<Resolution, PositiveX, PositiveY, PositiveZ>(subGrid).get(anchorPoint);
    }
};

// Given a collection of items (EditSubGrid or EditNode) and their representation as a dense voxel grid,
//  find duplicates w.r.t. translation, symmetry and axis permutation. Returns the list of unique items and
//  the mapping from original indices into this new array (outPrevLevelMapping).
template <size_t Level, typename Item>
std::vector<Item> findDuplicatesAmongFlattenedItems_gpu(std::span<const Item> items, std::span<const LargeSubGrid<(1u << Level)>> grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config)
{
    static constexpr size_t Resolution = 1u << Level;
    static constexpr size_t Resolution3 = Resolution * Resolution * Resolution;
    assert_always(items.size() == grids.size());
    constexpr auto targetMemoryUsage = 8llu * 1024llu * 1024llu * 1024llu;

#if SORT_BY_VOXEL_COUNT
    std::vector<uint32_t> itemIndicesByVoxelCount(items.size());
    {
        std::iota(std::begin(itemIndicesByVoxelCount), std::end(itemIndicesByVoxelCount), 0u);
        std::vector<uint16_t> itemVoxelCounts(items.size());
        std::transform(execution_policy, std::begin(itemIndicesByVoxelCount), std::end(itemIndicesByVoxelCount), std::begin(itemVoxelCounts),
            [&](uint32_t itemIdx) {
                return (uint16_t)grids[itemIdx].popcount();
            });
        std::stable_sort(execution_policy, std::begin(itemIndicesByVoxelCount), std::end(itemIndicesByVoxelCount), [&](uint32_t lhs, uint32_t rhs) { return itemVoxelCounts[lhs] > itemVoxelCounts[rhs]; });
        // std::mt19937 rng {};
        // std::shuffle(std::begin(itemIndicesByVoxelCount), std::end(itemIndicesByVoxelCount), rng);
    }
#endif

    // Generate hashes of what each grid looks like from each of the 8 corner of the grid.
    spdlog::info("Computing anchor point hashes");
    std::vector<HASH_TYPE> hashes(8 * grids.size());
    std::vector<TransformPointer> shiftPointers(8 * grids.size());
#if COMPARE_WITH_CPU
    // robin_hood::unordered_flat_map<size_t, std::vector<ShiftPointer>, HASH_FUNCTION> gridHashes_cpu;
    std::unordered_map<HASH_TYPE, std::vector<TransformPointer>, HASH_FUNCTION> gridHashes_cpu;
#endif
    const auto initializeLookUpTable = [&]<bool PositiveX, bool PositiveY, bool PositiveZ>(std::bool_constant<PositiveX>, std::bool_constant<PositiveY>, std::bool_constant<PositiveZ>) {
        //const glm::ivec3 anchorPoint { PositiveX ? 0 : Resolution - 1, PositiveY ? 0 : Resolution - 1, PositiveZ ? 0 : Resolution - 1 };
        const glm::bvec3 direction { PositiveX, PositiveY, PositiveZ };

        // Parallel hash computation.
        spdlog::info("Direction ({}, {}, {})", PositiveX, PositiveY, PositiveZ);
        const auto start = voxcom::morton_encode32(glm::uvec3(direction)) * grids.size();
#if 1
        // While it seems like a good idea to move this to the GPU, we quickly run out of memory.
        // I suspect this is due to the use of "local" memory (spilling to global memory) which is never returned back to cudaMalloc.
        constexpr size_t maxBatchSize = targetMemoryUsage / (sizeof(LargeSubGrid<Resolution>) + sizeof(HASH_TYPE));
        for (size_t batchStart = 0; batchStart < grids.size(); batchStart += maxBatchSize) {
            const auto batchEnd = std::min(batchStart + maxBatchSize, grids.size());
            const auto batchSize = batchEnd - batchStart;
            thrust::device_vector<LargeSubGrid<Resolution>> grids_device(std::begin(grids) + batchStart, std::begin(grids) + batchEnd);
            thrust::device_vector<HASH_TYPE> hashes_device(grids_device.size());

            // For each item/grid.
#if PROCESS_PER_WARP
            constexpr int workGroupSize = 64;
            static_assert(workGroupSize % threadsPerWarp == 0);
            const int numWorkGroups = computeNumWorkGroups(batchSize * threadsPerWarp, workGroupSize);
#else
            constexpr int workGroupSize = Resolution <= 4 ? 32 : Resolution * Resolution;
            const int numWorkGroups = (int)batchSize;
#endif
            computeGridHashCorner_kernel<PositiveX, PositiveY, PositiveZ, workGroupSize, Resolution><<<numWorkGroups, workGroupSize>>>(
                thrust::raw_pointer_cast(grids_device.data()), (uint32_t)grids_device.size(), thrust::raw_pointer_cast(hashes_device.data()));

            //thrust::transform(std::begin(grids_device), std::end(grids_device), std::begin(hashes_device), ComputeCorner<Resolution, PositiveX, PositiveY, PositiveZ> { .anchorPoint = anchorPoint });
            thrust::copy(std::begin(hashes_device), std::end(hashes_device), std::begin(hashes) + start + batchStart);
        }
#else
        std::transform(execution_policy, std::begin(grids), std::end(grids), std::begin(hashes) + start,
            [&](const LargeSubGrid<Resolution>& subGrid) { return computeGridHash<Resolution, PositiveX, PositiveY, PositiveZ>(subGrid).get(anchorPoint); });
#endif
        for (uint32_t itemIdx = 0; itemIdx < grids.size(); ++itemIdx) {
            shiftPointers[start + itemIdx] = TransformPointer::create(itemIdx, direction);
#if COMPARE_WITH_CPU
            gridHashes_cpu[hashes[start + itemIdx]].push_back(TransformPointer::create(itemIdx, direction));
#endif
        }
    };
    initializeLookUpTable(std::bool_constant<true>(), std::bool_constant<true>(), std::bool_constant<true>());
    initializeLookUpTable(std::bool_constant<true>(), std::bool_constant<true>(), std::bool_constant<false>());
    initializeLookUpTable(std::bool_constant<true>(), std::bool_constant<false>(), std::bool_constant<true>());
    initializeLookUpTable(std::bool_constant<true>(), std::bool_constant<false>(), std::bool_constant<false>());
    initializeLookUpTable(std::bool_constant<false>(), std::bool_constant<true>(), std::bool_constant<true>());
    initializeLookUpTable(std::bool_constant<false>(), std::bool_constant<true>(), std::bool_constant<false>());
    initializeLookUpTable(std::bool_constant<false>(), std::bool_constant<false>(), std::bool_constant<true>());
    initializeLookUpTable(std::bool_constant<false>(), std::bool_constant<false>(), std::bool_constant<false>());

    // Sort hashes & shiftPointers by hash.
    spdlog::info("Sorting hashes");
    sort_by_key<HASH_TYPE, TransformPointer>(hashes, shiftPointers);

    spdlog::info("Constructing hash table");
    // Hashes may occur multiple times. Compact the hashes array and store for each one the start&end index in the shiftPointers array.
    // MapHost<size_t, std::pair<uint32_t, uint32_t>> pointerRanges;
    CuckooMapHost<HASH_TYPE, std::pair<uint32_t, uint32_t>, 0> pointerRangesCuckoo;
    {
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        std::vector<HASH_TYPE> rangeHashes;
        uint32_t start = 0;
        HASH_TYPE prevHash = hashes[0];
        for (uint32_t i = 0; i < hashes.size(); ++i) {
            if (hashes[i] != prevHash) {
                ranges.push_back({ start, i });
                rangeHashes.push_back(prevHash);
                prevHash = hashes[i];
                start = i;
            }
        }
        ranges.push_back({ start, (uint32_t)hashes.size() });
        rangeHashes.push_back(hashes.back());
        spdlog::info("rangeHashes.size() = {}", rangeHashes.size());
        assert_always(rangeHashes.size() == ranges.size());
#if COMPARE_WITH_CPU
        assert_always(rangeHashes.size() == gridHashes_cpu.size());
#endif
        // pointerRanges = MapHost<size_t, std::pair<uint32_t, uint32_t>>(rangeHashes, ranges);
        pointerRangesCuckoo = CuckooMapHost<HASH_TYPE, std::pair<uint32_t, uint32_t>, 0> { rangeHashes, ranges };
    }

    spdlog::info("Searching for matches");
    // Generate hashes for each voxel of the grid, storing what the grid looks like from that voxel of the grid.
    // This is repeated for all eight directions that we can look into (positive/negative x/y/z).
    // We search the previously generated hash table for grids that looked the same, as seen from any voxel.
    //
    // The search may lead to one grid having multiple parents from which it can be constructed.
    // We decide to store the one that appears first in the array of items/grids. This allows groups of grids which
    // can all be transformed into each-other, to all pick the same parent.
    const uint64_t encodedSentinel = TransformPointer::sentinel().encodeFixed64(Level);

    using clock = std::chrono::high_resolution_clock;
    // Initialize the arrays storing for each item/grid, how it can be reconstructed from another grid.
    thrust::device_vector<TransformPointer> shiftPointers_device(std::begin(shiftPointers), std::end(shiftPointers));
    thrust::device_vector<uint64_t> encodedParents_device(grids.size(), encodedSentinel);

    const auto gridsMemoryUsage = grids.size() * sizeof(LargeSubGrid<Resolution>);
    const auto shiftPointersMemoryUsage = shiftPointers.size() * sizeof(TransformPointer);
    const auto encodedParentsMemoryUsage = grids.size() * sizeof(uint64_t);
    spdlog::info("grids memory usage = {}MiB", gridsMemoryUsage >> 20);
    spdlog::info("shiftPointers memory usage = {}MiB", shiftPointersMemoryUsage >> 20);
    spdlog::info("encodedParents usage = {}MiB", encodedParentsMemoryUsage >> 20);

    const auto gpuMemoryUsed = shiftPointersMemoryUsage + encodedParentsMemoryUsage + pointerRangesCuckoo.memoryUsedInBytes();
    if (gpuMemoryUsed > targetMemoryUsage)
        spdlog::warn("Running out of GPU memory ({}MiB used out of {}MiB)!", gpuMemoryUsed >> 20, targetMemoryUsage >> 20);
    const size_t gridsPerPass = std::min((targetMemoryUsage - gpuMemoryUsed) / sizeof(LargeSubGrid<Resolution>), 2llu * 1024llu * 1024llu * 1024llu); // Min 2GB of memory
    spdlog::info("Allocating GPU batches of {} MiB", (gridsPerPass * sizeof(LargeSubGrid<Resolution>)) >> 20);

    spdlog::info("GPU LAUNCH");
    for (size_t batchStart = 0; batchStart < grids.size(); batchStart += gridsPerPass) {
        spdlog::info("batch {} / {}", 1 + (batchStart / gridsPerPass), (grids.size() - 1) / gridsPerPass + 1);
        const size_t batchEnd = std::min(batchStart + gridsPerPass, grids.size());
        const size_t gridsInBatch = batchEnd - batchStart;
        std::vector<LargeSubGrid<Resolution>> batchGrids(batchEnd - batchStart);
        for (size_t i = 0; i < batchGrids.size(); ++i) {
#if SORT_BY_VOXEL_COUNT
            batchGrids[i] = grids[itemIndicesByVoxelCount[batchStart + i]];
#else
            batchGrids[i] = grids[batchStart + i];
#endif
        }
        thrust::device_vector<LargeSubGrid<Resolution>> batchGrids_device(std::begin(batchGrids), std::end(batchGrids));

        const auto searchDirections_gpu = [&]<bool PositiveX, bool PositiveY, bool PositiveZ>(std::bool_constant<PositiveX>, std::bool_constant<PositiveY>, std::bool_constant<PositiveZ>) {
            spdlog::info("{} grids; {} voxels; {} visits", gridsInBatch, gridsInBatch * Resolution3, gridsInBatch * Resolution3 * 6); // 6 axis permutations.
            spdlog::info("Direction ({}, {}, {})", PositiveX, PositiveY, PositiveZ);
            // For each item/grid.
#if PROCESS_PER_WARP
            constexpr int workGroupSize = 64;
            static_assert(workGroupSize % threadsPerWarp == 0);
            const int numWorkGroups = computeNumWorkGroups(gridsInBatch * threadsPerWarp, workGroupSize);
#else
            constexpr int workGroupSize = Resolution <= 4 ? 32 : Resolution * Resolution;
            const int numWorkGroups = (int)gridsInBatch;
#endif
            cudaDeviceSynchronize();
            spdlog::info("numWorkGroups = {}, workGroupSize = {}", numWorkGroups, workGroupSize);
            const auto t0 = clock::now();
            findMatches_kernel<PositiveX, PositiveY, PositiveZ, workGroupSize, Resolution><<<numWorkGroups, workGroupSize>>>(
                thrust::raw_pointer_cast(batchGrids_device.data()), uint32_t(gridsInBatch), (uint32_t)batchStart,
                thrust::raw_pointer_cast(shiftPointers_device.data()),
                // pointerRanges,
                pointerRangesCuckoo,
                thrust::raw_pointer_cast(encodedParents_device.data()),
                config);
            cudaDeviceSynchronize();
            const auto t1 = clock::now();
            const auto numBatchPermutedVoxels = gridsInBatch * Resolution3 * TransformPointer::NumUniqueAxisPermutations;
            const auto batchSearchTime = std::chrono::duration<double>(t1 - t0).count();
            spdlog::info("findMatches_kernel took {} ({:.2f} MiB/s)",
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0),
                double(numBatchPermutedVoxels * sizeof(HASH_TYPE)) / batchSearchTime / 1024 / 1024);
            CUDA_CHECK_ERROR();
        };
        searchDirections_gpu(std::bool_constant<true>(), std::bool_constant<true>(), std::bool_constant<true>());
        searchDirections_gpu(std::bool_constant<true>(), std::bool_constant<true>(), std::bool_constant<false>());
        searchDirections_gpu(std::bool_constant<true>(), std::bool_constant<false>(), std::bool_constant<true>());
        searchDirections_gpu(std::bool_constant<true>(), std::bool_constant<false>(), std::bool_constant<false>());
        searchDirections_gpu(std::bool_constant<false>(), std::bool_constant<true>(), std::bool_constant<true>());
        searchDirections_gpu(std::bool_constant<false>(), std::bool_constant<true>(), std::bool_constant<false>());
        searchDirections_gpu(std::bool_constant<false>(), std::bool_constant<false>(), std::bool_constant<true>());
        searchDirections_gpu(std::bool_constant<false>(), std::bool_constant<false>(), std::bool_constant<false>());
    }
    cudaDeviceSynchronize();
    spdlog::info("GPU FINISH");

    // thrust::device_vector<TransformPointer> parents_device(grids.size());
    // thrust::transform(std::begin(encodedParents_device), std::end(encodedParents_device), std::begin(parents_device),
    //     [] __device__(uint64_t encodedShiftPointer) {
    //         return TransformPointer::decodeFixed64(encodedShiftPointer, Level);
    //     });
    std::vector<uint64_t> encodedParents(grids.size());
    thrust::copy(std::begin(encodedParents_device), std::end(encodedParents_device), std::begin(encodedParents));
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

    // While a matching hash gives a very high likely hood of the two grids matching, there is a chance
    // that two different grids produce the same hash code (this is called a "hash collision").
    // To ensure correctness we transform the current grid and check whether it is indeed equivalent
    // to the matching "child" grid. This step is expensive but is required to prevent rare failures.
    std::atomic_size_t numHashCollisions = 0;
    std::transform(execution_policy, std::begin(grids), std::end(grids), std::begin(parents), std::begin(parents),
        [&](const LargeSubGrid<Resolution>& outGrid, const TransformPointer& shiftPointer) {
            if (shiftPointer == TransformPointer::sentinel())
                return shiftPointer;

            const auto& inGrid = grids[shiftPointer.ptr];
            const auto permutedGrid = permuteAxisOfSubGrid<Resolution>(inGrid, shiftPointer.getAxisPermutation());
            const auto permutedSymmetryGrid = applySymmetryToSubGrid<Resolution>(permutedGrid, shiftPointer.getSymmetry()); // <=== THIS IS EXPENSIVE.
            const auto permutedSymmetryShiftedGrid = translateSubGrid<Resolution>(permutedSymmetryGrid, shiftPointer.getTranslation()); // <=== THIS IS EXPENSIVE.
            if (permutedSymmetryShiftedGrid != outGrid) {
                // spdlog::warn("hash collision detected");
                numHashCollisions.fetch_add(1);
                return TransformPointer::sentinel();
            }
            return shiftPointer;
        });
    spdlog::info("Detected {}/{} hash collisions ({:.2}%)", numHashCollisions.load(), grids.size(), 100.0 * numHashCollisions.load() / grids.size());

#if COMPARE_WITH_CPU
    {
        // Initialize the arrays storing for each item/grid, how it can be reconstructed from another grid.
        std::vector<std::atomic_uint64_t> encodedParents_cpu(grids.size()); // Pointers to other grids for which the transformation is not invertible.
        std::for_each(execution_policy, std::begin(encodedParents_cpu), std::end(encodedParents_cpu), [=](std::atomic_uint64_t& atomicU64) { atomicU64.store(encodedSentinel, std::memory_order::relaxed); });

        const auto searchDirections_cpu = [&]<bool PositiveX, bool PositiveY, bool PositiveZ>(std::bool_constant<PositiveX>, std::bool_constant<PositiveY>, std::bool_constant<PositiveZ>) {
            spdlog::info("Direction ({}, {}, {})", PositiveX, PositiveY, PositiveZ);
            const glm::ivec3 anchorPoint { PositiveX ? 0 : Resolution - 1, PositiveY ? 0 : Resolution - 1, PositiveZ ? 0 : Resolution - 1 };
            const glm::bvec3 direction { PositiveX, PositiveY, PositiveZ };

            // For each item/grid.
            std::vector<size_t> itemIndices(items.size()); // THIS IS STUPID...
            std::iota(std::begin(itemIndices), std::end(itemIndices), 0);
            std::for_each(execution_policy, std::begin(itemIndices), std::end(itemIndices),
                [&](size_t itemIdx) {
                    const auto& grid = grids[itemIdx];
                    // For each way that we can permute the order of the axis.
                    for (uint8_t permutationID = 0; permutationID < TransformPointer::NumUniqueAxisPermutations; ++permutationID) {
                        // Reorder the axis of the grid accordingly, and then compute the hash for each voxel.
                        const auto axisPermutation = TransformPointer::decodeAxisPermutation(permutationID);
                        const auto& axisPermutedGrid = permuteAxisOfSubGrid<Resolution>(grid, axisPermutation);
                        const auto hashGrid = computeGridHash<Resolution, PositiveX, PositiveY, PositiveZ>(axisPermutedGrid);

                        // Loop over all voxels in the grid.
                        const auto invAxisPermutation = invertPermutation(axisPermutation);
                        for (uint32_t voxelIdx = 0; voxelIdx < Resolution * Resolution * Resolution; ++voxelIdx) {
                            const glm::ivec3 voxel = morton_decode32<3>(voxelIdx);
                            //  Get the hash of what the grid looks like from the selected voxel in the selected positive/negative x/y/z direction.
                            const auto hash = hashGrid.get(voxel);
                            if (hash == 0) // A hash of 0 indicates that the grid, as seen from this voxel, is empty.
                                continue;

                            // Find the list of original grids (seen from any of their 8 corners) that look the same as the current grid from this voxel.
                            const auto matchesIter = gridHashes_cpu.find(hash);
                            if (matchesIter == std::end(gridHashes_cpu))
                                continue;

                            for (const auto& matchingItem : matchesIter->second) {
                                const auto matchingItemIdx = matchingItem.ptr;
                                // Potential matching "child" grid which can be created by transforming the current grid.
                                auto& potentialChild = encodedParents_cpu[matchingItemIdx];
                                if (matchingItemIdx == itemIdx) // Don't match to self.
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
                                if (potentialChild.load(std::memory_order_relaxed) <= encodedShiftPointer)
                                    continue;

                                // While a matching hash gives a very high likely hood of the two grids matching, there is a chance
                                // that two different grids produce the same hash code (this is called a "hash collision").
                                // To ensure correctness we transform the current grid and check whether it is indeed equivalent
                                // to the matching "child" grid. This step is expensive but is required to prevent rare failures.
                                const auto permutedGrid = permuteAxisOfSubGrid<Resolution>(grid, shiftPointer.getAxisPermutation());
                                const auto permutedSymmetryGrid = applySymmetryToSubGrid<Resolution>(permutedGrid, shiftPointer.getSymmetry()); // <=== THIS IS EXPENSIVE.
                                const auto permutedSymmetryShiftedGrid = translateSubGrid<Resolution>(permutedSymmetryGrid, shiftPointer.getTranslation()); // <=== THIS IS EXPENSIVE.
                                if (grids[matchingItemIdx] != permutedSymmetryShiftedGrid)
                                    continue;

                                atomicMin64(potentialChild, encodedShiftPointer);
                            } // for all potential matches
                        } // voxelIdx
                    } // axis permutation
                }); // itemIdx
        };
        searchDirections_cpu(std::bool_constant<true>(), std::bool_constant<true>(), std::bool_constant<true>());
        searchDirections_cpu(std::bool_constant<true>(), std::bool_constant<true>(), std::bool_constant<false>());
        searchDirections_cpu(std::bool_constant<true>(), std::bool_constant<false>(), std::bool_constant<true>());
        searchDirections_cpu(std::bool_constant<true>(), std::bool_constant<false>(), std::bool_constant<false>());
        searchDirections_cpu(std::bool_constant<false>(), std::bool_constant<true>(), std::bool_constant<true>());
        searchDirections_cpu(std::bool_constant<false>(), std::bool_constant<true>(), std::bool_constant<false>());
        searchDirections_cpu(std::bool_constant<false>(), std::bool_constant<false>(), std::bool_constant<true>());
        searchDirections_cpu(std::bool_constant<false>(), std::bool_constant<false>(), std::bool_constant<false>());

        std::vector<TransformPointer> parents_cpu(grids.size());
        std::transform(execution_policy, std::begin(encodedParents_cpu), std::end(encodedParents_cpu), std::begin(parents_cpu), [](const std::atomic_uint64_t& encodedShiftPointer) { return TransformPointer::decodeFixed64(encodedShiftPointer.load(), Level); });
        for (size_t i = 0; i < grids.size(); ++i) {
            const auto cpuResult = parents_cpu[i];
            const auto gpuResult = parents[i];
            if (cpuResult != gpuResult)
                spdlog::info("i = {}", i);
            assert_always(gpuResult == cpuResult);
        }
    }
#endif // COMPARE_WITH_CPU

    // We now know for every grid whether there is (one) other grid from which it can be reconstructed.
    // When nodes can be reconstructed from each-other (the transformation is invertible) then two nodes
    // could point to each-other and create a circle. A similar problem is that we may create chains of
    // transformations: node A is reconstructed from node B, which is reconstructed from node C. We want
    // to limit ourselves to a single level of indirection in order to keep traversal of the DAG efficient.
    // The following code breaks these chains & cycles by indicating which nodes must be stored (isUnique),
    // and which ones are allowed to be reconstructed (!isUnique).
    spdlog::info("Process results");
    std::vector<bool> isUnique(items.size(), false);
    size_t numUnique = 0, numUniqueInvariant = 0, numDoubleIndirect = 0;
#if 1
    for (size_t itemIdx = 0; itemIdx < items.size(); ++itemIdx) {
        const auto parent = parents[itemIdx];
        if (parent == TransformPointer::sentinel()) {
            isUnique[itemIdx] = true;
        } else if (!isUnique[itemIdx]) {
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
    std::fill(std::begin(outPrevLevelMapping), std::end(outPrevLevelMapping), TransformPointer());
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
    return out;
}

template std::vector<EditSubGrid<void>> findDuplicatesAmongFlattenedItems_gpu<2, EditSubGrid<void>>(std::span<const EditSubGrid<void>> items, std::span<const LargeSubGrid<4>> grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<EditNode<TransformPointer>> findDuplicatesAmongFlattenedItems_gpu<3, EditNode<TransformPointer>>(std::span<const EditNode<TransformPointer>> items, std::span<const LargeSubGrid<8>> grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);
template std::vector<EditNode<TransformPointer>> findDuplicatesAmongFlattenedItems_gpu<4, EditNode<TransformPointer>>(std::span<const EditNode<TransformPointer>> items, std::span<const LargeSubGrid<16>> grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);

template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_gpu<2, uint32_t>(std::span<const uint32_t>, std::span<const LargeSubGrid<4>>, std::vector<TransformPointer>&, const TransformDAGConfig&);
template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_gpu<3, uint32_t>(std::span<const uint32_t>, std::span<const LargeSubGrid<8>>, std::vector<TransformPointer>&, const TransformDAGConfig&);
template std::vector<uint32_t> findDuplicatesAmongFlattenedItems_gpu<4, uint32_t>(std::span<const uint32_t>, std::span<const LargeSubGrid<16>>, std::vector<TransformPointer>&, const TransformDAGConfig&);

}