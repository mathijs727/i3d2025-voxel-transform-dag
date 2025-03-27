#pragma once
#include <array>
#include <bit>
#include <bitset>
#include <voxcom/utility/error_handling.h>
#include <voxcom/utility/hash.h>
#include <voxcom/utility/my_cuda.h>
#include <voxcom/voxel/structure.h>

template <uint32_t Size>
class CUDABitSet {
public:
    HOST_DEVICE CUDABitSet()
    {
        for (uint32_t i = 0; i < NumU64; ++i)
            storage[i] = 0;
    }
    HOST_DEVICE void set(uint32_t i)
    {
        const auto u64 = i >> 6;
        const auto bit = i & 63;
        storage[u64] |= 1llu << bit;
    }
    HOST_DEVICE void set(uint32_t i, bool value)
    {
        const auto u64 = i >> 6;
        const auto bit = i & 63;
        if (value)
            storage[u64] |= 1llu << bit;
        else
            storage[u64] &= (~(1llu << bit));
    }
    HOST_DEVICE bool test(uint32_t i) const
    {
        const auto u64 = i >> 6;
        const auto bit = i & 63;
        return (storage[u64] >> bit) & 0b1;
    }
    HOST_DEVICE void flip(uint32_t i)
    {
        const auto u64 = i >> 6;
        const auto bit = i & 63;
        storage[u64] ^= 1llu << bit;
    }
    HOST_DEVICE bool operator==(const CUDABitSet<Size>& other) const
    {
        for (uint32_t i = 0; i < NumU64; ++i) {
            if (storage[i] != other.storage[i])
                return false;
        }
        return true;
    }
    HOST_DEVICE bool operator<(const CUDABitSet<Size>& other) const
    {
        for (uint32_t i = 0; i < NumU64; ++i) {
            if (storage[i] < other.storage[i])
                return true;
            else if (storage[i] > other.storage[i])
                return false;
            // if same then continue to next U64
        }
        return false;
    }
    HOST_DEVICE uint32_t popcount() const
    {
        uint32_t out = 0;
        for (uint64_t v : storage)
            out += std::popcount(v);
        return out;
    }

public:
    static_assert(Size % 64 == 0);
    static constexpr uint32_t NumU64 = Size / 64;

public:
    uint64_t storage[NumU64];
};

template <uint32_t Resolution>
struct LargeSubGrid {
    static constexpr uint32_t strideX = 1;
    static constexpr uint32_t strideY = Resolution * strideX;
    static constexpr uint32_t strideZ = Resolution * strideY;
    // std::bitset<Resolution * Resolution * Resolution> voxels;
    CUDABitSet<Resolution * Resolution * Resolution> voxels;

    HOST_DEVICE bool operator<(const LargeSubGrid<Resolution>& other) const { return voxels < other.voxels; };
    HOST_DEVICE bool operator==(const LargeSubGrid<Resolution>& other) const { return voxels == other.voxels; };
    HOST_DEVICE void set(const glm::uvec3& v) { voxels.set(v.x * strideX + v.y * strideY + v.z * strideZ); }
    HOST_DEVICE void set(const glm::uvec3& v, bool value) { voxels.set(v.x * strideX + v.y * strideY + v.z * strideZ, value); }
    HOST_DEVICE void set(uint32_t i) { voxels.set(i); }
    HOST_DEVICE bool get(const glm::uvec3& v) const { return voxels.test(v.x * strideX + v.y * strideY + v.z * strideZ); }
    HOST_DEVICE bool get(uint32_t i) const { return voxels.test(i); }
    HOST_DEVICE void flip(uint32_t i) { voxels.flip(i); }
    HOST_DEVICE uint32_t popcount() const { return voxels.popcount(); }
};

template <typename T, uint32_t Resolution>
struct TypedLargeSubGrid {
    static constexpr uint32_t strideX = 1;
    static constexpr uint32_t strideY = Resolution * strideX;
    static constexpr uint32_t strideZ = Resolution * strideY;
    alignas(64) std::array<T, Resolution * Resolution * Resolution> voxels;

    HOST_DEVICE bool operator<(const TypedLargeSubGrid<T, Resolution>& other) const { return voxels < other.voxels; };
    HOST_DEVICE bool operator==(const TypedLargeSubGrid<T, Resolution>& other) const { return voxels == other.voxels; };
    HOST_DEVICE void set(const glm::uvec3& v, const T& value) { voxels.at(v.x * strideX + v.y * strideY + v.z * strideZ) = value; }
    HOST_DEVICE void set(uint32_t i, const T& value) { voxels.at(i) = value; }
    HOST_DEVICE T get(const glm::uvec3& v) const { return voxels.at(v.x * strideX + v.y * strideY + v.z * strideZ); }
    HOST_DEVICE T get(uint32_t i) const { return voxels.at(i); }
};
template <typename T, uint32_t Resolution>
struct TypedLargeSubGrid2D {
    static constexpr uint32_t strideX = 1;
    static constexpr uint32_t strideY = Resolution * strideX;
    alignas(64) std::array<T, Resolution * Resolution> voxels;

    HOST_DEVICE bool operator<(const TypedLargeSubGrid2D<T, Resolution>& other) const { return voxels < other.voxels; };
    HOST_DEVICE bool operator==(const TypedLargeSubGrid2D<T, Resolution>& other) const { return voxels == other.voxels; };
    HOST_DEVICE void set(const glm::uvec2& v, const T& value) { voxels.at(v.x * strideX + v.y * strideY) = value; }
    HOST_DEVICE void set(uint32_t i, const T& value) { voxels.at(i) = value; }
    HOST_DEVICE T get(const glm::uvec2& v) const { return voxels.at(v.x * strideX + v.y * strideY); }
    HOST_DEVICE T get(uint32_t i) const { return voxels.at(i); }
};

template <size_t Level, typename Structure>
HOST_DEVICE LargeSubGrid<(1u << Level)> createLargeSubGrid(const Structure& structure, uint32_t nodeIdx)
{
    constexpr uint32_t resolution = (1u << Level);
    LargeSubGrid<resolution> out;
    for (uint32_t z = 0; z < resolution; ++z) {
        for (uint32_t y = 0; y < resolution; ++y) {
            for (uint32_t x = 0; x < resolution; ++x) {
                if (structure.get(glm::ivec3(x, y, z), Level, nodeIdx))
                    out.set(glm::uvec3(x, y, z));
            }
        }
    }
    return out;
}

template <size_t RootLevel>
HOST_DEVICE uint32_t addLargeSubGridToStructure(const LargeSubGrid<(1u << RootLevel)>& subGrid, voxcom::EditStructure<void, uint32_t>& outStructure)
{
    constexpr uint32_t Resolution = (1u << RootLevel);

    if constexpr (RootLevel == outStructure.subGridLevel) {
        voxcom::EditSubGrid<void> outSubGrid {};
        for (uint32_t z = 0; z < Resolution; ++z) {
            for (uint32_t y = 0; y < Resolution; ++y) {
                for (uint32_t x = 0; x < Resolution; ++x) {
                    const glm::uvec3 voxel { x, y, z };
                    if (subGrid.get(voxel))
                        outSubGrid.set(voxel);
                }
            }
        }
        const uint32_t out = (uint32_t)outStructure.subGrids.size();
        outStructure.subGrids.push_back(outSubGrid);
        return out;
    } else { // RootLevel != outStructure.subGridLevel
        // Create local octree using existing code.
        voxcom::EditStructure<void, uint32_t> intermediate(Resolution);
        for (uint32_t z = 0; z < Resolution; ++z) {
            for (uint32_t y = 0; y < Resolution; ++y) {
                for (uint32_t x = 0; x < Resolution; ++x) {
                    const glm::uvec3 voxel { x, y, z };
                    if (subGrid.get(voxel))
                        intermediate.set(voxel);
                }
            }
        }

        // Copy subgrids.
        uint32_t prevLevelOffset = (uint32_t)outStructure.subGrids.size();
        outStructure.subGrids.resize(outStructure.subGrids.size() + intermediate.subGrids.size());
        std::copy(std::begin(intermediate.subGrids), std::end(intermediate.subGrids), std::begin(outStructure.subGrids) + prevLevelOffset);

        // Copy nodes level-by-level while updating their pointers.
        voxcom::assert_always(outStructure.subGridLevel == intermediate.subGridLevel);
        voxcom::assert_always(intermediate.nodesPerLevel[intermediate.rootLevel].size() == 1);
        for (uint32_t level = intermediate.subGridLevel + 1; level <= intermediate.rootLevel; ++level) {
            const auto& inNodes = intermediate.nodesPerLevel[level];
            auto& outNodes = outStructure.nodesPerLevel[level];
            const auto currentLevelOffset = (uint32_t)outNodes.size();
            for (auto node : inNodes) {
                for (uint32_t& child : node.children) {
                    if (child != node.EmptyChild)
                        child += prevLevelOffset;
                }
                outNodes.push_back(node);
            }
            prevLevelOffset = currentLevelOffset;
        }
        return prevLevelOffset;
    } // RootLevel != outStructure.subGridLevel
}

template <uint32_t Resolution>
class fmt::formatter<LargeSubGrid<Resolution>> {
public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr inline auto format(const LargeSubGrid<Resolution>& subGrid, Context& ctx) const
    {
        for (uint32_t z = 0; z < Resolution; ++z) {
            for (uint32_t y = 0; y < Resolution; ++y) {
                fmt::format_to(ctx.out(), "[");
                for (uint32_t x = 0; x < Resolution; ++x) {
                    const glm::uvec3 voxel { x, y, z };
                    fmt::format_to(ctx.out(), "{} ", subGrid.get(voxel) ? "x" : " ");
                }
                fmt::format_to(ctx.out(), "]\n");
            }
            fmt::format_to(ctx.out(), "\n");
        }
        return ctx.out();
    }
};
template <typename T, uint32_t Resolution>
class fmt::formatter<TypedLargeSubGrid<T, Resolution>> {
public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr inline auto format(const TypedLargeSubGrid<T, Resolution>& subGrid, Context& ctx) const
    {
        for (uint32_t z = 0; z < Resolution; ++z) {
            for (uint32_t y = 0; y < Resolution; ++y) {
                fmt::format_to(ctx.out(), "[");
                for (uint32_t x = 0; x < Resolution; ++x) {
                    const glm::uvec3 voxel { x, y, z };
                    fmt::format_to(ctx.out(), "{} ", subGrid.get(voxel));
                }
                fmt::format_to(ctx.out(), "]\n");
            }
            fmt::format_to(ctx.out(), "\n");
        }
        return ctx.out();
    }
};
template <typename T, uint32_t Resolution>
class fmt::formatter<TypedLargeSubGrid2D<T, Resolution>> {
public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr inline auto format(const TypedLargeSubGrid2D<T, Resolution>& subGrid, Context& ctx) const
    {
        for (uint32_t y = 0; y < Resolution; ++y) {
            fmt::format_to(ctx.out(), "[");
            for (uint32_t x = 0; x < Resolution; ++x) {
                const glm::uvec2 voxel { x, y };
                fmt::format_to(ctx.out(), "{} ", subGrid.get(voxel));
            }
            fmt::format_to(ctx.out(), "]\n");
        }
        return ctx.out();
    }
};

// https://stackoverflow.com/questions/5085915/what-is-the-best-hash-function-for-uint64-t-keys-ranging-from-0-to-its-max-value
inline uint64_t murmurmHash3(uint64_t key)
{
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccd;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53;
    key ^= key >> 33;
    return key;
}

namespace std {
template <uint32_t Size>
struct hash<CUDABitSet<Size>> {
    inline size_t operator()(const CUDABitSet<Size>& bitset) const noexcept
    {
        size_t seed = 0;
        for (uint64_t v : bitset.storage)
            voxcom::hash_combine(seed, murmurmHash3(v));
        return seed;
    }
};
template <uint32_t Resolution>
struct hash<LargeSubGrid<Resolution>> {
    inline size_t operator()(const LargeSubGrid<Resolution>& subGrid) const noexcept
    {
        return std::hash<CUDABitSet<Resolution * Resolution * Resolution>>()(subGrid.voxels);
    }
};
template <typename T, uint32_t Resolution>
struct hash<TypedLargeSubGrid<T, Resolution>> {
    size_t operator()(const TypedLargeSubGrid<T, Resolution>& grid) const
    {
        size_t seed = 0;
        for (const T& v : grid.voxels)
            voxcom::hash_combine(seed, v);
        return seed;
    }
};
template <typename T, uint32_t Resolution>
struct hash<TypedLargeSubGrid2D<T, Resolution>> {
    size_t operator()(const TypedLargeSubGrid2D<T, Resolution>& grid) const
    {
        size_t seed = 0;
        for (const T& v : grid.voxels)
            voxcom::hash_combine(seed, v);
        return seed;
    }
};
}