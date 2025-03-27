#pragma once
#include "large_sub_grid.h"
#include "voxcom/utility/error_handling.h"
#include "voxcom/utility/my_cuda.h"
#include "voxcom/voxel/structure.h"
#include <array>
#include <cstring>
#include <filesystem>
#include <glm/vec3.hpp>
#include <optional>
#include <span>
#include <vector>

#define COMPACT_TRANFORM_POINTER 1
#define MAX_TRANSLATION_LEVEL 4

namespace voxcom {

struct TransformPointer {
    static constexpr size_t tbits = 6; // Translation bits.

#if COMPACT_TRANFORM_POINTER
    // TODO(Mathijs): allow for left shift (negative shift amount).
    uint64_t ptr : 37 = 0;

    // Symmetry
    uint64_t sx : 1 = 0, sy : 1 = 0, sz : 1 = 0;

    // Axis-permutation
    uint64_t p0 : 2 = 0, p1 : 2 = 1, p2 : 2 = 2;

    // Translation
    uint64_t tx : tbits = 0, ty : tbits = 0, tz : tbits = 0;
#else
    uint64_t ptr;
    glm::bvec3 symmetry;
    glm::u8vec3 axisPermutation;
    glm::ivec3 translation;
#endif

    HOST_DEVICE static TransformPointer sentinel();
    constexpr auto operator<=>(const TransformPointer& rhs) const = default;
    HOST_DEVICE operator uint32_t() const;
    HOST_DEVICE TransformPointer& operator=(uint32_t p);

    HOST_DEVICE inline static TransformPointer create(uint64_t ptr, const glm::bvec3& symmetry = glm::bvec3(false), const glm::u8vec3& axisPermutation = glm::u8vec3(0, 1, 2), const glm::ivec3& translation = glm::ivec3(0))
    {
        TransformPointer out;
        out.ptr = ptr;
        out.setSymmetry(symmetry);
        out.setAxisPermutation(axisPermutation);
        out.setTranslation(translation);
        return out;
    }
    HOST_DEVICE inline void setSymmetry(const glm::bvec3& symmetry_)
    {
#if COMPACT_TRANFORM_POINTER
        sx = symmetry_.x;
        sy = symmetry_.y;
        sz = symmetry_.z;
#else
        this->symmetry = symmetry_;
#endif
    }
    HOST_DEVICE inline void setAxisPermutation(const glm::u8vec3& axisPermutation_)
    {
#if COMPACT_TRANFORM_POINTER
        p0 = axisPermutation_.x;
        p1 = axisPermutation_.y;
        p2 = axisPermutation_.z;
#else
        this->axisPermutation = axisPermutation_;
#endif
    }
    HOST_DEVICE inline void setTranslation(const glm::ivec3& translation_)
    {
#if COMPACT_TRANFORM_POINTER
        constexpr int translationShift = 1u << (tbits - 1);
        tx = (unsigned)(translation_.x + translationShift);
        ty = (unsigned)(translation_.y + translationShift);
        tz = (unsigned)(translation_.z + translationShift);
#else
        this->translation = glm::ivec3(translation_);
#endif
    }

    HOST_DEVICE inline glm::bvec3 getSymmetry() const
    {
#if COMPACT_TRANFORM_POINTER
        return glm::bvec3(sx, sy, sz);
#else
        return this->symmetry;
#endif
    }

    HOST_DEVICE inline glm::u8vec3 getAxisPermutation() const
    {
#if COMPACT_TRANFORM_POINTER
        return glm::u8vec3(p0, p1, p2);
#else
        return this->axisPermutation;
#endif
    }

    HOST_DEVICE inline glm::ivec3 getTranslation() const
    {
#if COMPACT_TRANFORM_POINTER
        constexpr auto translationShift = 1u << (tbits - 1);
        return glm::ivec3((int)tx - translationShift, (int)ty - translationShift, (int)tz - translationShift);
#else
        return this->translation;
#endif
    }

    HOST_DEVICE inline bool hasSymmetry() const
    {
        return glm::any(getSymmetry());
    }
    HOST_DEVICE inline bool hasAxisPermutation() const
    {
        return getAxisPermutation() != glm::u8vec3(0, 1, 2);
    }
    HOST_DEVICE inline bool hasTranslation() const
    {
        return glm::any(glm::notEqual(getTranslation(), glm::ivec3(0)));
    }
    HOST_DEVICE inline bool hasTransform() const
    {
        return hasTranslation() || hasSymmetry() || hasAxisPermutation();
    }

    static constexpr size_t NumUniqueAxisPermutations = 6;
    HOST_DEVICE static glm::u8vec3 decodeInverseAxisPermutation(uint32_t encodedAxisPermutation);
    HOST_DEVICE static uint32_t encodeAxisPermutation(const glm::u8vec3& axisPermutation);

    HOST_DEVICE static inline glm::u8vec3 decodeAxisPermutation(uint32_t encodedAxisPermutation)
    {
        const std::array axisPermutations {
            glm::u8vec3(0, 1, 2),
            glm::u8vec3(0, 2, 1),
            glm::u8vec3(1, 0, 2),
            glm::u8vec3(1, 2, 0),
            glm::u8vec3(2, 0, 1),
            glm::u8vec3(2, 1, 0)
        };
        return axisPermutations[encodedAxisPermutation];
    }

    HOST_DEVICE static TransformPointer decodeFixed64(uint64_t encodedPointer, uint32_t level);
    HOST_DEVICE uint64_t encodeFixed64(uint32_t level) const;
};

#if COMPACT_TRANFORM_POINTER
static_assert(sizeof(TransformPointer) == sizeof(uint64_t));
#endif

struct TransformDAGConfig {
    bool symmetry = true, axisPermutation = true, translation = true;
    uint32_t maxTranslationLevel = 4;
};
template <template <typename, typename> typename Structure>
EditStructure<void, TransformPointer> constructTransformDAG(const Structure<void, uint32_t>& inDag, const TransformDAGConfig& config);
StaticStructure<void, TransformPointer> constructStaticTransformDAG(StaticStructure<void, uint32_t>&& inDag, const TransformDAGConfig& config);

StaticStructure<void, TransformPointer> constructStaticTransformDAGHierarchical(const StaticStructure<void, uint32_t>& inDag, const TransformDAGConfig& config);
StaticStructure<void, TransformPointer> constructStaticTransformDAGHierarchical(const StaticStructure<void, uint32_t>& inDag, uint32_t midLevel, std::vector<uint32_t>& outMidLevelRemainingNodeStarts, const TransformDAGConfig& config);

template <template <typename, typename> typename Structure>
EditStructure<void, TransformPointer> constructTransformDAGHierarchical(const Structure<void, uint32_t>& inDag, const TransformDAGConfig& config);
template <template <typename, typename> typename Structure>
EditStructure<void, TransformPointer> constructTransformDAGHierarchical(const Structure<void, uint32_t>& inDag, uint32_t midLevel, std::vector<uint32_t>& outMidLevelInvMapping, const TransformDAGConfig& config);

EditStructure<void, TransformPointer> findAllHierarchicalTransforms(const EditStructure<void, uint32_t>& inDag);

// uint64_t bvec3ToU64(bool x, bool y, bool z);
// uint64_t bvec3ToU64(const glm::bvec3& v);
// glm::bvec3 u64ToBvec3(uint64_t v);

// uint32_t applySymmetry(uint32_t childIndex, const glm::bvec3& symmetry);
// glm::u8vec3 invertPermutation(const glm::u8vec3& permutation);
template <typename T>
HOST_DEVICE inline constexpr glm::vec<3, T> applyAxisPermutation(const glm::vec<3, T>& inVec, const glm::u8vec3& permutation)
{
    return glm::vec<3, T> {
        inVec[permutation[0]],
        inVec[permutation[1]],
        inVec[permutation[2]]
    };
}
template <glm::u8vec3 permutation, typename T>
HOST_DEVICE inline constexpr glm::vec<3, T> applyAxisPermutation(const glm::vec<3, T>& inVec)
{
    glm::vec<3, T> out;
    if constexpr (permutation[0] == 0)
        out.x = inVec.x;
    else if constexpr (permutation[0] == 1)
        out.x = inVec.y;
    else
        out.x = inVec.z;

    if constexpr (permutation[1] == 0)
        out.y = inVec.x;
    else if constexpr (permutation[1] == 1)
        out.y = inVec.y;
    else
        out.y = inVec.z;

    if constexpr (permutation[2] == 0)
        out.z = inVec.x;
    else if constexpr (permutation[2] == 1)
        out.z = inVec.y;
    else
        out.z = inVec.z;
    return out;
}
// uint32_t applyAxisPermutation(uint32_t childIndex, const glm::ivec3& permutation);
HOST_DEVICE inline uint64_t bvec3ToU64(bool x, bool y, bool z)
{
    uint64_t out = 0;
    out |= (uint64_t)x << 0;
    out |= (uint64_t)y << 1;
    out |= (uint64_t)z << 2;
    return out;
}

HOST_DEVICE inline uint64_t bvec3ToU64(const glm::bvec3& v)
{
    uint64_t out = 0;
    out |= (uint64_t)v.x << 0;
    out |= (uint64_t)v.y << 1;
    out |= (uint64_t)v.z << 2;
    return out;
}

HOST_DEVICE inline glm::bvec3 u64ToBvec3(uint64_t v)
{
    return glm::bvec3(v & 0b001, v & 0b010, v & 0b100);
}

inline uint32_t applySymmetry(uint32_t childIndex, const glm::bvec3& symmetry)
{
    const auto childPos = morton_decode32<3>(childIndex);
    const auto symmetryChildPos = glm::mix(childPos, glm::uvec3(1) - childPos, symmetry);
    return voxcom::morton_encode32(symmetryChildPos);
}

HOST_DEVICE inline glm::u8vec3 invertPermutation(const glm::u8vec3& permutation)
{
    glm::u8vec3 out;
    out[permutation[0]] = 0;
    out[permutation[1]] = 1;
    out[permutation[2]] = 2;
    return out;
}

HOST_DEVICE inline uint32_t applyAxisPermutation(uint32_t childIndex, const glm::ivec3& permutation)
{
    // const glm::uvec3 inChildPos = morton_decode32<3>(childIndex);
    // const glm::uvec3 outChildPos {
    //     inChildPos[permutation[0]],
    //     inChildPos[permutation[1]],
    //     inChildPos[permutation[2]]
    // };
    // return morton_encode32(outChildPos);
    uint32_t out = 0;
    if (childIndex & (1u << permutation[0]))
        out |= 0b001; // x
    if (childIndex & (1u << permutation[1]))
        out |= 0b010; // y
    if (childIndex & (1u << permutation[2]))
        out |= 0b100; // z
    return out;
}

template <uint32_t Level, typename Structure>
class FlatGridGenerator {
public:
    FlatGridGenerator(const Structure* pStructure, std::span<const uint32_t> indices);

    // size_t size() const;
    LargeSubGrid<(1u << Level)> operator[](uint32_t nodeIdx) const;
    void fillHashGrid(uint32_t nodeIdx, TypedLargeSubGrid<uint32_t, (1u << Level)>& out) const;
    void fillHashGrid(uint32_t nodeIdx, TypedLargeSubGrid<uint64_t, (1u << Level)>& out) const;

private:
    const Structure* m_pStructure;
    std::span<const uint32_t> m_indices;
};

template <uint32_t Level>
class StaticFlatGridGenerator {
public:
    StaticFlatGridGenerator(const StaticStructure<void, TransformPointer>* pOctree, std::span<const uint32_t> itemStarts);

    // size_t size() const;
    LargeSubGrid<(1u << Level)> operator[](uint32_t nodeIdx) const;
    void fillHashGrid(uint32_t nodeIdx, TypedLargeSubGrid<uint32_t, (1u << Level)>& out) const;
    void fillHashGrid(uint32_t nodeIdx, TypedLargeSubGrid<uint64_t, (1u << Level)>& out) const;

    size_t getSize() const { return m_itemStarts.size(); }

private:
    const StaticStructure<void, TransformPointer>* m_pOctree;
    std::span<const uint32_t> m_itemStarts;
};

template <size_t Level, typename Item, typename GridGenerator>
std::vector<Item> findDuplicatesAmongFlattenedItems_cpp2(std::span<const Item> items, const GridGenerator& gridGenerator, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);

template <size_t Level, typename Item>
std::vector<Item> findDuplicatesAmongFlattenedItems_gpu(std::span<const Item> items, std::span<const LargeSubGrid<(1u << Level)>> grids, std::vector<TransformPointer>& outPrevLevelMapping, const TransformDAGConfig& config);

HOST_DEVICE inline glm::u8vec3 TransformPointer::decodeInverseAxisPermutation(uint32_t encodedAxisPermutation)
{
    const std::array inverseAxisPermutations {
        glm::u8vec3(0, 1, 2),
        glm::u8vec3(0, 2, 1),
        glm::u8vec3(1, 0, 2),
        glm::u8vec3(2, 0, 1),
        glm::u8vec3(1, 2, 0),
        glm::u8vec3(2, 1, 0)
    };
    return inverseAxisPermutations[encodedAxisPermutation];
}

HOST_DEVICE inline uint32_t TransformPointer::encodeAxisPermutation(const glm::u8vec3& axisPermutation)
{
    const auto out = 2 * axisPermutation[0] + (axisPermutation[2] < axisPermutation[1]);
    assert(decodeAxisPermutation(out) == axisPermutation);
    return out;
}

HOST_DEVICE inline TransformPointer& TransformPointer::operator=(uint32_t p)
{
    this->ptr = p;
    return *this;
}

HOST_DEVICE inline TransformPointer::operator uint32_t() const
{
    return (uint32_t)ptr;
}

HOST_DEVICE inline TransformPointer TransformPointer::sentinel()
{
    return TransformPointer::create(0xFFFF'FFFF);
}

HOST_DEVICE inline TransformPointer TransformPointer::decodeFixed64(uint64_t encodedPointer, uint32_t level)
{
    glm::ivec3 translation { 0 };
    if (level != 0) {
        const uint32_t translationBits = level + 1;
        const uint64_t translationMask = (1llu << translationBits) - 1llu;

        glm::uvec3 unsignedTranslation;
        unsignedTranslation.x = unsigned(encodedPointer & translationMask);
        encodedPointer >>= translationBits;
        unsignedTranslation.y = unsigned(encodedPointer & translationMask);
        encodedPointer >>= translationBits;
        unsignedTranslation.z = unsigned(encodedPointer & translationMask);
        encodedPointer >>= translationBits;
        translation = glm::ivec3(unsignedTranslation) - glm::ivec3(1 << level);
    }

    const glm::bvec3 symmetry = u64ToBvec3(encodedPointer & 0b111);
    encodedPointer >>= 3;

    const glm::u8vec3 axisPermutation = decodeAxisPermutation(encodedPointer & 0b111);
    encodedPointer >>= 3;

    return TransformPointer::create(encodedPointer, symmetry, axisPermutation, translation);
}

HOST_DEVICE inline uint64_t TransformPointer::encodeFixed64(uint32_t level) const
{
    // Pointer and symmetry.
    uint64_t out = ptr;
    out <<= 3;
    out |= encodeAxisPermutation(getAxisPermutation());
    out <<= 3;
    out |= bvec3ToU64(getSymmetry());

    // Shift.
    if (level != 0) {
        const uint32_t shiftBits = level + 1;
        const glm::uvec3 offsetShift = glm::uvec3(getTranslation() + glm::ivec3(1 << level));
        out <<= shiftBits;
        out |= offsetShift.z;
        out <<= shiftBits;
        out |= offsetShift.y;
        out <<= shiftBits;
        out |= offsetShift.x;
    }

    return out;
}

}

template <>
class fmt::formatter<voxcom::TransformPointer> {
public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr inline auto format(const voxcom::TransformPointer& pointer, Context& ctx) const
    {
        return fmt::format_to(ctx.out(), "(ptr = {}, shift = {}, symmetry = {}, axisPermutation = {})", pointer.ptr, pointer.getTranslation(), pointer.getSymmetry(), pointer.getAxisPermutation());
    }
};

namespace std {

template <>
struct hash<voxcom::TransformPointer> {
    inline size_t operator()(const voxcom::TransformPointer& ptr) const
    {
        size_t seed = 0;
        voxcom::hash_combine(seed, ptr.ptr);
        const auto symmetry = ptr.getSymmetry();
        voxcom::hash_combine(seed, symmetry.x);
        voxcom::hash_combine(seed, symmetry.y);
        voxcom::hash_combine(seed, symmetry.z);
        const auto axisPermutation = ptr.getAxisPermutation();
        voxcom::hash_combine(seed, axisPermutation.x);
        voxcom::hash_combine(seed, axisPermutation.y);
        voxcom::hash_combine(seed, axisPermutation.z);
        const auto translation = ptr.getTranslation();
        voxcom::hash_combine(seed, translation.x);
        voxcom::hash_combine(seed, translation.y);
        voxcom::hash_combine(seed, translation.z);
        return seed;
    }
};

}
