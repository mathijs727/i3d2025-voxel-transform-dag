#include "voxcom/utility/disable_all_warnings.h"
#include "voxcom/voxel/transform_dag.h"
#include <algorithm>
#include <array>
#include <bitset>
#include <execution>
#include <glm/vec3.hpp>
#include <limits>
#include <spdlog/spdlog.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
DISABLE_WARNINGS_PUSH()
#include <robin_hood.h>
DISABLE_WARNINGS_POP()
#include "voxcom/voxel/large_sub_grid.h"

namespace voxcom {

// Find n-th set bit in mask (starting from least-significant bit).
static int fns(uint32_t mask, int offset)
{
    for (int i = 0; i < 32; ++i) {
        if ((mask >> i) & 0b1) {
            if ((offset--) == 0) {
                return i;
            }
        }
    }
    return -1;
}

struct GenericTransformID {
    std::array<uint8_t, 8> childPermutations { 0, 1, 2, 3, 4, 5, 6, 7 };

    constexpr auto operator<=>(const GenericTransformID&) const noexcept = default;
};
struct GenericTransformPointer {
    uint32_t nodeIdx;
    GenericTransformID transform;

    static GenericTransformPointer sentinel() { return { .nodeIdx = 0xFFFF'FFFF, .transform = {} }; }
    constexpr auto operator<=>(const GenericTransformPointer&) const noexcept = default;
};

#if 0 // Limit to symmetry + axis
static constexpr std::array transforms {
    GenericTransformID { .childPermutations = { 0, 1, 2, 3, 4, 5, 6, 7 } },
    GenericTransformID { .childPermutations = { 1, 0, 3, 2, 5, 4, 7, 6 } },
    GenericTransformID { .childPermutations = { 2, 3, 0, 1, 6, 7, 4, 5 } },
    GenericTransformID { .childPermutations = { 3, 2, 1, 0, 7, 6, 5, 4 } },
    GenericTransformID { .childPermutations = { 4, 5, 6, 7, 0, 1, 2, 3 } },
    GenericTransformID { .childPermutations = { 5, 4, 7, 6, 1, 0, 3, 2 } },
    GenericTransformID { .childPermutations = { 6, 7, 4, 5, 2, 3, 0, 1 } },
    GenericTransformID { .childPermutations = { 7, 6, 5, 4, 3, 2, 1, 0 } },
    GenericTransformID { .childPermutations = { 0, 1, 4, 5, 2, 3, 6, 7 } },
    GenericTransformID { .childPermutations = { 1, 0, 5, 4, 3, 2, 7, 6 } },
    GenericTransformID { .childPermutations = { 4, 5, 0, 1, 6, 7, 2, 3 } },
    GenericTransformID { .childPermutations = { 5, 4, 1, 0, 7, 6, 3, 2 } },
    GenericTransformID { .childPermutations = { 2, 3, 6, 7, 0, 1, 4, 5 } },
    GenericTransformID { .childPermutations = { 3, 2, 7, 6, 1, 0, 5, 4 } },
    GenericTransformID { .childPermutations = { 6, 7, 2, 3, 4, 5, 0, 1 } },
    GenericTransformID { .childPermutations = { 7, 6, 3, 2, 5, 4, 1, 0 } },
    GenericTransformID { .childPermutations = { 0, 2, 1, 3, 4, 6, 5, 7 } },
    GenericTransformID { .childPermutations = { 2, 0, 3, 1, 6, 4, 7, 5 } },
    GenericTransformID { .childPermutations = { 1, 3, 0, 2, 5, 7, 4, 6 } },
    GenericTransformID { .childPermutations = { 3, 1, 2, 0, 7, 5, 6, 4 } },
    GenericTransformID { .childPermutations = { 4, 6, 5, 7, 0, 2, 1, 3 } },
    GenericTransformID { .childPermutations = { 6, 4, 7, 5, 2, 0, 3, 1 } },
    GenericTransformID { .childPermutations = { 5, 7, 4, 6, 1, 3, 0, 2 } },
    GenericTransformID { .childPermutations = { 7, 5, 6, 4, 3, 1, 2, 0 } },
    GenericTransformID { .childPermutations = { 0, 2, 4, 6, 1, 3, 5, 7 } },
    GenericTransformID { .childPermutations = { 2, 0, 6, 4, 3, 1, 7, 5 } },
    GenericTransformID { .childPermutations = { 4, 6, 0, 2, 5, 7, 1, 3 } },
    GenericTransformID { .childPermutations = { 6, 4, 2, 0, 7, 5, 3, 1 } },
    GenericTransformID { .childPermutations = { 1, 3, 5, 7, 0, 2, 4, 6 } },
    GenericTransformID { .childPermutations = { 3, 1, 7, 5, 2, 0, 6, 4 } },
    GenericTransformID { .childPermutations = { 5, 7, 1, 3, 4, 6, 0, 2 } },
    GenericTransformID { .childPermutations = { 7, 5, 3, 1, 6, 4, 2, 0 } },
    GenericTransformID { .childPermutations = { 0, 4, 1, 5, 2, 6, 3, 7 } },
    GenericTransformID { .childPermutations = { 4, 0, 5, 1, 6, 2, 7, 3 } },
    GenericTransformID { .childPermutations = { 1, 5, 0, 4, 3, 7, 2, 6 } },
    GenericTransformID { .childPermutations = { 5, 1, 4, 0, 7, 3, 6, 2 } },
    GenericTransformID { .childPermutations = { 2, 6, 3, 7, 0, 4, 1, 5 } },
    GenericTransformID { .childPermutations = { 6, 2, 7, 3, 4, 0, 5, 1 } },
    GenericTransformID { .childPermutations = { 3, 7, 2, 6, 1, 5, 0, 4 } },
    GenericTransformID { .childPermutations = { 7, 3, 6, 2, 5, 1, 4, 0 } },
    GenericTransformID { .childPermutations = { 0, 4, 2, 6, 1, 5, 3, 7 } },
    GenericTransformID { .childPermutations = { 4, 0, 6, 2, 5, 1, 7, 3 } },
    GenericTransformID { .childPermutations = { 2, 6, 0, 4, 3, 7, 1, 5 } },
    GenericTransformID { .childPermutations = { 6, 2, 4, 0, 7, 3, 5, 1 } },
    GenericTransformID { .childPermutations = { 1, 5, 3, 7, 0, 4, 2, 6 } },
    GenericTransformID { .childPermutations = { 5, 1, 7, 3, 4, 0, 6, 2 } },
    GenericTransformID { .childPermutations = { 3, 7, 1, 5, 2, 6, 0, 4 } },
    GenericTransformID { .childPermutations = { 7, 3, 5, 1, 6, 2, 4, 0 } }
};
static constexpr size_t NumTransformIDs = transforms.size();
static GenericTransformID decodeTransformID(uint32_t encodedTransformID);
static uint32_t encodeTransformID(const GenericTransformID& transformID)
{
    const uint32_t out = (uint32_t)std::distance(std::begin(transforms), std::find(std::begin(transforms), std::end(transforms), transformID));
    assert(decodeTransformID(out) == transformID);
    return out;
}
static GenericTransformID decodeTransformID(uint32_t encodedTransformID)
{
    return transforms[encodedTransformID];
}

static TransformPointer convertPointer(const GenericTransformPointer& ptr)
{
    static constexpr std::array lut {
        std::pair { glm::bvec3(false, false, false), glm::u8vec3(0, 1, 2) },
        std::pair { glm::bvec3(true, false, false), glm::u8vec3(0, 1, 2) },
        std::pair { glm::bvec3(false, true, false), glm::u8vec3(0, 1, 2) },
        std::pair { glm::bvec3(true, true, false), glm::u8vec3(0, 1, 2) },
        std::pair { glm::bvec3(false, false, true), glm::u8vec3(0, 1, 2) },
        std::pair { glm::bvec3(true, false, true), glm::u8vec3(0, 1, 2) },
        std::pair { glm::bvec3(false, true, true), glm::u8vec3(0, 1, 2) },
        std::pair { glm::bvec3(true, true, true), glm::u8vec3(0, 1, 2) },
        std::pair { glm::bvec3(false, false, false), glm::u8vec3(0, 2, 1) },
        std::pair { glm::bvec3(true, false, false), glm::u8vec3(0, 2, 1) },
        std::pair { glm::bvec3(false, true, false), glm::u8vec3(0, 2, 1) },
        std::pair { glm::bvec3(true, true, false), glm::u8vec3(0, 2, 1) },
        std::pair { glm::bvec3(false, false, true), glm::u8vec3(0, 2, 1) },
        std::pair { glm::bvec3(true, false, true), glm::u8vec3(0, 2, 1) },
        std::pair { glm::bvec3(false, true, true), glm::u8vec3(0, 2, 1) },
        std::pair { glm::bvec3(true, true, true), glm::u8vec3(0, 2, 1) },
        std::pair { glm::bvec3(false, false, false), glm::u8vec3(1, 0, 2) },
        std::pair { glm::bvec3(true, false, false), glm::u8vec3(1, 0, 2) },
        std::pair { glm::bvec3(false, true, false), glm::u8vec3(1, 0, 2) },
        std::pair { glm::bvec3(true, true, false), glm::u8vec3(1, 0, 2) },
        std::pair { glm::bvec3(false, false, true), glm::u8vec3(1, 0, 2) },
        std::pair { glm::bvec3(true, false, true), glm::u8vec3(1, 0, 2) },
        std::pair { glm::bvec3(false, true, true), glm::u8vec3(1, 0, 2) },
        std::pair { glm::bvec3(true, true, true), glm::u8vec3(1, 0, 2) },
        std::pair { glm::bvec3(false, false, false), glm::u8vec3(1, 2, 0) },
        std::pair { glm::bvec3(true, false, false), glm::u8vec3(1, 2, 0) },
        std::pair { glm::bvec3(false, true, false), glm::u8vec3(1, 2, 0) },
        std::pair { glm::bvec3(true, true, false), glm::u8vec3(1, 2, 0) },
        std::pair { glm::bvec3(false, false, true), glm::u8vec3(1, 2, 0) },
        std::pair { glm::bvec3(true, false, true), glm::u8vec3(1, 2, 0) },
        std::pair { glm::bvec3(false, true, true), glm::u8vec3(1, 2, 0) },
        std::pair { glm::bvec3(true, true, true), glm::u8vec3(1, 2, 0) },
        std::pair { glm::bvec3(false, false, false), glm::u8vec3(2, 0, 1) },
        std::pair { glm::bvec3(true, false, false), glm::u8vec3(2, 0, 1) },
        std::pair { glm::bvec3(false, true, false), glm::u8vec3(2, 0, 1) },
        std::pair { glm::bvec3(true, true, false), glm::u8vec3(2, 0, 1) },
        std::pair { glm::bvec3(false, false, true), glm::u8vec3(2, 0, 1) },
        std::pair { glm::bvec3(true, false, true), glm::u8vec3(2, 0, 1) },
        std::pair { glm::bvec3(false, true, true), glm::u8vec3(2, 0, 1) },
        std::pair { glm::bvec3(true, true, true), glm::u8vec3(2, 0, 1) },
        std::pair { glm::bvec3(false, false, false), glm::u8vec3(2, 1, 0) },
        std::pair { glm::bvec3(true, false, false), glm::u8vec3(2, 1, 0) },
        std::pair { glm::bvec3(false, true, false), glm::u8vec3(2, 1, 0) },
        std::pair { glm::bvec3(true, true, false), glm::u8vec3(2, 1, 0) },
        std::pair { glm::bvec3(false, false, true), glm::u8vec3(2, 1, 0) },
        std::pair { glm::bvec3(true, false, true), glm::u8vec3(2, 1, 0) },
        std::pair { glm::bvec3(false, true, true), glm::u8vec3(2, 1, 0) },
        std::pair { glm::bvec3(true, true, true), glm::u8vec3(2, 1, 0) }
    };
    static_assert(transforms.size() == lut.size());
    const auto [symmetry, axisPermutation] = lut[encodeTransformID(ptr.transform)];
    return TransformPointer::create(ptr.nodeIdx, symmetry, axisPermutation);
}
#else
static constexpr size_t NumTransformIDs = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1;
static GenericTransformID decodeTransformID(uint32_t encodedTransformID);
static uint32_t encodeTransformID(const GenericTransformID& transformID)
{
    // Permutation only allows shuffling of children; we cannot have the same child twice.
    // For example: [0, 1, 2, 3, 4, 4, 5, 6] is not allowed because child 4 is referenced twice.
    // Hence, the number of potential options goes down as we process the children.
    // The first child has 8 options, the second child has 7 options (cannot pick the same as the first child).
    //
    // This conceptually gives us a compacted permutation array with values ranges:
    // [0-8, 0-7, 0-6, 0-5, 0-4, 0-3, 0-2, 0-1, 0]
    //
    // We encode this compacted array as follows:
    // [c0, c1, c2, c3, c4, c5, c6, c7]
    // c7*0! + c6*1! + c5*2! + c4*3! + c3*4! + c2*5! + c1*6! + c0*7!
    constexpr std::array<uint32_t, 8> factorialLUT { 0, 1, 2, 6, 24, 120, 720, 5040 };

    uint32_t out = 0;
    uint32_t availableChildren = 0xFF;
    for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
        const uint8_t selectedChild = transformID.childPermutations[childIdx];
        const int compactedSelectedChild = std::popcount(availableChildren & (0xFF >> (8 - selectedChild)));
        assert((availableChildren >> selectedChild) & 0b1);
        availableChildren ^= 1u << selectedChild;
        out += compactedSelectedChild * factorialLUT[7 - childIdx];
    }
    assert(std::popcount(availableChildren) == 0);
    assert(decodeTransformID(out) == transformID);
    return out;
}
static GenericTransformID decodeTransformID(uint32_t encodedTransformID)
{
    constexpr std::array<uint32_t, 8> factorialLUT { 0, 1, 2, 6, 24, 120, 720, 5040 };

    GenericTransformID out {};
    uint32_t availableChildren = 0xFF;
    for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
        int compactedSelectedChild = 0;
        if (childIdx != 7) { // Last child does not have any choice.
            compactedSelectedChild = encodedTransformID / factorialLUT[7 - childIdx];
            encodedTransformID = encodedTransformID % factorialLUT[7 - childIdx];
        }

        const int selectedChild = fns(availableChildren, compactedSelectedChild);
        assert((availableChildren >> selectedChild) & 0b1);
        availableChildren ^= 1u << selectedChild;
        out.childPermutations[childIdx] = (uint8_t)selectedChild;
    }
    return out;
}

static TransformPointer convertPointer(const GenericTransformPointer& ptr)
{
    return TransformPointer::create(ptr.nodeIdx);
}
#endif
using InvarianceBitMask = std::bitset<NumTransformIDs>;

static GenericTransformID inverse(const GenericTransformID& transformID)
{
    GenericTransformID out {};
    for (uint8_t inChildIdx = 0; inChildIdx < 8; ++inChildIdx) {
        out.childPermutations[transformID.childPermutations[inChildIdx]] = inChildIdx;
    }
    return out;
}
static GenericTransformID applyTransform(const GenericTransformID& lhs, const GenericTransformID& rhs)
{
    GenericTransformID out {};
    for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
        out.childPermutations[childIdx] = rhs.childPermutations[lhs.childPermutations[childIdx]];
    }
    return out;
}

static uint32_t applyTransformToBitMask2x2x2(uint32_t bitmask2x2x2, const GenericTransformID& transform)
{
    uint32_t outBitMask2x2x2 = 0;
    for (uint32_t outVoxelIdx = 0; outVoxelIdx < 8; ++outVoxelIdx) {
        const auto inVoxelIdx = transform.childPermutations[outVoxelIdx];
        if ((bitmask2x2x2 >> inVoxelIdx) & 0b1)
            outBitMask2x2x2 |= 1u << outVoxelIdx;
    }
    return outBitMask2x2x2;
}
static uint64_t applyTransformToBitMask4x4x4(uint64_t bitmask4x4x4, const GenericTransformID& transform)
{
    uint64_t out = 0;
    for (uint32_t outChildIdx = 0; outChildIdx < 8; ++outChildIdx) {
        const uint32_t inChildIdx = transform.childPermutations[outChildIdx];
        const uint32_t bitmask2x2x2 = (bitmask4x4x4 >> (8 * inChildIdx)) & 0xFF;
        const uint64_t transformedBitmask2x2x2 = applyTransformToBitMask2x2x2(bitmask2x2x2, transform);
        out |= transformedBitmask2x2x2 << (outChildIdx * 8);
    }
    assert_always(std::popcount(out) == std::popcount(bitmask4x4x4));
    return out;
}

template <typename T>
static std::array<T, 8> applyTransformToArray(const std::array<T, 8>& items, const GenericTransformID& transform)
{
    std::array<T, 8> out;
    for (uint32_t outVoxelIdx = 0; outVoxelIdx < 8; ++outVoxelIdx) {
        const auto inVoxelIdx = transform.childPermutations[outVoxelIdx];
        out[outVoxelIdx] = items[inVoxelIdx];
    }
    return out;
}

struct GenericTaggedPointer {
    GenericTransformPointer ptr;
    InvarianceBitMask* pInvarianceBitMask;

    static GenericTaggedPointer sentinel()
    {
        return { .ptr = GenericTransformPointer::sentinel(), .pInvarianceBitMask = nullptr };
    }
    bool operator==(const GenericTaggedPointer& rhs) const
    {
        if (ptr.nodeIdx != rhs.ptr.nodeIdx)
            return false;
        if (ptr.nodeIdx == sentinel().ptr.nodeIdx)
            return true;
        assert(pInvarianceBitMask == rhs.pInvarianceBitMask);

        // Comparing lhs (*this) and rhs both referring to the same canonical representation C (pointer.nodeIdx).
        // The transformation T_L(C) and T_R(C) applied to lhs & rhs respectively, is described by their transformPointer.
        //
        // We know the invariances of the canonical representation C. In other words, we know for which transformations this holds:
        // T(C) = C
        //
        // To answer the question T_L(C) == T_R(C) we rewrite the equation as follows:
        // T_L(C) == T_R(C)
        // T_L^-1(T_L(C)) == T_L^-1(T_R(C))
        // C == T_L^-1(T_R(C))
        //
        // We thus need to find T=T_L^-1(T_R(C)) to tell whether C is invariant to it.
        const auto lhsInverseTransform = inverse(ptr.transform);
        const auto requiredInvariance = applyTransform(lhsInverseTransform, rhs.ptr.transform);
        assert(applyTransform(lhsInverseTransform, ptr.transform) == GenericTransformID());

        const auto requiredInvarianceID = encodeTransformID(requiredInvariance);
        return pInvarianceBitMask->test(requiredInvarianceID);
    }
};

[[maybe_unused]] static std::string bitmask2x2x2_to_string(uint32_t bitmask2x2x2)
{
    std::string out = "";
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            out += "[";
            for (uint32_t x = 0; x < 2; ++x) {
                const uint32_t idx = x | (y << 1) | (z << 2);
                out += ((bitmask2x2x2 >> idx) & 0b1) ? "x" : " ";
            }
            out += "]\n";
        }
        out += "\n";
    }
    return out;
}
[[maybe_unused]] static std::string bitmask4x4x4_to_string(uint64_t bitmask4x4x4)
{
    std::string out = "";
    for (uint32_t z = 0; z < 4; ++z) {
        for (uint32_t y = 0; y < 4; ++y) {
            out += "[";
            for (uint32_t x = 0; x < 4; ++x) {
                const uint32_t idx = morton_encode32(glm::uvec3(x, y, z));
                out += ((bitmask4x4x4 >> idx) & 0b1) ? "x" : " ";
            }
            out += "]\n";
        }
        out += "\n";
    }
    return out;
}

static std::tuple<std::array<GenericTaggedPointer, 256>, std::vector<InvarianceBitMask>> computeCanonicalLeafRepresentations()
{
    // Loop over each potential canonical 2x2x2 bitmask.
    // Apply every possible transformation, and store it as the selected representation if
    // it has the smallest value when interpreted as a uint8_t.
    std::vector<InvarianceBitMask> invarianceBitMasks((size_t)256);
    std::fill(std::begin(invarianceBitMasks), std::end(invarianceBitMasks), InvarianceBitMask {});
    std::array<GenericTaggedPointer, 256> out;
    std::fill(std::begin(out), std::end(out), GenericTaggedPointer { .ptr = { .nodeIdx = 0xFFFF } });
    for (uint32_t canonicalBitMask2x2x2 = 0; canonicalBitMask2x2x2 < 256; ++canonicalBitMask2x2x2) {
        //  Loop over all potential transformations of this 2x2x2 bitmask.
        for (uint32_t encodedtransformID = 0; encodedtransformID < NumTransformIDs; ++encodedtransformID) {
            const auto transformID = decodeTransformID(encodedtransformID);
            assert(encodeTransformID(transformID) == encodedtransformID);
            const auto transformedBitMask2x2x2 = applyTransformToBitMask2x2x2(canonicalBitMask2x2x2, transformID);

            if (canonicalBitMask2x2x2 < out[transformedBitMask2x2x2].ptr.nodeIdx) {
                out[transformedBitMask2x2x2] = GenericTaggedPointer {
                    .ptr = {
                        .nodeIdx = canonicalBitMask2x2x2,
                        .transform = transformID,
                    },
                    .pInvarianceBitMask = nullptr
                };
            }

            if (transformedBitMask2x2x2 == canonicalBitMask2x2x2)
                invarianceBitMasks[canonicalBitMask2x2x2].set(encodedtransformID);
        }
    }

    // Store invariances of the selected canonical representation in the TaggedPointers.
    for (GenericTaggedPointer& pointer : out)
        pointer.pInvarianceBitMask = &invarianceBitMasks[pointer.ptr.nodeIdx];

    std::unordered_set<uint64_t> uniquePtrs;
    for (const auto& ptr : out)
        uniquePtrs.insert(ptr.ptr.nodeIdx);
    spdlog::info("Num [sym+perm] canonical 2x2x2 grids: {}", uniquePtrs.size());
    return { out, std::move(invarianceBitMasks) };
}

struct GenericTransformNode {
    std::array<GenericTaggedPointer, 8> children;

    static GenericTransformNode fromBitmask4x4x4(uint64_t bitmask4x4x4, std::span<const GenericTaggedPointer> canonical2x2x2)
    {
        GenericTransformNode out {};
        for (uint32_t index = 0; index < 8; ++index) {
            const uint32_t bitmask2x2x2 = (bitmask4x4x4 >> (8 * index)) & 0xFF;
            out.children[index] = canonical2x2x2[bitmask2x2x2];
            assert(applyTransformToBitMask2x2x2(out.children[index].ptr.nodeIdx, out.children[index].ptr.transform) == bitmask2x2x2);
        }
        return out;
    }

    uint64_t toBitmask4x4x4() const
    {
        uint64_t out = 0;
        for (uint32_t index = 0; index < 8; ++index) {
            const auto transformedBitMask2x2x2 = applyTransformToBitMask2x2x2(children[index].ptr.nodeIdx, children[index].ptr.transform);
            out |= (uint64_t)transformedBitMask2x2x2 << (8 * index);
        }
        return out;
    }

    GenericTransformNode transform(const GenericTransformID& transform) const
    {
        static constexpr std::array baseNode { 0, 1, 2, 3, 4, 5, 6, 7 };

        GenericTransformNode out {};
        for (uint32_t outChildIdx = 0; outChildIdx < 8; ++outChildIdx) {
            const auto inChildIdx = transform.childPermutations[outChildIdx];
            auto childPointer = children[inChildIdx];
            // const auto test1 = applyTransformToArray(applyTransformToArray(baseNode, childPointer.transform), transform);
            childPointer.ptr.transform = applyTransform(transform, childPointer.ptr.transform);
            // const auto test2 = applyTransformToArray(baseNode, childPointer.transform);
            // assert_always(test1 == test2);
            out.children[outChildIdx] = childPointer;
        }
        return out;
    }

    bool operator==(const GenericTransformNode& rhs) const = default;
};
}

template <>
class fmt::formatter<voxcom::GenericTransformID> {
public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr inline auto format(const voxcom::GenericTransformID& transform, Context& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}", transform.childPermutations);
    }
};
template <>
class fmt::formatter<voxcom::GenericTaggedPointer> {
public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr inline auto format(const voxcom::GenericTaggedPointer& pointer, Context& ctx) const
    {
        return fmt::format_to(ctx.out(), "(nodeIdx = {}, transform = {})", pointer.ptr.nodeIdx, pointer.ptr.transform);
    }
};
template <>
class fmt::formatter<voxcom::GenericTransformNode> {
public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr inline auto format(const voxcom::GenericTransformNode& node, Context& ctx) const
    {
        auto iter = ctx.out();
        iter = fmt::format_to(iter, "{{");
        for (const auto& child : node.children) {
            iter = fmt::format_to(iter, "\t{}\n", child);
        }
        iter = fmt::format_to(iter, "}}");
        return iter;
    }
};

namespace voxcom {
struct GenericTransformNodeHash {
    inline size_t operator()(const GenericTransformNode& node) const noexcept
    {
        size_t seed = 0;
        for (const auto& child : node.children) {
            voxcom::hash_combine(seed, child.ptr.nodeIdx);
        }
        return seed;
    }
};

static void verify(const EditStructure<void, uint32_t>& originalDAG, const EditStructure<void, GenericTransformPointer>& transformDAG);

EditStructure<void, TransformPointer> findAllHierarchicalTransforms(const EditStructure<void, uint32_t>& inDag)
{
    using NodeMap = std::unordered_map<GenericTransformNode, GenericTaggedPointer, GenericTransformNodeHash>;

    // Find a matching node.
    const auto findTransformNode = [&](const GenericTransformNode& node, NodeMap& inOutUniqueNodes, InvarianceBitMask& outInvarianceBitMask) -> std::optional<GenericTaggedPointer> {
        // Find matching canonical node which can be transformed into this one.
        outInvarianceBitMask = {};
        for (uint32_t encodedTransformID = 0; encodedTransformID < NumTransformIDs; ++encodedTransformID) {
            const auto transformID = decodeTransformID(encodedTransformID);
            const auto transformedNode = node.transform(transformID);

            if (auto iter = inOutUniqueNodes.find(transformedNode); iter != std::end(inOutUniqueNodes)) {
                // We computed how to go from node => matchingNode but we need to store how to go from matchingNode => node.
                auto out = iter->second;
                out.ptr.transform = inverse(transformID);
                return out;
            }

            if (transformedNode == node)
                outInvarianceBitMask.set(encodedTransformID);
        }
        return {};
    };

    const auto processLevel = [&findTransformNode]<typename T>(std::span<const T> inItems, std::span<const GenericTransformNode> inNodes, std::vector<T>& outItems, std::span<GenericTaggedPointer> outPrevLevelMapping, std::span<InvarianceBitMask> outPrevLevelInvarianceBitmasks) {
        assert_always(outItems.size() == 0);
        assert_always(inItems.size() == inNodes.size());
        assert_always(outPrevLevelMapping.size() == inNodes.size());
#if 1 // Parallel vs easy to understand.
      // Create groups of TransformNodes with the same children pointers (order independent).
        std::vector<std::pair<size_t, size_t>> hashKeys(inItems.size());
        {
            std::vector<size_t> indices(inItems.size());
            std::iota(std::begin(indices), std::end(indices), 0);
            std::transform(std::execution::par, std::begin(inNodes), std::end(inNodes), std::begin(indices), std::begin(hashKeys),
                [](GenericTransformNode node, size_t index) {
                    std::sort(std::begin(node.children), std::end(node.children), [](const GenericTaggedPointer& lhs, const GenericTaggedPointer& rhs) { return lhs.ptr.nodeIdx < rhs.ptr.nodeIdx; });
                    size_t seed = 0;
                    for (const auto& child : node.children)
                        hash_combine(seed, child.ptr.nodeIdx);
                    return std::pair { seed, index };
                });
        }
        std::sort(std::execution::par, std::begin(hashKeys), std::end(hashKeys));
        // Sequentially loop over the sorted array to create groups/clusters.
        size_t prevKey = hashKeys[0].first;
        size_t prevI = 0;
        std::vector<std::pair<size_t, size_t>> ranges;
        for (size_t i = 0; i < hashKeys.size(); ++i) {
            const size_t key = hashKeys[i].first;
            if (key != prevKey) {
                ranges.push_back({ prevI, i });
                prevI = i;
                prevKey = key;
            }
        }
        ranges.push_back({ prevI, hashKeys.size() });

        // Search within each group of potentially matching nodes (in parallel).
        // std::vector<GenericTaggedPointer> uniqueIndices(inItems.size());
        std::vector<uint8_t> shouldOutputItems(inItems.size(), (uint8_t)0);
        std::for_each(std::execution::par, std::begin(ranges), std::end(ranges),
            [&](const std::pair<size_t, size_t>& range) {
                const auto [begin, end] = range;
                NodeMap uniqueNodesLUT;
                uniqueNodesLUT.reserve(end - begin);

                for (size_t j = begin; j < end; ++j) {
                    const auto index = hashKeys[j].second;
                    const auto& node = inNodes[index];
                    InvarianceBitMask invarianceBitMask;
                    if (auto optExistingNode = findTransformNode(node, uniqueNodesLUT, invarianceBitMask); optExistingNode.has_value()) {
                        outPrevLevelMapping[index] = optExistingNode.value();
                    } else {
                        outPrevLevelInvarianceBitmasks[index] = invarianceBitMask;
                        outPrevLevelMapping[index] = uniqueNodesLUT[node] = GenericTaggedPointer { .ptr = { .nodeIdx = (uint32_t)index, .transform = {} }, .pInvarianceBitMask = &outPrevLevelInvarianceBitmasks[index] };
                        shouldOutputItems[index] = 1;
                    }
                }
            });

        // Output only those nodes which are deemed unique.
        // Update the pointers to the nodes during this compaction step.
        // References to existing nodes may only occur in the sorted array *after* the unique node.
        for (const auto& [_, index] : hashKeys) {
            if (shouldOutputItems[index]) {
                outPrevLevelMapping[index].ptr.nodeIdx = (uint32_t)outItems.size();
                outItems.push_back(inItems[index]);
            } else {
                auto& taggedPointer = outPrevLevelMapping[index];
                taggedPointer.ptr.nodeIdx = outPrevLevelMapping[taggedPointer.ptr.nodeIdx].ptr.nodeIdx;
            }
        }
#else
        NodeMap uniqueNodesLUT;
        uniqueNodesLUT.reserve(inItems.size());
        InvarianceBitMask invarianceBitMask;
        for (size_t i = 0; i < inItems.size(); ++i) {
            const auto& node = inNodes[i];
            if (auto optExistingNode = findTransformNode(node, uniqueNodesLUT, invarianceBitMask); optExistingNode.has_value()) {
                outPrevLevelMapping[i] = optExistingNode.value();
            } else {
                const auto handle = (uint32_t)outItems.size();
                outPrevLevelInvarianceBitmasks[i] = invarianceBitMask;
                outPrevLevelMapping[i] = uniqueNodesLUT[node] = GenericTaggedPointer { .nodeIdx = handle, .transform = {}, .pInvarianceBitMask = &outPrevLevelInvarianceBitmasks[i] };
                outItems.push_back(inItems[i]);
            }
        }
#endif
    };

    EditStructure<void, GenericTransformPointer> intermediateStructure;
    intermediateStructure.resolution = inDag.resolution;
    intermediateStructure.rootLevel = inDag.rootLevel;

    // Find unique leaf subgrids under symmetry and axis permutations.
    std::vector<GenericTaggedPointer> prevLevelMapping(inDag.subGrids.size());
    std::vector<InvarianceBitMask> prevLevelInvarianceBitMasks, curLevelInvarianceBitMasks;
    curLevelInvarianceBitMasks.resize(inDag.subGrids.size());
    {
        const auto [canonicalRepresentations, canonicalInvarianceBitMasks] = computeCanonicalLeafRepresentations();
        std::vector<GenericTransformNode> inShiftNodes(inDag.subGrids.size());
        std::transform(std::execution::par, std::begin(inDag.subGrids), std::end(inDag.subGrids), std::begin(inShiftNodes),
            [&](const EditSubGrid<void>& subGrid) { return GenericTransformNode::fromBitmask4x4x4(subGrid.bitmask, canonicalRepresentations); });
        processLevel(std::span<const EditSubGrid<void>>(inDag.subGrids), inShiftNodes, intermediateStructure.subGrids, prevLevelMapping, curLevelInvarianceBitMasks);
        spdlog::info("[{}] Reduced SVO leaves from {} to {} using all permutations", intermediateStructure.subGridLevel, inDag.subGrids.size(), intermediateStructure.subGrids.size());
    }

    // Traverse inner nodes from the bottom up.
    intermediateStructure.nodesPerLevel.resize(inDag.nodesPerLevel.size());
    for (uint32_t level = inDag.subGridLevel + 1; level <= inDag.rootLevel; ++level) {
        const auto& inLevelNodes = inDag.nodesPerLevel[level];
        std::swap(prevLevelInvarianceBitMasks, curLevelInvarianceBitMasks);

        // Update child pointers.
        std::vector<GenericTransformNode> inShiftNodes(inLevelNodes.size());
        std::transform(std::execution::par, std::begin(inLevelNodes), std::end(inLevelNodes), std::begin(inShiftNodes),
            [&](const EditNode<uint32_t>& inNode) {
                GenericTransformNode outNode {};
                for (size_t i = 0; i < 8; ++i) {
                    const auto inChild = inNode.children[i];
                    if (inChild == inNode.EmptyChild)
                        outNode.children[i] = GenericTaggedPointer::sentinel();
                    else
                        outNode.children[i] = prevLevelMapping[inChild];
                }
                return outNode;
            });

        // Remove tag from child pointers.
        std::vector<EditNode<GenericTransformPointer>> inEditNodes(inLevelNodes.size());
        std::transform(std::execution::par, std::begin(inShiftNodes), std::end(inShiftNodes), std::begin(inEditNodes),
            [](const GenericTransformNode& node) {
                std::array<GenericTransformPointer, 8> children;
                // std::transform(std::begin(node.children), std::end(node.children), std::begin(children),
                //     [](const GenericTaggedPointer& ptr) { return convertPointer(ptr.ptr); });
                std::transform(std::begin(node.children), std::end(node.children), std::begin(children), [](const GenericTaggedPointer& ptr) { return ptr.ptr; });
                return EditNode<GenericTransformPointer> { .children = children };
            });

        auto& outLevelNodes = intermediateStructure.nodesPerLevel[level];
        prevLevelMapping.resize(inShiftNodes.size());
        curLevelInvarianceBitMasks.resize(inShiftNodes.size());
        processLevel(std::span<const EditNode<GenericTransformPointer>>(inEditNodes), inShiftNodes, outLevelNodes, prevLevelMapping, curLevelInvarianceBitMasks);
        spdlog::info("[{}] Reduced SVO nodes from {} to {} using all permutations", level, inLevelNodes.size(), outLevelNodes.size());
    }

    spdlog::info("Verifying...");
    verify(inDag, intermediateStructure);
    spdlog::info("SUCCESS!");

    // Convert from GenericTransformPointer to the more CompactTransformPointer format, which can only hold symmetry + axis permutation (+ translation).
    EditStructure<void, TransformPointer> out;
    out.resolution = inDag.resolution;
    out.rootLevel = inDag.rootLevel;
    out.subGrids = std::move(intermediateStructure.subGrids);
    out.nodesPerLevel.resize(intermediateStructure.nodesPerLevel.size());
    for (size_t level = 0; level < out.nodesPerLevel.size(); ++level) {
        const auto& inLevelNodes = intermediateStructure.nodesPerLevel[level];
        auto& outLevelNodes = out.nodesPerLevel[level];
        outLevelNodes.resize(inLevelNodes.size());
        std::transform(std::begin(inLevelNodes), std::end(inLevelNodes), std::begin(outLevelNodes),
            [](const EditNode<GenericTransformPointer>& inNode) {
                EditNode<TransformPointer> outNode;
                for (size_t childIdx = 0; childIdx < 8; ++childIdx)
                    outNode.children[childIdx] = convertPointer(inNode.children[childIdx]);
                return outNode;
            });
    }
    return out;
}

static void verify_recurse(
    uint32_t level,
    const EditStructure<void, uint32_t>& originalDAG, uint32_t originalNodeIdx,
    const EditStructure<void, GenericTransformPointer>& transformDAG, uint32_t transformNodeIdx, const GenericTransformID& transform)
{
    if (level == originalDAG.subGridLevel) {
        const uint64_t originalSubGrid = originalDAG.subGrids[originalNodeIdx].bitmask;
        const uint64_t transformedSubGrid = applyTransformToBitMask4x4x4(transformDAG.subGrids[transformNodeIdx].bitmask, transform);
        assert_always(originalSubGrid == transformedSubGrid);
    } else {
        const uint32_t childLevel = level - 1;
        for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
            const auto& originalChild = originalDAG.nodesPerLevel[level][originalNodeIdx].children[childIdx];
            const auto transformedChildIdx = transform.childPermutations[childIdx];
            const auto& transformedChild = transformDAG.nodesPerLevel[level][transformNodeIdx].children[transformedChildIdx];
            assert_always((originalChild == EditNode<uint32_t>::EmptyChild) == (transformedChild == EditNode<GenericTransformPointer>::EmptyChild));
            if (originalChild != EditNode<uint32_t>::EmptyChild) {
                verify_recurse(
                    childLevel,
                    originalDAG, originalChild,
                    transformDAG, transformedChild.nodeIdx, applyTransform(transform, transformedChild.transform));
            }
        }
    }
}

static void verify(const EditStructure<void, uint32_t>& originalDAG, const EditStructure<void, GenericTransformPointer>& transformDAG)
{
    verify_recurse(originalDAG.rootLevel, originalDAG, 0, transformDAG, 0, {});
}

}