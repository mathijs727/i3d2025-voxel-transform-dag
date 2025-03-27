#pragma once
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <glm/vec3.hpp>
#include <vector>
#include <voxcom/core/bounds.h>
#include <voxcom/utility/error_handling.h>
#include <voxcom/utility/size_of.h>
#include <voxcom/voxel/morton.h>
#include <voxcom/voxel/structure.h>

namespace voxcom {

template <bool ExtendedInvariance>
struct invariance_mask { };
template <>
struct invariance_mask<true> {
    using type = uint32_t;
    static constexpr type no_invariance = 1;
};
template <>
struct invariance_mask<false> {
    using type = glm::bvec3;
    static constexpr type no_invariance = glm::bvec3(false);
};

// Referred to as "tagged pointer" in the original paper.
template <bool ExtendedInvariance>
struct SymmetryPointer {
    uint32_t ptr;
    glm::bvec3 transform;

    typename invariance_mask<ExtendedInvariance>::type invariance;

    static constexpr SymmetryPointer sentinel()
    {
        if constexpr (ExtendedInvariance) {
            return { .ptr = 0xFFFF'FFFF, .transform = glm::bvec3(false), .invariance = 0xFF };
        } else {
            return { .ptr = 0xFFFF'FFFF, .transform = glm::bvec3(false), .invariance = glm::bvec3(true) };
        }
    }

    bool operator==(const SymmetryPointer& rhs) const;
    // Read/write pointer, keeping the transform/invariance as-is.
    explicit operator uint32_t() const;
    SymmetryPointer& operator=(uint32_t p);
};
template <bool ExtendedInvariance>
struct SymmetryNode {
    std::array<SymmetryPointer<ExtendedInvariance>, 8> children;

    static SymmetryNode fromBitmask4x4x4(uint64_t bitmask4x4x4);
    uint64_t toBitmask4x4x4() const;

    SymmetryNode mirror(bool mirrorX, bool mirrorY, bool mirrorZ) const;

    bool operator==(const SymmetryNode& rhs) const = default;
};

template <bool ExtendedInvariance, template <typename, typename> typename Structure>
EditStructure<void, SymmetryPointer<ExtendedInvariance>> constructSSVDAG(const Structure<void, uint32_t>& octree);

template <bool ExtendedInvariance, template <typename, typename> typename Structure>
void verifySSVDAG(const Structure<void, uint32_t>& octree, const EditStructure<void, SymmetryPointer<ExtendedInvariance>>& ssvdag);

}

