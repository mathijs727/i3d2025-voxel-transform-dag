#pragma once
#include "voxcom/utility/fmt_glm.h"
#include "voxcom/voxel/attributes.h"
#include <type_traits>
#include <vector>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

namespace voxcom {

template <typename Attribute_>
struct VoxelGrid {
public:
    using Attribute = Attribute_;

    unsigned resolution;
    std::vector<bool> filled;
    attribute_vector_t<Attribute> attributes;
    size_t strideY, strideZ;

public:
    constexpr VoxelGrid(unsigned resolution)
        : resolution(resolution)
    {
        filled.resize((size_t)resolution * resolution * resolution, false);
        if constexpr (!std::is_void_v<Attribute>)
            attributes.resize(filled.size());

        // strideX = 1
        strideY = resolution;
        strideZ = (size_t)resolution * resolution;
    }

    constexpr void set(const glm::ivec3& voxel) noexcept
        requires(std::is_void_v<Attribute>)
    {
        const size_t index = getIndex(voxel);
        if (index >= filled.size())
            return;
        filled[index] = true;
    }
    constexpr void set(const glm::ivec3& voxel, std::conditional_t<std::is_void_v<Attribute>, int, Attribute> value) noexcept
        requires(!std::is_void_v<Attribute>)
    {
        const size_t index = getIndex(voxel);
        if (index >= filled.size())
            return;
        filled[index] = true;
        attributes[index] = value;
    }

    [[nodiscard]] constexpr bool get(const glm::ivec3& voxel, std::conditional_t<std::is_void_v<Attribute>, int, Attribute>& attribute) const noexcept
    {
        const size_t index = getIndex(voxel);
        if (index >= filled.size() || !filled[index])
            return false;

        attribute = attributes[index];
        return true;
    }
    [[nodiscard]] constexpr bool get(const glm::ivec3& voxel) const noexcept
    {
        const size_t index = getIndex(voxel);
        return index < filled.size() && filled[index];
    }

    constexpr size_t getIndex(const glm::ivec3& voxel) const noexcept
    {
        return voxel.z * strideZ + voxel.y * strideY + voxel.x;
    }

    constexpr size_t computeMemoryUsage() const noexcept
    {
        size_t out = filled.size() / 8; // std::vector<bool> is specialized to use 1 bit per value.
        if (!std::is_void_v<Attribute>)
            out += attributes.size() * sizeof(Attribute);
        return out;
    }

    constexpr VoxelGrid downSample2() const noexcept
    {
        VoxelGrid out { resolution / 2 };
        for (uint32_t z = 0; z < resolution; ++z) {
            for (uint32_t y = 0; y < resolution; ++y) {
                for (uint32_t x = 0; x < resolution; ++x) {
                    const glm::ivec3 voxel { x, y, z };
                    const glm::ivec3 halfVoxel = voxel / 2;
                    if (get(voxel))
                        out.set(halfVoxel);
                }
            }
        }
        return out;
    }

    constexpr VoxelGrid upSample2() const noexcept
    {
        VoxelGrid out { 2 * resolution };
        for (uint32_t z = 0; z < resolution; ++z) {
            for (uint32_t y = 0; y < resolution; ++y) {
                for (uint32_t x = 0; x < resolution; ++x) {
                    const glm::uvec3 voxel { x, y, z };
                    const glm::uvec3 parentVoxel = 2u * voxel;
                    if (get(voxel)) {
                        for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
                            out.set(parentVoxel + morton_decode32<3>(childIdx));
                        }
                    }
                }
            }
        }
        return out;
    }
};

}