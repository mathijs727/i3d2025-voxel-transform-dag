#pragma once
#include "voxcom/utility/my_cuda.h"
#include "voxcom/utility/disable_all_warnings.h"
DISABLE_WARNINGS_PUSH()
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <libmorton/morton.h>
DISABLE_WARNINGS_POP()

namespace voxcom {

inline auto morton_encode32(const glm::uvec2& xy) { return libmorton::morton2D_32_encode(xy.x, xy.y); }
inline auto morton_encode32(const glm::uvec3& xyz) { return libmorton::morton3D_32_encode(xyz.x, xyz.y, xyz.z); }
inline auto morton_encode64(const glm::uvec2& xy) { return libmorton::morton2D_64_encode(xy.x, xy.y); }
inline auto morton_encode64(const glm::uvec3& xyz) { return libmorton::morton3D_64_encode(xyz.x, xyz.y, xyz.z); }
template <size_t Dims>
auto morton_decode32(uint32_t);
template <>
inline auto morton_decode32<2>(uint32_t morton)
{
    uint_fast16_t x, y;
    libmorton::morton2D_32_decode(morton, x, y);
    return glm::uvec2(x, y);
}
template <>
inline auto morton_decode32<3>(uint32_t morton)
{
    uint_fast16_t x, y, z;
    libmorton::morton3D_32_decode(morton, x, y, z);
    return glm::uvec3(x, y, z);
}
template <size_t Dims>
inline auto morton_decode64(uint64_t);
template <>
inline auto morton_decode64<2>(uint64_t morton)
{
    uint_fast32_t x, y;
    libmorton::morton2D_64_decode(morton, x, y);
    return glm::uvec2(x, y);
}
template <>
inline auto morton_decode64<3>(uint64_t morton)
{
    uint_fast32_t x, y, z;
    libmorton::morton3D_64_decode(morton, x, y, z);
    return glm::uvec3(x, y, z);
}

}