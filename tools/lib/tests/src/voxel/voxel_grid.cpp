#include <catch2/catch_all.hpp>
#include "voxcom/attributes/color.h"
#include <ostream>
#include <voxcom/voxel/morton.h>
#include <voxcom/voxel/voxel_grid.h>

using namespace voxcom;
using namespace voxcom;

TEST_CASE("VoxelGrid<u8vec4> setters and getters", "[VoxelGrid]")
{
    SECTION("Basic")
    {
        VoxelGrid<RGB> grid { 3 };
        grid.set(glm::ivec3(0, 0, 0), RGB { 0, 0, 0 });
        grid.set(glm::ivec3(1, 0, 0), RGB { 0, 0, 0 });
        grid.set(glm::ivec3(0, 1, 0), RGB { 0, 0, 1 });
        grid.set(glm::ivec3(1, 1, 0), RGB { 0, 0, 1 });
        grid.set(glm::ivec3(0, 2, 0), RGB { 0, 1, 0 });
        grid.set(glm::ivec3(1, 2, 0), RGB { 0, 1, 0 });

        const auto getValue = [&](const auto& pos) {
            RGB out;
            REQUIRE(grid.get(pos, out));
            return out;
        };
        REQUIRE(getValue(glm::ivec3(0, 0, 0)) == RGB { 0, 0, 0 });
        REQUIRE(getValue(glm::ivec3(1, 0, 0)) == RGB { 0, 0, 0 });
        REQUIRE(getValue(glm::ivec3(0, 1, 0)) == RGB { 0, 0, 1 });
        REQUIRE(getValue(glm::ivec3(1, 1, 0)) == RGB { 0, 0, 1 });
        REQUIRE(getValue(glm::ivec3(0, 2, 0)) == RGB { 0, 1, 0 });
        REQUIRE(getValue(glm::ivec3(1, 2, 0)) == RGB { 0, 1, 0 });
    }
}

TEST_CASE("VoxelGrid<void> setters and getters", "[VoxelGrid]")
{
    SECTION("Random 16x16x16")
    {
        constexpr unsigned resolution = 16;
        constexpr unsigned resolution3 = resolution * resolution * resolution;

        // Value for each pixel.
        const auto voxelValues = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));

        VoxelGrid<void> voxelGrid { resolution };

        REQUIRE(voxelGrid.resolution == resolution);

        for (unsigned i = 0; i < resolution3; i++) {
            if (!voxelValues[i])
                continue;

            const glm::uvec3 voxel = morton_decode32<3>(i);
            voxelGrid.set(voxel);
        }

        for (unsigned i = 0; i < resolution3; i++) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            CAPTURE(voxel.x, voxel.y, voxel.z, voxelValues[i]);
            REQUIRE(voxelGrid.get(voxel) == (bool)voxelValues[i]);
        }
    }
}
