#include <catch2/catch_all.hpp>
#include <voxcom/core/image.h>
#include <voxcom/voxel/morton.h>
#include <voxcom/voxel/structure.h>
#include <voxcom/voxel/voxel_grid.h>

using namespace voxcom;

namespace voxcom {
std::ostream& operator<<(std::ostream& stream, const EditStructure<void, uint32_t>& octree)
{
    stream << "EditStructure { ";
    stream << "resolution = " << octree.resolution << ", ";
    stream << "structureType = " << (int)octree.structureType << ", ";
    stream << "}";
    return stream;
}
}

TEST_CASE("EditStructure<void, uint32_t> default constructor", "[EditStructure]")
{
    EditStructure<void, uint32_t> octree {};
    REQUIRE(octree.nodesPerLevel.empty());
    REQUIRE(octree.resolution == 0);
    REQUIRE(octree.structureType == StructureType::Tree);
}

TEST_CASE("Construct EditStructure<void, uint32_t> using set() method", "[EditStructure]")
{
    SECTION("16x16x16 Random")
    {
        constexpr unsigned resolution = 16;
        constexpr unsigned resolution3 = resolution * resolution * resolution;

        // Value for each pixel.
        const auto voxelValues = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));

        EditStructure<void, uint32_t> octree { resolution };
        REQUIRE(octree.nodesPerLevel.size() == 5);
        REQUIRE(octree.resolution == resolution);
        REQUIRE(octree.structureType == StructureType::Tree);
        REQUIRE(octree.structureType == StructureType::Tree);

        for (unsigned i = 0; i < resolution3; i++) {
            if (!voxelValues[i])
                continue;

            const glm::uvec3 voxel = morton_decode32<3>(i);
            octree.set(voxel);
        }

        // Lowest levels should be empty because we store 4x4x4 regions as a single leaf.
        REQUIRE(octree.nodesPerLevel[0].empty());
        REQUIRE(octree.nodesPerLevel[1].empty());
        REQUIRE(octree.nodesPerLevel[2].empty());

        for (unsigned i = 0; i < resolution3; i++) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            CAPTURE(voxel.x, voxel.y, voxel.z, voxelValues[i]);
            REQUIRE(octree.get(voxel) == (bool)voxelValues[i]);
        }
    }
}

TEST_CASE("Construct EditStructure<void, uint32_t> from voxel grid", "[EditStructure][VoxelGrid]")
{
    SECTION("Random 16x16x16")
    {
        constexpr unsigned resolution = 16;
        constexpr unsigned resolution3 = resolution * resolution * resolution;

        // Value for each pixel.
        const auto voxelValues = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));

        VoxelGrid<void> voxelGrid { resolution };
        for (unsigned i = 0; i < resolution3; i++) {
            if (!voxelValues[i])
                continue;

            const glm::uvec3 voxel = morton_decode32<3>(i);
            voxelGrid.set(voxel);
        }
        const EditStructure<void, uint32_t> octree { voxelGrid };

        for (unsigned i = 0; i < resolution3; i++) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            CAPTURE(voxel.x, voxel.y, voxel.z, voxelValues[i]);
            REQUIRE(octree.get(voxel) == (bool)voxelValues[i]);
        }
    }
}
