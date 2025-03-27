#include <catch2/catch_all.hpp>
#include <cstring> // memcmp
#include <type_traits>
#include <voxcom/voxel/structure.h>
#include <voxcom/voxel/export_import_structure.h>

using namespace voxcom;

TEST_CASE("EditStructure<void, uint32_t> save/load", "[EditStructure]")
{
    constexpr unsigned resolution = 16;
    constexpr unsigned resolution3 = resolution * resolution * resolution;
    // Value for each pixel.
    const auto voxelValues = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));
    EditStructure<void, uint32_t> octree { resolution };
    for (unsigned i = 0; i < resolution3; i++) {
        if (voxelValues[i]) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            octree.set(voxel);
        }
    }

    const std::filesystem::path filePath = "test_octree_bool.bin";
    exportStructure(octree, filePath);

    const auto loadedOctree = importEditStructure<void>(filePath);
    REQUIRE(octree.resolution == loadedOctree.resolution);
    REQUIRE(octree.structureType == loadedOctree.structureType);
    REQUIRE(octree.nodesPerLevel.size() == loadedOctree.nodesPerLevel.size());
    for (size_t level = 0; level < octree.nodesPerLevel.size(); level++) {
        REQUIRE(octree.nodesPerLevel[level].size() == loadedOctree.nodesPerLevel[level].size());
        for (size_t nodeIdx = 0; nodeIdx < octree.nodesPerLevel[level].size(); nodeIdx++) {
            REQUIRE(octree.nodesPerLevel[level][nodeIdx] == loadedOctree.nodesPerLevel[level][nodeIdx]);
        }
    }
}

/* TEST_CASE("EncodedOctree<void> save/load", "[EditStructure][EncodedOctree]")
{
    constexpr unsigned resolution = 16;
    constexpr unsigned resolution3 = resolution * resolution * resolution;
    // Value for each pixel.
    const auto voxelValues = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));
    EditStructure<void> octree { resolution };
    for (unsigned i = 0; i < resolution3; i++) {
        if (voxelValues[i]) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            octree.set(voxel, true);
        }
    }

    EncodedOctree<void> encodedOctree { octree };

    const std::filesystem::path filePath = "test_encoded_octree_void.bin";
    saveOctree(encodedOctree, filePath);

    const auto loadedOctree = loadEncodedOctree<void>(filePath);
    REQUIRE(encodedOctree.resolution == loadedOctree.resolution);
    REQUIRE(encodedOctree.nodesPerLevel.size() == loadedOctree.nodesPerLevel.size());
    for (size_t level = 0; level < octree.nodesPerLevel.size(); level++) {
        const auto& nodesPerLevel = encodedOctree.nodesPerLevel[level];
        const auto& loadedNodesPerLevel = loadedOctree.nodesPerLevel[level];
        CAPTURE(level, nodesPerLevel, loadedNodesPerLevel);
        REQUIRE(nodesPerLevel.size() == loadedNodesPerLevel.size());
        REQUIRE(std::memcmp(nodesPerLevel.data(), loadedNodesPerLevel.data(), nodesPerLevel.size() * sizeof(std::remove_reference_t<decltype(nodesPerLevel)>::value_type)) == 0);
    }
} */
