#include <catch2/catch_all.hpp>
#include <voxcom/voxel/atomic_encoded_octree.h>
#include <voxcom/voxel/encoded_octree.h>
#include <voxcom/voxel/octree.h>
#include "voxcom/attributes/color.h"
#include <iostream>

using namespace voxcom;
using namespace voxcom;

/*TEST_CASE("Construct AtomicEncodedOctree<void> from Octree<void>", "[AtomicEncodedOctree][EncodedOctree][Octree]")
{
    constexpr unsigned resolution = 16;
    constexpr unsigned resolution3 = resolution * resolution * resolution;

    // Value for each pixel.
    const auto voxelValues = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));
    Octree<void> octree { resolution };
    for (unsigned i = 0; i < resolution3; i++) {
        if (voxelValues[i])
            octree.set(morton_decode32<3>(i), true);
    }
    AtomicEncodedOctree encodedOctree { EncodedOctree { octree } };

    SECTION("Basic member variables match")
    {
        REQUIRE(encodedOctree.resolution == octree.resolution);
        REQUIRE(encodedOctree.treeType == octree.treeType);
        REQUIRE(encodedOctree.memoryPerLevel.size() == octree.nodesPerLevel.size());
    }
    SECTION("Encoded tree is smaller than original")
    {
        for (size_t level = 0; level < octree.nodesPerLevel.size(); level++) {
            const size_t originalLevelSizeBytes = octree.nodesPerLevel[level].size() * sizeof(OctreeNode<bool>);
            const size_t encodedLevelSizeBytes = encodedOctree.memoryPerLevel[level].size() * sizeof(uint32_t);
            CAPTURE(level);
            REQUIRE(encodedLevelSizeBytes <= originalLevelSizeBytes);
        }
    }

    SECTION("Re-encoded octree is the same as the original")
    {
        const auto reencodedOctree = static_cast<Octree<void>>(static_cast<EncodedOctree<void>>(encodedOctree));
        REQUIRE(reencodedOctree == octree);
    }
}*/

TEST_CASE("Construct AtomicEncodedOctree<RGB> from Octree<RGB>", "[AtomicEncodedOctree][EncodedOctree][Octree]")
{
    constexpr unsigned resolution = 16;
    constexpr unsigned resolution3 = resolution * resolution * resolution;

    // Value for each pixel.
    const auto voxelFilled = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));
    const auto voxelValues = GENERATE(take(3, chunk(resolution3 * 3, random<uint32_t>(0, 255))));
    Octree<RGB> octree { resolution };
    for (unsigned i = 0; i < resolution3; i++) {
        if (voxelFilled[i]) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            const RGB voxelValue { (uint8_t)voxelValues[i * 3 + 0], (uint8_t)voxelValues[i * 3 + 1], (uint8_t)voxelValues[i * 3 + 2] };
            octree.set(voxel, voxelValue);
        }
    }
    AtomicEncodedOctree encodedOctree { EncodedOctree { octree } };

    SECTION("Basic member variables match")
    {
        REQUIRE(encodedOctree.resolution == octree.resolution);
        REQUIRE(encodedOctree.nodesPerLevel.size() == octree.nodesPerLevel.size());
    }
    SECTION("Encoded tree is smaller than original")
    {
        const size_t originalSize = octree.computeMemoryUsage();
        const size_t encodedSize = encodedOctree.toEncodedOctree().computeMemoryUsage();
        REQUIRE(encodedSize <= originalSize);
        CAPTURE(encodedSize, originalSize);
    }

    SECTION("Re-encoded octree is the same as the original")
    {
        const auto reencodedOctree = static_cast<Octree<RGB>>(encodedOctree.toEncodedOctree());
        REQUIRE(reencodedOctree.treeType == TreeType::Tree);
        REQUIRE(reencodedOctree == octree);
    }
}
