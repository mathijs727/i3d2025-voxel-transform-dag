#include "voxcom/attributes/color.h"
#include <catch2/catch_all.hpp>
#include <voxcom/attributes/attributes.h>
#include <voxcom/voxel/octree.h>

using namespace voxcom;
using namespace voxcom;

TEST_CASE("Collect attributes from Octree<RGB>", "[Octree]")
{
    SECTION("16x16x16 Random")
    {
        constexpr unsigned resolution = 16;
        constexpr unsigned resolution3 = resolution * resolution * resolution;

        // Value for each pixel.
        const auto voxelFilled = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));
        const auto voxelValues = GENERATE(take(3, chunk(resolution3 * 3, random<uint32_t>(0, 255))));

        Octree<RGB> octree { resolution };
        REQUIRE(octree.nodesPerLevel.size() == 5);
        REQUIRE(octree.resolution == resolution);
        REQUIRE(octree.treeType == TreeType::Tree);

        for (unsigned i = 0; i < resolution3; i++) {
            if (!voxelFilled[i])
                continue;
            const glm::uvec3 voxel = morton_decode32<3>(i);
            const RGB voxelValue { (uint8_t)voxelValues[i * 3 + 0], (uint8_t)voxelValues[i * 3 + 1], (uint8_t)voxelValues[i * 3 + 2] };
            octree.set(voxel, voxelValue);
        }

        const auto attributes = collectAttributesDepthFirst(octree);
        size_t idx = 0;
        for (unsigned i = 0; i < resolution3; i++) {
            if (!voxelFilled[i])
                continue;
            const glm::uvec3 voxel = morton_decode32<3>(i);
            RGB originalAttribute;
            REQUIRE(octree.get(voxel, originalAttribute));
            const RGB attribute = attributes[idx++];
            REQUIRE(attribute.r == originalAttribute.r);
            REQUIRE(attribute.g == originalAttribute.g);
            REQUIRE(attribute.b == originalAttribute.b);
        }
        REQUIRE(idx == attributes.size());
    }
}

TEST_CASE("Convert attribute octree (RGB) to Octree<void>", "[Octree]")
{
    SECTION("16x16x16 Random")
    {
        constexpr unsigned resolution = 16;
        constexpr unsigned resolution3 = resolution * resolution * resolution;

        // Value for each pixel.
        const auto voxelFilled = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));

        Octree<RGB> octree { resolution };
        REQUIRE(octree.nodesPerLevel.size() == 5);
        REQUIRE(octree.resolution == resolution);
        REQUIRE(octree.treeType == TreeType::Tree);

        for (unsigned i = 0; i < resolution3; i++) {
            if (!voxelFilled[i])
                continue;
            const glm::uvec3 voxel = morton_decode32<3>(i);
            const RGB voxelValue { 1, 2, 3 };
            octree.set(voxel, voxelValue);
        }

        const auto strippedOctree = stripAttributes(octree);
        REQUIRE(strippedOctree.nodesPerLevel.size() == 5);
        REQUIRE(strippedOctree.resolution == resolution);
        REQUIRE(strippedOctree.treeType == TreeType::Tree);

        for (unsigned i = 0; i < resolution3; i++) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            REQUIRE(octree.nodesPerLevel.size() == 5);
            const bool filled = strippedOctree.get(voxel);
            CAPTURE(i);
            REQUIRE(filled == (bool)voxelFilled[i]);
        }
    }
}

TEST_CASE("Transform attribute of octree", "[Octree]")
{
    SECTION("16x16x16 Random")
    {
        constexpr unsigned resolution = 16;
        constexpr unsigned resolution3 = resolution * resolution * resolution;

        // Value for each pixel.
        const auto voxelFilled = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));
        const auto voxelValues = GENERATE(take(3, chunk(resolution3 * 3, random<uint32_t>(0, 255))));

        Octree<RGB> octree { resolution };
        REQUIRE(octree.nodesPerLevel.size() == 5);
        REQUIRE(octree.resolution == resolution);
        REQUIRE(octree.treeType == TreeType::Tree);

        for (unsigned i = 0; i < resolution3; i++) {
            if (!voxelFilled[i])
                continue;
            const glm::uvec3 voxel = morton_decode32<3>(i);
            const RGB voxelValue { (uint8_t)voxelValues[i * 3 + 0], (uint8_t)voxelValues[i * 3 + 1], (uint8_t)voxelValues[i * 3 + 2] };
            octree.set(voxel, voxelValue);
        }

        const auto redChannelOctree = transformAttributes<uint8_t>(
            octree,
            [](const RGB& color) -> uint8_t {
                return color.r;
            });
        REQUIRE(redChannelOctree.nodesPerLevel.size() == 5);
        REQUIRE(redChannelOctree.resolution == resolution);
        REQUIRE(redChannelOctree.treeType == TreeType::Tree);

        for (unsigned i = 0; i < resolution3; i++) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            uint8_t c;
            const bool filled = redChannelOctree.get(voxel, c);
            CAPTURE(i, filled);
            REQUIRE(filled == (bool)voxelFilled[i]);
            if (filled) {
                const uint8_t referenceRedChannel = (uint8_t)voxelValues[i * 3 + 0];
                REQUIRE(c == referenceRedChannel);
            }
        }
    }
}

TEST_CASE("Transform attribute of octree into booleans", "[Octree]")
{
    SECTION("16x16x16 Random")
    {
        constexpr unsigned resolution = 16;
        constexpr unsigned resolution3 = resolution * resolution * resolution;

        // Value for each pixel.
        const auto voxelFilled = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));
        const auto voxelValues = GENERATE(take(3, chunk(resolution3 * 3, random<uint32_t>(0, 255))));

        Octree<RGB> octree { resolution };
        REQUIRE(octree.nodesPerLevel.size() == 5);
        REQUIRE(octree.resolution == resolution);
        REQUIRE(octree.treeType == TreeType::Tree);

        for (unsigned i = 0; i < resolution3; i++) {
            if (!voxelFilled[i])
                continue;
            const glm::uvec3 voxel = morton_decode32<3>(i);
            const RGB voxelValue { (uint8_t)voxelValues[i * 3 + 0], (uint8_t)voxelValues[i * 3 + 1], (uint8_t)voxelValues[i * 3 + 2] };
            octree.set(voxel, voxelValue);
        }

        const auto redChannelBitOctree = transformAttributesToBinary(
            octree,
            [](const RGB& color) -> bool {
                return color.r & 0b1;
            });
        REQUIRE(redChannelBitOctree.nodesPerLevel.size() == 5);
        REQUIRE(redChannelBitOctree.resolution == resolution);
        REQUIRE(redChannelBitOctree.treeType == TreeType::Tree);

        for (unsigned i = 0; i < resolution3; i++) {
            CAPTURE(i);
            const glm::uvec3 voxel = morton_decode32<3>(i);
            const uint8_t referenceRedChannel = (uint8_t)voxelValues[i * 3 + 0];
            const bool referenceRedChannelBit = referenceRedChannel & 0b1;
            if (voxelFilled[i] && referenceRedChannelBit) {
                REQUIRE(redChannelBitOctree.get(voxel));
            } else {
                REQUIRE(!redChannelBitOctree.get(voxel));
            }
        }
    }
}
