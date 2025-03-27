#include "voxcom/attributes/color.h"
#include <algorithm> // shuffle
#include <bit> // popcount
#include <catch2/catch_all.hpp>
#include <iostream>
#include <numeric> // iota
#include <random>
#include <voxcom/format/fmt_glm.h>
#include <voxcom/voxel/encoded_octree.h>
#include <voxcom/voxel/octree.h>

using namespace voxcom;
using namespace voxcom;

/* TEST_CASE("Construct EncodedOctree<bool> from Octree<void>", "[EncodedOctree][Octree]")
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
    EncodedOctree encodedOctree { octree };

    SECTION("Basic member variables match")
    {
        REQUIRE(encodedOctree.resolution == octree.resolution);
        REQUIRE(encodedOctree.nodesPerLevel.size() == octree.nodesPerLevel.size());
    }
    SECTION("Encoded tree is smaller than original")
    {
        for (size_t level = 0; level < octree.nodesPerLevel.size(); level++) {
            const size_t originalLevelSizeBytes = octree.nodesPerLevel[level].size() * sizeof(OctreeNode<bool>);
            const size_t encodedLevelSizeBytes = encodedOctree.nodesPerLevel[level].size() * sizeof(decltype(encodedOctree.nodesPerLevel)::value_type::value_type);
            CAPTURE(level);
            REQUIRE(encodedLevelSizeBytes <= originalLevelSizeBytes);
        }
    }

    SECTION("Re-encoded octree is the same as the original")
    {
        const auto reencodedOctree = static_cast<Octree<void>>(encodedOctree);
        REQUIRE(reencodedOctree == octree);
    }
} */

TEST_CASE("Construct EncodedOctree<RGB> from Octree<RGB>", "[EncodedOctree][Octree]")
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
    EncodedOctree<RGB> encodedOctree { octree };

    for (unsigned i = 0; i < resolution3; i++) {
        const glm::uvec3 voxel = morton_decode32<3>(i);

        const bool expectedFilled = voxelFilled[i];
        const RGB expectedValue { (uint8_t)voxelValues[i * 3 + 0], (uint8_t)voxelValues[i * 3 + 1], (uint8_t)voxelValues[i * 3 + 2] };

        RGB actualValue;
        const bool actualFilled = octree.get(voxel, actualValue);
        REQUIRE(actualFilled == expectedFilled);
        if (actualFilled) {
            REQUIRE(actualValue.r == expectedValue.r);
            REQUIRE(actualValue.g == expectedValue.g);
            REQUIRE(actualValue.b == expectedValue.b);
        }
    }

    SECTION("Basic member variables match")
    {
        REQUIRE(encodedOctree.resolution == octree.resolution);
        REQUIRE(encodedOctree.nodesPerLevel.size() == octree.nodesPerLevel.size());
    }
    SECTION("Encoded tree is smaller than original")
    {
        const size_t originalSize = octree.computeMemoryUsage();
        const size_t encodedSize = encodedOctree.computeMemoryUsage();
        REQUIRE(encodedSize <= originalSize);
        CAPTURE(encodedSize, originalSize);
    }

    SECTION("Re-encoded octree is the same as the original")
    {
        const auto reencodedOctree = static_cast<Octree<RGB>>(encodedOctree);
        REQUIRE(reencodedOctree.treeType == TreeType::Tree);
        REQUIRE(reencodedOctree == octree);
    }
}

static void shuffleOctree(EncodedOctree<RGB>& encodedOctree)
{
    // Shuffle the leaf node attributes
    auto& leafNodes = encodedOctree.nodesPerLevel[encodedOctree.subGridLevel];
    std::vector<size_t> indices(leafNodes.size() / 2);
    std::iota(std::begin(indices), std::end(indices), 0);

    std::random_device rd;
    std::mt19937 g { rd() };
    std::shuffle(std::begin(indices), std::end(indices), g);

    using Node = typename EncodedOctree<RGB>::Node;
    std::vector<Node> newLeafNodes(leafNodes.size());
    std::vector<RGB> newAttributes;
    for (size_t oldNodeIdx : indices) {
        const auto bitmask = leafNodes[oldNodeIdx * 2 + 0].bitmask4x4x4;
        const auto oldLeafPtr = leafNodes[oldNodeIdx * 2 + 1].firstAttributePtr;
        leafNodes[oldNodeIdx * 2 + 1].firstAttributePtr = (uint64_t)newAttributes.size();
        for (int i = 0; i < std::popcount(bitmask); i++)
            newAttributes.push_back(encodedOctree.attributes[oldLeafPtr + i]);
    }
    encodedOctree.attributes = std::move(newAttributes);
}

TEST_CASE("Shuffle attributes of EncodedOctree<RGB>", "[EncodedOctree][Octree]")
{
    constexpr unsigned resolution = 16;
    constexpr unsigned resolution3 = resolution * resolution * resolution;

    // Value for each pixel.
    const auto voxelValues = GENERATE(take(3, chunk(resolution3 * 3, random<uint32_t>(0, 255))));
    Octree<RGB> octree { resolution };
    for (unsigned i = 0; i < resolution3; i++) {
        const glm::uvec3 voxel = morton_decode32<3>(i);
        const RGB voxelValue { (uint8_t)voxelValues[i * 3 + 0], (uint8_t)voxelValues[i * 3 + 1], (uint8_t)voxelValues[i * 3 + 2] };
        octree.set(voxel, voxelValue);
    }
    EncodedOctree encodedOctree { octree };

    // A shuffle could randomly create the exact same tree (aka not move anything at all).
    // Check for this case and shuffle again to make sure that the test is not effectively skipped.
    while (true) {
        const auto oldAttributes = encodedOctree.attributes;
        shuffleOctree(encodedOctree);
        if (!std::equal(std::begin(encodedOctree.attributes), std::end(encodedOctree.attributes), std::begin(oldAttributes)))
            break;
    }

    SECTION("Traversing new tree gives same results")
    {
        const auto shuffledOctree = static_cast<Octree<RGB>>(encodedOctree);
        for (unsigned i = 0; i < resolution3; i++) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            const RGB expectedValue { (uint8_t)voxelValues[i * 3 + 0], (uint8_t)voxelValues[i * 3 + 1], (uint8_t)voxelValues[i * 3 + 2] };
            RGB actualValue;
            REQUIRE(shuffledOctree.get(voxel, actualValue));
            REQUIRE(actualValue == expectedValue);
        }
    }
}

TEST_CASE("Sort attributes of EncodedOctree<RGB> in depth first order", "[EncodedOctree][Octree]")
{
    constexpr unsigned resolution = 16;
    constexpr unsigned resolution3 = resolution * resolution * resolution;

    // Value for each pixel.
    const auto voxelValues = GENERATE(take(3, chunk(resolution3 * 3, random<uint32_t>(0, 255))));
    Octree<RGB> octree { resolution };
    for (unsigned i = 0; i < resolution3; i++) {
        const glm::uvec3 voxel = morton_decode32<3>(i);
        const RGB voxelValue { (uint8_t)voxelValues[i * 3 + 0], (uint8_t)voxelValues[i * 3 + 1], (uint8_t)voxelValues[i * 3 + 2] };
        octree.set(voxel, voxelValue);
    }
    EncodedOctree encodedOctree { octree };

    // A shuffle could randomly create a tree that is already depth-first ordered.
    // Check for this case and shuffle again to make sure that the test is not effectively skipped.
    while (true) {
        shuffleOctree(encodedOctree);
        const auto oldAttributes = encodedOctree.attributes;
        encodedOctree.reorderAttributesDepthFirst();
        if (!std::equal(std::begin(encodedOctree.attributes), std::end(encodedOctree.attributes), std::begin(oldAttributes)))
            break;
    }

    SECTION("Traversing new tree gives same results")
    {
        const auto sortedOctree = static_cast<Octree<RGB>>(encodedOctree);
        for (unsigned i = 0; i < resolution3; i++) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            const RGB expectedValue { (uint8_t)voxelValues[i * 3 + 0], (uint8_t)voxelValues[i * 3 + 1], (uint8_t)voxelValues[i * 3 + 2] };
            RGB actualValue;
            REQUIRE(sortedOctree.get(voxel, actualValue));
            REQUIRE(actualValue == expectedValue);
        }
    }
}
