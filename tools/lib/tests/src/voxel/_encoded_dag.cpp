#include "voxcom/attributes/color.h"
#include <catch2/catch_all.hpp>
#include <voxcom/voxel/encoded_dag.h>
#include <voxcom/voxel/encoded_octree.h>
#include <voxcom/voxel/octree.h>
#include <iostream>
#include <stack>

using namespace voxcom;
using namespace voxcom;

TEST_CASE("Construct EncodedDAG<RGB> from EncodedOctree<RGB>", "[EncodedDAG][EncodedOctree][Octree]")
{
    constexpr unsigned resolution = 32;
    constexpr unsigned resolution3 = resolution * resolution * resolution;

    // Value for each pixel.
    //const auto voxelFilled = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));
    const auto voxelValues = GENERATE(take(3, chunk(resolution3 * 3, random<uint32_t>(0, 4))));
    Octree<RGB> octree { resolution };
    for (unsigned i = 0; i < resolution3; i++) {
        // i >= 512 creates some sparsity by not filling the first 8x8x8 region.
        // i % 5    creates an irregular pattern.
        if (i >= 512 && i % 5 == 0) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            const RGB voxelValue { (uint8_t)voxelValues[i * 3 + 0], (uint8_t)voxelValues[i * 3 + 1], (uint8_t)voxelValues[i * 3 + 2] };
            octree.set(voxel, voxelValue);
        }
    }
    const EncodedOctree encodedOctree { octree };
    const EncodedDAG encodedDAG = compressAttributesOctreeToDAG(encodedOctree);

    SECTION("Basic member variables match")
    {
        REQUIRE(encodedDAG.resolution == encodedOctree.resolution);
        REQUIRE(encodedDAG.rootLevel == encodedOctree.rootLevel);
        REQUIRE(encodedDAG.subGridLevel == encodedOctree.subGridLevel);
        REQUIRE(encodedDAG.memoryPerLevel.size() == encodedOctree.nodesPerLevel.size());
    }

    SECTION("DAG traversal visits the same nodes as octree")
    {
        struct StackItem {
            size_t level;
            size_t octreeNode, dagNode, svoOffset;
        };
        std::stack<StackItem> stack;
        stack.push({ .level = encodedOctree.rootLevel, .octreeNode = 0, .dagNode = 0, .svoOffset = 0 });
        while (!stack.empty()) {
            const auto stackItem = stack.top();
            stack.pop();

            if (stackItem.level == encodedOctree.subGridLevel) {
                const uint64_t octreeBitmask = encodedOctree.nodesPerLevel[stackItem.level][stackItem.octreeNode].bitmask4x4x4;
                const uint64_t octreeFirstLeafPtr = encodedOctree.nodesPerLevel[stackItem.level][stackItem.octreeNode + 1].firstAttributePtr;
                const uint64_t dagBitmask = encodedDAG.bitmasks[stackItem.dagNode];
                REQUIRE(octreeBitmask == dagBitmask);

                const int numChildren = std::popcount(dagBitmask);
                for (int childOffset = 0; childOffset < numChildren; ++childOffset) {
                    const auto octreeAttribute = encodedOctree.attributes[octreeFirstLeafPtr + childOffset];
                    const auto dagAttribute = encodedDAG.attributes[stackItem.svoOffset + childOffset];
                    REQUIRE(octreeAttribute == dagAttribute);
                }
            } else {
                const auto& octreeNode = encodedOctree.nodesPerLevel[stackItem.level][stackItem.octreeNode];
                const auto* pDAGMemory = &encodedDAG.memoryPerLevel[stackItem.level][stackItem.dagNode];
                const auto bitmaskDAG = *pDAGMemory++;
                REQUIRE(bitmaskDAG == octreeNode.inner.bitmask);

                // Traverse to children.
                const int numChildren = std::popcount(bitmaskDAG);
                for (int childOffset = 0; childOffset < numChildren; ++childOffset) {
                    stack.push({ .level = stackItem.level - 1,
                        .octreeNode = octreeNode.inner.firstChildPtr + childOffset * (stackItem.level == encodedOctree.subGridLevel + 1 ? 2 : 1),
                        .dagNode = pDAGMemory[2 * childOffset],
                        .svoOffset = stackItem.svoOffset + pDAGMemory[2 * childOffset + 1] });
                }
            }
        }
    }
}

TEST_CASE("EncodedDAG<RGB> optimal compression for checkerboard pattern", "[EncodedDAG][EncodedOctree][Octree]")
{
    constexpr unsigned resolution = 32;
    constexpr unsigned resolution3 = resolution * resolution * resolution;

    // Fill octree with repeating checkerboard pattern.
    Octree<RGB> octree { resolution };
    for (unsigned i = 0; i < resolution3; i++) {
        if (i % 2 == 0) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            octree.set(voxel, RGB { 0, 0, 0 });
        }
    }
    const EncodedOctree encodedOctree { octree };
    const EncodedDAG encodedDAG = compressAttributesOctreeToDAG(encodedOctree);

    SECTION("Basic member variables match")
    {
        REQUIRE(encodedDAG.resolution == encodedOctree.resolution);
        REQUIRE(encodedDAG.rootLevel == encodedOctree.rootLevel);
        REQUIRE(encodedDAG.subGridLevel == encodedOctree.subGridLevel);
        REQUIRE(encodedDAG.memoryPerLevel.size() == encodedOctree.nodesPerLevel.size());
    }

    SECTION("Nodes are reused")
    {
        REQUIRE(encodedDAG.memoryPerLevel[3].size() == 17); // bitmask + 8 children (ptr + SVO offset)
        REQUIRE(encodedDAG.memoryPerLevel[4].size() == 17); // bitmask + 8 children (ptr + SVO offset)
        REQUIRE(encodedDAG.memoryPerLevel[5].size() == 17); // bitmask + 8 children (ptr + SVO offset)
    }
}