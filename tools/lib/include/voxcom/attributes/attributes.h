#pragma once
#include "voxcom/attributes/color.h"
#include "voxcom/utility/error_handling.h"
#include "voxcom/utility/hash.h"
#include "voxcom/voxel/encoded_octree.h"
#include "voxcom/voxel/morton.h"
#include "voxcom/voxel/octree.h"
#include <algorithm>
#include <cassert>
#include <functional> // std::hash
#include <span>
#include <stack>
#include <tuple>
#include <vector>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

namespace voxcom {

template <typename T>
size_t countAttributesLowestLevel(const Octree<T>&);
template <typename T>
std::vector<T> collectAttributesLowestLevel(const Octree<T>&);
template <typename T>
std::vector<T> collectAttributesDepthFirst(const Octree<T>&);
template <typename T>
std::vector<T> collectAttributesDepthFirst(const EncodedOctree<T>&);
template <typename T>
std::vector<std::pair<glm::ivec3, T>> collectAttributesWithPositions(const EncodedOctree<T>& octree);
template <typename T, typename F>
void replaceAttributesDepthFirst(EncodedOctree<T>&, F&&);

template <typename T>
Octree<void> stripAttributes(const Octree<T>&);
template <typename T, typename S, typename F>
Octree<T> transformAttributes(const Octree<S>& octree, F&& func);
template <typename S, typename F>
Octree<void> removeAttributes(const Octree<S>& octree, F&& func);
template <typename T>
std::vector<T> filterUniqueAttributes(std::span<const T>);

template <typename T>
size_t countAttributesLowestLevel(const Octree<T>& octree)
{
    for (const auto& nodes : octree.nodesPerLevel) {
        // Iterate bottom-up until we find a non-empty level.
        if (nodes.empty())
            continue;

        size_t attributeCount = 0;
        for (const auto& node : nodes) {
            for (const auto& child : node.children) {
                if (std::holds_alternative<OctreeNode<T>::leaf_type>(child))
                    attributeCount++;
            }
        }
        return attributeCount;
    }
    // Empty tree.
    return 0;
}

template <typename T, typename S, typename F>
Octree<T> transformAttributes(const Octree<S>& octree, F&& func)
{
    Octree<T> out { octree.resolution };
    out.treeType = octree.treeType;
    out.nodesPerLevel = octree.nodesPerLevel;
    out.subGrids.resize(octree.subGrids.size());
    std::transform(
        std::begin(octree.subGrids), std::end(octree.subGrids),
        std::begin(out.subGrids),
        [&](const SubGrid<S>& inSubGrid) -> SubGrid<T> {
            SubGrid<T> out { .bitmask = inSubGrid.bitmask };
            uint64_t voxelBit = 1u;
            for (uint32_t voxelIdx = 0; voxelIdx < 64; ++voxelIdx) {
                if (inSubGrid.bitmask & voxelBit)
                    out.attributes[voxelIdx] = func(inSubGrid.attributes[voxelIdx]);
                voxelBit <<= 1;
            }
            return out;
        });
    return out;
}

template <typename S, typename F>
Octree<void> transformAttributesToBinary(const Octree<S>& octree, F&& func)
{
    assert_supported_tree_type<TreeType::Tree>(octree);

    Octree<void> out { octree.resolution };
    out.treeType = octree.treeType;
    out.nodesPerLevel = octree.nodesPerLevel;
    out.subGrids.resize(octree.subGrids.size());
    std::transform(
        std::begin(octree.subGrids), std::end(octree.subGrids),
        std::begin(out.subGrids),
        [&](const SubGrid<S>& inSubGrid) -> SubGrid<void> {
            uint64_t outBitmask = 0, voxelBit = 1;
            for (uint32_t voxelIdx = 0; voxelIdx < 64; ++voxelIdx, voxelBit <<= 1) {
                if ((inSubGrid.bitmask & voxelBit) && func(inSubGrid.attributes[voxelIdx]))
                    outBitmask |= voxelBit;
            }
            return SubGrid<void> { .bitmask = outBitmask };
        });
    return out;
}

// Replace the leaf attributes in the octree according to the given function.
// The function takes the depth first node index (sorting nodes along morton order) and returns the attribute T for that leaf.
template <typename T, typename F>
void replaceAttributesDepthFirst(EncodedOctree<T>& octree, F&& func)
{
    size_t depthFirstNodeIdx = 0;

    struct StackItem {
        size_t level;
        size_t idx;
    };
    std::stack<StackItem> stack;
    stack.push({ octree.nodesPerLevel.size() - 1, 0 });
    while (!stack.empty()) {
        const auto [level, idx] = stack.top();
        stack.pop();

        if (level != octree.leafLevel) {
            const auto& node = octree.nodesPerLevel[level][idx];
            const uint64_t bitmask = node.inner.bitmask;
            const int childNodeSize = level == octree.leafLevel + 1 ? 2 : 1;
            for (int childOffset = 0; childOffset < childNodeSize * std::popcount(bitmask); childOffset += childNodeSize) {
                stack.push({ level - 1, node.inner.firstChildPtr + childOffset });
            }
        } else {
            const auto& bitmask4x4x4 = octree.nodesPerLevel[level][idx].bitmask4x4x4;
            const auto& firstLeafPtr = octree.nodesPerLevel[level][idx + 1].firstLeafPtr;
            for (int childOffset = 0; childOffset < std::popcount(bitmask4x4x4); childOffset++) {
                octree.leaves[firstLeafPtr + childOffset] = func(depthFirstNodeIdx++);
            }
        }
    }
}

}
