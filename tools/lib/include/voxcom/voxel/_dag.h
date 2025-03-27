#pragma once
#include "voxcom/utility/error_handling.h"
#include "voxcom/voxel/octree.h"
#include <algorithm>
#include <array>
#include <span>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <variant>
#include <vector>

namespace voxcom {

struct OctreeNodeHash {
    inline size_t operator()(const OctreeNode& node) const noexcept
    {
        size_t seed = 0;
        for (const auto child : node.children)
            hash_combine(seed, child);
        return seed;
    }
};

template <typename Attribute>
struct SubGridHash {
    inline size_t operator()(const SubGrid<Attribute>& subGrid) const noexcept
    {
        size_t seed = 0;
        hash_combine(seed, subGrid.bitmask);
        if constexpr (!std::is_void_v<Attribute>) {
            for (uint64_t voxelIdx = 0, voxelBit = 1; voxelIdx < 64; ++voxelIdx, voxelBit <<= 1) {
                if (subGrid.bitmask & voxelBit)
                    hash_combine(seed, subGrid.attributes[voxelIdx]);
            }
        }
        return seed;
    }
};

// NOTE(Mathijs): intentionally taking a std::vector instead of a std::span because we mutate the input
// octrees during construction. Mutating the inputs is probably unexpected for the caller since the result
// of the function is a return value (and not a mutable reference argument).
template <typename Attribute>
[[nodiscard]] Octree<Attribute> octreeToDAG(std::vector<Octree<Attribute>> octrees)
{
    using NodeCache = std::unordered_map<OctreeNode, OctreeNode::Child, OctreeNodeHash>;
    using SubGridCache = std::unordered_map<SubGrid<Attribute>, OctreeNode::Child, SubGridHash<Attribute>>;

    const size_t numLevels = octrees[0].nodesPerLevel.size();
    for (const auto& octree : octrees) {
        assert_supported_tree_type<TreeType::Tree, TreeType::DAG>(octree);
        assert(octree.nodesPerLevel.size() == numLevels);
    }

    Octree<Attribute> out { octrees[0].resolution };

    // Collect unique sub grids.
    SubGridCache subGridCache;
    for (auto& octree : octrees) {
        // Make a copy of all unique sub grids.
        for (const auto& inSubGrid : octree.subGrids) {
            if (auto iter = subGridCache.find(inSubGrid); iter == std::end(subGridCache)) {
                const uint32_t newSubGridIdx = (uint32_t)out.subGrids.size();
                out.subGrids.push_back(inSubGrid);
                subGridCache[inSubGrid] = newSubGridIdx;
            }
        }

        // Update pointers in the level above (except if we're at the root level).
        for (auto& inNode : octree.nodesPerLevel[out.subGridLevel + 1]) {
            for (auto& child : inNode.children) {
                if (child != EmptyChild) {
                    const auto& oldSubGrid = octree.subGrids[child];
                    child = subGridCache.find(oldSubGrid)->second; // child is a reference, modifies in place.
                }
            }
        }
    }

    // Collect unique nodes.
    for (size_t level = out.subGridLevel + 1; level <= out.rootLevel; level++) {
        std::vector<OctreeNode> outLevelNodes;

        NodeCache nodeCache;
        for (auto& octree : octrees) {
            // Level 0 is the deepest level. Attribute values are stored at the parent level however, so level 0 will be empty.
            assert(octree.nodesPerLevel[0].size() == 0);

            // Make a copy of all unique nodes.
            for (const auto& inNode : octree.nodesPerLevel[level]) {
                if (auto iter = nodeCache.find(inNode); iter == std::end(nodeCache)) {
                    const uint32_t newNodeIdx = (uint32_t)outLevelNodes.size();
                    outLevelNodes.push_back(inNode);
                    nodeCache[inNode] = newNodeIdx;
                }
            }

            // Update pointers in the level above (except if we're at the root level).
            if (level != out.rootLevel) {
                const auto parentLevel = level + 1;
                for (auto& inNode : octree.nodesPerLevel[parentLevel]) {
                    for (auto& child : inNode.children) {
                        if (child != EmptyChild) {
                            const auto& oldNode = octree.nodesPerLevel[level][child];
                            child = nodeCache.find(oldNode)->second; // child is a reference, modifies in place.
                        }
                    }
                }
            }
        }

        out.nodesPerLevel[level] = std::move(outLevelNodes);
    }
    // The root nodes should always be copied, even if they contain duplicates.
    out.nodesPerLevel[out.rootLevel].clear();
    for (const auto& octree : octrees)
        out.nodesPerLevel[out.rootLevel].push_back(octree.nodesPerLevel[out.rootLevel][0]);

    out.treeType = TreeType::MultiDAG;
    return out;
}

// NOTE(Mathijs): pass octree directly instead of as a reference because we will mutate it during construction.
// Mutating the inputs is probably unexpected for the caller since the result of the function is a return value
// (and not a mutable reference argument).
template <typename Attribute>
[[nodiscard]] Octree<Attribute> octreeToDAG(const Octree<Attribute>& octree)
{
    const std::vector octrees { octree };
    auto out = octreeToDAG(octrees);
    out.treeType = TreeType::DAG; 
    return out;
}

}

