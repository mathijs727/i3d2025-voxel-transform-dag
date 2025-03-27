#include "voxcom/voxel/encoded_octree.h"
#include "voxcom/utility/error_handling.h"
#include <bit>
#include <cassert>
#include <span>
#include <stack>
#include <unordered_map>

namespace voxcom {

template <typename Attribute>
EncodedOctree<Attribute>::EncodedOctree(const Octree<Attribute>& octree)
    : rootLevel((uint32_t)octree.nodesPerLevel.size() - 1)
    , resolution(octree.resolution)
{
    assert_supported_tree_type<TreeType::Tree>(octree);

    nodesPerLevel.resize(octree.nodesPerLevel.size());
    const auto root = addSubtree(octree);
    nodesPerLevel[rootLevel].push_back(root);
}

template <typename Attribute>
typename EncodedOctree<Attribute>::Node EncodedOctree<Attribute>::addSubtree(const Octree<Attribute>& octree)
{
    assert_supported_tree_type<TreeType::Tree>(octree);

    // Encode leaf nodes.
    std::unordered_map<uint32_t, std::pair<Node, Node>> subGridMapping;
    {
        for (uint32_t subGridIdx = 0; subGridIdx < octree.subGrids.size(); ++subGridIdx) {
            const auto& subGrid = octree.subGrids[subGridIdx];

            const uint64_t firstAttributePtr = (uint64_t)attributes.size();
            for (uint64_t voxelIdx = 0, voxelBit = 1; voxelIdx < 64; ++voxelIdx, voxelBit <<= 1) {
                if (subGrid.bitmask & voxelBit)
                    attributes.push_back(subGrid.attributes[voxelIdx]);
            }

            // Store the 4x4x4 bitmask and pointer to the attributes.
            subGridMapping[subGridIdx] = std::pair {
                Node { .bitmask4x4x4 = subGrid.bitmask },
                Node { .firstAttributePtr = firstAttributePtr }
            };
        }
    }

    // Encode inner nodes.
    std::unordered_map<uint32_t, Node> prevLevelChildNodeMapping;
    for (uint32_t parentLevel = octree.subGridLevel + 1; parentLevel <= rootLevel; parentLevel++) {
        std::unordered_map<uint32_t, Node> curLevelChildNodeMapping;
        const auto& inNodes = octree.nodesPerLevel[parentLevel];
        auto& outChildNodes = nodesPerLevel[parentLevel - 1];

        for (uint32_t inNodeIdx = 0; inNodeIdx < inNodes.size(); inNodeIdx++) {
            // Store the children consecutively in memory.
            const auto& inNode = inNodes[inNodeIdx];
            const uint64_t firstChildPtr = (uint64_t)outChildNodes.size();
            uint64_t bitmask = 0;
            for (uint32_t childIdx = 0; childIdx < 8; childIdx++) {
                const auto& child = inNode.children[childIdx];
                if (child != EmptyChild) {
                    if (parentLevel == octree.subGridLevel + 1) {
                        const auto [subGridBitmask, firstAttributePtr] = subGridMapping.find(child)->second;
                        outChildNodes.push_back(subGridBitmask);
                        outChildNodes.push_back(firstAttributePtr);
                    } else {
                        outChildNodes.push_back(prevLevelChildNodeMapping.find(child)->second);
                    }
                    bitmask |= (uint64_t)1 << childIdx;
                }
            }

            assert_always(bitmask <= 0xFF);
            assert_always(firstChildPtr < (((uint64_t)1 << 56) - 1));
            Node outNode {};
            outNode.inner.bitmask = bitmask;
            outNode.inner.firstChildPtr = firstChildPtr;
            curLevelChildNodeMapping[inNodeIdx] = outNode;
        }

        prevLevelChildNodeMapping = std::move(curLevelChildNodeMapping);
        subGridMapping.clear();
    }

    assert(prevLevelChildNodeMapping.size() == 1); // Root level has 1 node.
    return std::begin(prevLevelChildNodeMapping)->second;
}

template <typename Attribute>
EncodedOctree<Attribute>::operator Octree<Attribute>() const
{
    Octree<Attribute> out;
    out.resolution = resolution;
    out.rootLevel = rootLevel;
    out.treeType = TreeType::Tree;
    out.nodesPerLevel.resize(rootLevel + 1);

    std::unordered_map<uint64_t, OctreeNode::Child> prevLevelChildNodeMapping;
    // Decode the leaf nodes into 4x4x4 subgrids with attributes.
    {
        const auto& inNodes = nodesPerLevel[subGridLevel];
        for (size_t nodeIdx = 0; nodeIdx < inNodes.size(); nodeIdx += 2) {
            const uint64_t bitmask = inNodes[nodeIdx].bitmask4x4x4;
            const uint64_t firstAttributePtr = inNodes[nodeIdx + 1].firstAttributePtr;

            SubGrid<Attribute> subGrid { .bitmask = bitmask };
            if constexpr (!std::is_void_v<Attribute>) {
                uint64_t voxelBit = 1u, nonEmptyVoxel = 0;
                for (uint32_t voxelIdx = 0; voxelIdx < 64; ++voxelIdx) {
                    if (subGrid.bitmask & voxelBit)
                        subGrid.attributes[voxelIdx] = attributes[firstAttributePtr + (nonEmptyVoxel++)];
                    voxelBit <<= 1;
                }
            }
            out.subGrids.push_back(subGrid);
        }
    }

    // Decode the inner nodes.
    for (size_t level = subGridLevel + 1; level < nodesPerLevel.size(); level++) {
        const uint32_t childLevelSize = (level - 1 == subGridLevel ? 2 : 1);
        for (const Node inNode : nodesPerLevel[level]) {
            OctreeNode outNode;
            for (uint32_t childIdx = 0, childBit = 1, cursor = (uint32_t)inNode.inner.firstChildPtr / childLevelSize; childIdx < 8; ++childIdx, childBit <<= 1) {
                if (inNode.inner.bitmask & childBit)
                    outNode.children[childIdx] = (cursor++);
                else
                    outNode.children[childIdx] = EmptyChild;
            }
            out.nodesPerLevel[level].push_back(outNode);
        }
    }
    return out;
}

template <typename Attribute>
Octree<void> EncodedOctree<Attribute>::toBinaryOctree() const
{
    auto attributeOctree = Octree<Attribute>(*this);
    if constexpr (std::is_void_v<Attribute>)
        return attributeOctree;

    Octree<void> out;
    out.nodesPerLevel = std::move(attributeOctree.nodesPerLevel);
    out.subGrids.resize(attributeOctree.subGrids.size());
    std::transform(
        std::begin(attributeOctree.subGrids), std::end(attributeOctree.subGrids),
        std::begin(out.subGrids),
        [](const SubGrid<Attribute>& inSubGrid) { return SubGrid<void> { .bitmask = inSubGrid.bitmask }; });
    out.resolution = resolution;
    out.rootLevel = rootLevel;
    out.treeType = TreeType::Tree;
    return out;
}

template <typename Attribute>
size_t EncodedOctree<Attribute>::computeMemoryUsage() const
{
    size_t out = 0;
    for (const auto& nodes : nodesPerLevel) {
        out += nodes.size() * sizeof(typename std::remove_reference_t<decltype(nodes)>::value_type);
    }
    out += attributes.size() * sizeof(typename decltype(attributes)::value_type);
    return out;
}

template <typename Attribute>
void EncodedOctree<Attribute>::reorderAttributesDepthFirst()
{
    if constexpr (std::is_same_v<Attribute, bool>)
        return;

    std::vector<Attribute> sortedAttributes;
    sortedAttributes.reserve(attributes.size());

    // Parallel voxelization may have reordered subtrees arbitrarily. The original paper uses
    // "SVO" (morton code) ordered attributes so we need to traverse the tree depth-first and
    // store the attributes accordingly.
    struct StackItem {
        size_t level, cursor;
    };
    std::stack<StackItem> traversalStack;
    traversalStack.push({ rootLevel, 0 });
    while (!traversalStack.empty()) {
        auto [level, cursor] = traversalStack.top();
        traversalStack.pop();

        auto& levelNodes = nodesPerLevel[level];
        if (level == subGridLevel) {
            // Root of the 4x4x4 regions subtree.
            const auto bitmask = levelNodes[cursor++].bitmask4x4x4;
            // Make a local copy of the pointer to the first leaf and then update it to the new address.
            uint64_t& firstAttributePtrRef = levelNodes[cursor++].firstAttributePtr;
            assert(firstAttributePtrRef < attributes.size());
            uint64_t firstAttributePtr = firstAttributePtrRef;
            firstAttributePtrRef = (uint64_t)sortedAttributes.size();
            for (int leafOffset = 0; leafOffset < std::popcount(bitmask); leafOffset++) {
                sortedAttributes.push_back(attributes[firstAttributePtr + leafOffset]);
            }
        } else {
            // Depth first traversal.
            const auto node = levelNodes[cursor++];
            size_t activeChild = 0;
            for (size_t childIdx = 0; childIdx < 8; childIdx++) {
                if (node.inner.bitmask & (1llu << childIdx)) {
                    traversalStack.push({ level - 1, node.inner.firstChildPtr + activeChild });
                    if (level - 1 == subGridLevel)
                        activeChild += 2;
                    else
                        activeChild++;
                }
            }
        }
    }

    attributes = std::move(sortedAttributes);
    attributesOrder = AttributesOrder::DepthFirst;
}

// template struct EncodedOctree<void>;
template struct EncodedOctree<voxcom::RGB>;
}