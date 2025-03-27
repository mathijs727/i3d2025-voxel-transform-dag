#include "pointer_encoding.h"
#include "transform_dag_encoding.h"
#include <algorithm>
#include <bit>
#include <exception>
#include <numeric>
#include <ranges>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <voxcom/utility/size_of.h>

using namespace voxcom;

template <typename T, typename BaseType>
bool NoTransformBaseDAG<T, BaseType>::get(const glm::ivec3& voxel) const
{
    const T* pThis = (const T*)this;
    uint32_t nodeIdx = 0;
    for (uint32_t level = this->rootLevel; level > this->subGridLevel; --level) {
        const BaseType* pNode = &this->nodesPerLevel[level][nodeIdx];

        // Bit pattern: y|x
        const int childLevel = level - 1;
        const glm::uvec3 childID = (voxel >> childLevel) & 0b1;
        const uint32_t childIdx = voxcom::morton_encode32(childID);
        if (!pThis->hasChild(pNode, childIdx))
            return false;
        nodeIdx = pThis->getChildHandle(pNode, childIdx);
    }

    const auto subGrid = this->subGrids[nodeIdx];
    const uint32_t voxelIndex = voxcom::morton_encode32(glm::uvec3(voxel & 0b11));
    const uint64_t voxelBit = ((uint64_t)1) << voxelIndex;
    return (subGrid.bitmask & voxelBit);
}

NoTransformDAG32 constructNoTransform32(const EditStructure<void, uint32_t>& octree)
{
    auto dag = octree;
    dag.toDAG();

    NoTransformDAG32 out {};
    out.subGrids = std::move(dag.subGrids);
    out.resolution = dag.resolution;
    out.rootLevel = dag.rootLevel;

    std::vector<uint32_t> prevLevelMapping(out.subGrids.size());
    std::iota(std::begin(prevLevelMapping), std::end(prevLevelMapping), 0);

    // Traverse inner nodes from the bottom up.
    out.nodesPerLevel.resize(dag.nodesPerLevel.size());
    for (uint32_t level = dag.subGridLevel + 1; level <= dag.rootLevel; ++level) {
        auto inLevelNodes = std::move(dag.nodesPerLevel[level]);
        auto& outLevelNodes = out.nodesPerLevel[level];
        for (auto& node : inLevelNodes) {
            for (uint32_t& child : node.children) {
                if (child != node.EmptyChild)
                    child = prevLevelMapping[child];
            }
        }

        prevLevelMapping.clear();
        for (const auto& node : inLevelNodes) {
            prevLevelMapping.push_back((uint32_t)outLevelNodes.size());
            outLevelNodes.push_back(node.getChildMask());
            for (uint32_t child : node.children) {
                if (child != node.EmptyChild)
                    outLevelNodes.push_back(child);
            }
        }
    }
    return out;
}

bool NoTransformDAG32::hasChild(const BaseType* node, uint32_t childIndex) const
{
    return (node[0] >> childIndex) & 0b1;
}

uint32_t NoTransformDAG32::getChildHandle(const BaseType* node, uint32_t childIndex) const
{
    const uint32_t preMask = ~(0xFFFFFFFF << childIndex);
    const uint32_t childOffset = std::popcount(node[0] & preMask);
    return node[1 + childOffset];
}

NoTransformHybridDAG32 constructNoTransformHybrid32(const EditStructure<void, uint32_t>& octree)
{
    auto dag = octree;
    dag.toDAG();

    NoTransformHybridDAG32 out {};
    out.resolution = dag.resolution;
    out.rootLevel = dag.rootLevel;
    out.nodesPerLevel.resize(dag.nodesPerLevel.size());

#if 1

    std::vector<uint32_t> prevLevelMapping(dag.subGrids.size());
    std::vector<size_t> prevLevelParents(dag.subGrids.size());
    {
        const auto& parentNodes = dag.nodesPerLevel[dag.subGridLevel + 1];
        for (const auto& [parentIdx, parentNode] : std::views::enumerate(parentNodes)) {
            for (const auto [childIdx, childHandle] : std::views::enumerate(parentNode.children)) {
                if (childHandle != EditNode<uint32_t>::EmptyChild)
                    prevLevelParents[childHandle] = 8 * parentIdx + childIdx;
            }
        }

        std::vector<uint32_t> sortedIndices(dag.subGrids.size());
        std::iota(std::begin(sortedIndices), std::end(sortedIndices), 0);
        std::sort(std::begin(sortedIndices), std::end(sortedIndices), [&](uint32_t lhs, uint32_t rhs) { return prevLevelParents[lhs] < prevLevelParents[rhs]; });

        for (uint32_t itemIdx : sortedIndices) {
            prevLevelMapping[itemIdx] = (uint32_t)dag.subGrids.size();
            out.subGrids.push_back(dag.subGrids[itemIdx]);
        }
    }

    for (uint32_t level = dag.subGridLevel + 1; level <= dag.rootLevel; ++level) {
        const auto& inNodes = dag.nodesPerLevel[level];

        std::vector<size_t> curLevelParents(inNodes.size(), (uint32_t)-1);
        if (level != dag.rootLevel) {
            const auto& parentNodes = dag.nodesPerLevel[level + 1];
            for (const auto& [parentIdx, parentNode] : std::views::enumerate(parentNodes)) {
                for (const auto [childIdx, childHandle] : std::views::enumerate(parentNode.children)) {
                    if (childHandle != EditNode<uint32_t>::EmptyChild)
                        curLevelParents[childHandle] = 8 * parentIdx + childIdx;
                }
            }
        }

        std::vector<uint32_t> sortedIndices(inNodes.size());
        std::iota(std::begin(sortedIndices), std::end(sortedIndices), 0);
        std::sort(std::begin(sortedIndices), std::end(sortedIndices), [&](uint32_t lhs, uint32_t rhs) { return curLevelParents[lhs] < curLevelParents[rhs]; });

        auto& outNodes = out.nodesPerLevel[level];
        std::vector<uint32_t> curLevelMapping(inNodes.size());
        for (uint32_t itemIdx : sortedIndices) {
            curLevelMapping[itemIdx] = (uint32_t)outNodes.size();

            const auto& inNode = inNodes[itemIdx];
            uint32_t indirectChildMask = 0, directChildMask = 0;
            uint32_t firstDirectChild = std::numeric_limits<uint32_t>::max();
            outNodes.push_back(0u); // bitmask
            for (auto [childIdx, child] : std::views::enumerate(inNode.children)) {
                if (child == inNode.EmptyChild)
                    continue;

                if ((prevLevelParents[child] / 8) == itemIdx) {
                    directChildMask |= 1u << childIdx;
                    firstDirectChild = std::min(firstDirectChild, prevLevelMapping[child]);
                } else {
                    indirectChildMask |= 1u << childIdx;
                    outNodes.push_back(prevLevelMapping[child]);
                }
            }
            if (directChildMask)
                outNodes.push_back(firstDirectChild);

            auto& outChildMask = outNodes[curLevelMapping[itemIdx]];
            outChildMask = indirectChildMask | (directChildMask << 8);
        }

        prevLevelMapping = std::move(curLevelMapping);
        prevLevelParents = std::move(curLevelParents);
    }

    assert_always(prevLevelMapping.size() == 1);

#else
    const auto writeChildren = [&]<typename T>(const std::vector<T>& inItems, std::vector<T>& outItems, const EditNode<uint32_t>& parent) {
        const uint32_t outputIndex = (uint32_t)outItems.size();
        for (uint32_t child : parent.children) {
            if (child == EditNode<uint32_t>::EmptyChild)
                continue;
            outItems.push_back(inItems[child]);
        }
        // assert_always(outputIndex < (1u << 24));
        return (outputIndex << 8) | parent.getChildMask();
    };

    std::vector<uint32_t> prevLevelMapping(dag.nodesPerLevel[dag.subGridLevel + 1].size());
    {
        const auto& inLevelNodes = dag.nodesPerLevel[dag.subGridLevel + 1];
        for (size_t i = 0; i < inLevelNodes.size(); ++i) {
            const EditNode<uint32_t>& inNode = inLevelNodes[i];
            prevLevelMapping[i] = writeChildren(dag.subGrids, out.subGrids, inNode);
        }
    }
    for (uint32_t level = dag.subGridLevel + 2; level <= dag.rootLevel; ++level) {
        const auto childLevel = level - 1;
        auto inLevelNodes = std::move(dag.nodesPerLevel[level]);
        auto& outChildLevelNodes = out.nodesPerLevel[childLevel];

        std::vector<uint32_t> curLeveLmapping(inLevelNodes.size());
        for (size_t i = 0; i < inLevelNodes.size(); ++i) {
            const EditNode<uint32_t>& inNode = inLevelNodes[i];
            prevLevelMapping[i] = writeChildren(prevLevelMapping, outChildLevelNodes, inNode);
        }

        prevLevelMapping = std::move(curLeveLmapping);
    }
    assert_always(prevLevelMapping.size() == 1);
    out.rootHandle = prevLevelMapping[0];
#endif
    return out;
}

bool NoTransformHybridDAG32::hasChild(const BaseType* node, uint32_t childIndex) const
{
    const uint32_t indirectChildMask = node[0] & 0xFF;
    const uint32_t directChildMask = (node[0] >> 8) & 0xFF;
    const uint32_t childMask = indirectChildMask | directChildMask;
    return (childMask >> childIndex) & 0b1;
}

uint32_t NoTransformHybridDAG32::getChildHandle(const BaseType* node, uint32_t childIndex) const
{
    const uint32_t indirectChildMask = node[0] & 0xFF;
    const uint32_t directChildMask = (node[0] >> 8) & 0xFF;
    
    const uint32_t prefixMask = ~(0xFFFFFFFF << childIndex);
    if ((indirectChildMask >> childIndex) & 0b1) {
        // Regular SVDAG
        const uint32_t childOffset = std::popcount(indirectChildMask & prefixMask);
        return node[1 + childOffset];
    } else {
        // SVO-style encoding
        assert_always((directChildMask >> childIndex) & 0b1);
        const uint32_t firstNode = node[1 + std::popcount(indirectChildMask)];
        // TODO(Mathijs): dereference pointer, iterate over children, stop at the std::popcount(directChildMask & prefixMask)'th child.
        return firstNode; // + ...
    }
}

NoTransformDAG16 constructNoTransform16(const EditStructure<void, uint32_t>& octree)
{
    // Convert SVO to SVDAG.
    auto dag = octree;
    dag.toDAG();
    // Sort subgrids/nodes by reference count.
    dag.subGrids = sortByReferenceCount<EditSubGrid<void>, uint32_t>(dag.subGrids, dag.nodesPerLevel[dag.subGridLevel + 1]);
    for (uint32_t level = dag.subGridLevel + 1; level < dag.rootLevel; ++level) {
        dag.nodesPerLevel[level] = sortByReferenceCount<EditNode<uint32_t>, uint32_t>(dag.nodesPerLevel[level], dag.nodesPerLevel[level + 1]);
    }

    // Convert to final U16 DAG.
    NoTransformDAG16 out {};
    out.subGrids = std::move(dag.subGrids);
    out.resolution = dag.resolution;
    out.rootLevel = dag.rootLevel;

    spdlog::info("[{}] {} KiB ({} leaves)", out.subGridLevel, (out.subGrids.size() * sizeof(EditSubGrid<void>)) >> 10, out.subGrids.size());
    // Traverse inner nodes from the bottom up.
    std::vector<uint32_t> prevLevelMapping(out.subGrids.size());
    std::iota(std::begin(prevLevelMapping), std::end(prevLevelMapping), 0);
    out.nodesPerLevel.resize(dag.nodesPerLevel.size());
    for (uint32_t level = dag.subGridLevel + 1; level <= dag.rootLevel; ++level) {
        auto inLevelNodes = std::move(dag.nodesPerLevel[level]);
        auto& outLevelNodes = out.nodesPerLevel[level];
        for (auto& node : inLevelNodes) {
            for (uint32_t& child : node.children) {
                if (child != node.EmptyChild)
                    child = prevLevelMapping[child];
            }
        }

        prevLevelMapping.clear();
        for (const auto& node : inLevelNodes) {
            prevLevelMapping.push_back((uint32_t)outLevelNodes.size());

            const size_t childMaskPtr = push<uint16_t, uint16_t>(outLevelNodes, 0);
            uint16_t childMask = 0;
            for (uint32_t i = 0; i < 8; ++i) {
                const auto child = node.children[i];
                if (child == node.EmptyChild)
                    continue;

                if (child <= std::numeric_limits<uint16_t>::max()) {
                    push<uint16_t, uint16_t>(outLevelNodes, (uint16_t)child);
                    childMask |= 0b01 << (2 * i);
                } else {
                    push<uint16_t, uint32_t>(outLevelNodes, child);
                    childMask |= 0b10 << (2 * i);
                }
            }
            outLevelNodes[childMaskPtr] = childMask;
        }
        spdlog::info("[{}] {} KiB ({} nodes)", level, (outLevelNodes.size() * sizeof(uint16_t)) >> 10, inLevelNodes.size());
    }
    return out;
}

bool NoTransformDAG16::hasChild(const BaseType* node, uint32_t childIndex) const
{
    return (node[0] >> (2 * childIndex)) & 0b11;
}

uint32_t NoTransformDAG16::getChildHandle(const BaseType* pNode, uint32_t childIndex) const
{
    const uint16_t childMask = pNode[0];
    const BaseType* ptr = pNode + 1; // First child pointer.
    for (uint32_t i = 0; i < childIndex; ++i) {
        const uint16_t ptrBits = (childMask >> (2 * i)) & 0b11;
        if (ptrBits == 0b01)
            ptr += 1;
        else if (ptrBits == 0b10 || ptrBits == 0b11)
            ptr += 2;
    }

    const uint16_t ptrBits = (childMask >> (2 * childIndex)) & 0b11;
    assert(ptrBits != 0b00);
    if (ptrBits == 0b01) {
        return ptr[0];
    } else {
        // Must be either 0b10 or 0b11;
        uint32_t out { 0 };
        std::memcpy(&out, ptr, sizeof(out));
        return out;
    }
}

template class NoTransformBaseDAG<NoTransformDAG16, uint16_t>;
template class NoTransformBaseDAG<NoTransformDAG32, uint32_t>;
template class NoTransformBaseDAG<NoTransformHybridDAG32, uint32_t>;
