#include "pointer_encoding.h"
#include "transform_dag_encoding.h"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>
#include <voxcom/utility/error_handling.h>
#include <voxcom/voxel/transform_dag.h>
#include <voxcom/voxel/ssvdag.h>

using namespace voxcom;

template <bool ExtendedInvariance>
SSVDAG32 constructSSVDAG32(const EditStructure<void, uint32_t>& structure)
{
    auto editSSVDAG = constructSSVDAG<ExtendedInvariance>(structure);
    // Sort subgrids/nodes by reference count.
    editSSVDAG.subGrids = sortByReferenceCount<EditSubGrid<void>, SymmetryPointer<ExtendedInvariance>>(editSSVDAG.subGrids, editSSVDAG.nodesPerLevel[editSSVDAG.subGridLevel + 1]);
    for (uint32_t level = editSSVDAG.subGridLevel + 1; level < editSSVDAG.rootLevel; ++level) {
        editSSVDAG.nodesPerLevel[level] = sortByReferenceCount<EditNode<SymmetryPointer<ExtendedInvariance>>, SymmetryPointer<ExtendedInvariance>>(editSSVDAG.nodesPerLevel[level], editSSVDAG.nodesPerLevel[level + 1]);
    }

    SSVDAG32 out {};
    out.subGrids = std::move(editSSVDAG.subGrids);
    out.resolution = editSSVDAG.resolution;
    out.rootLevel = editSSVDAG.rootLevel;

    // Traverse inner nodes from the bottom up.
    std::vector<uint32_t> prevLevelMapping(out.subGrids.size());
    std::iota(std::begin(prevLevelMapping), std::end(prevLevelMapping), 0);
    out.nodesPerLevel.resize(editSSVDAG.nodesPerLevel.size());
    for (uint32_t level = editSSVDAG.subGridLevel + 1; level <= editSSVDAG.rootLevel; ++level) {
        auto inLevelNodes = std::move(editSSVDAG.nodesPerLevel[level]);
        auto& outLevelNodes = out.nodesPerLevel[level];
        for (auto& node : inLevelNodes) {
            for (SymmetryPointer<ExtendedInvariance>& child : node.children) {
                if (child != node.EmptyChild)
                    child.ptr = prevLevelMapping[child.ptr];
            }
        }

        prevLevelMapping.clear();
        for (const auto& node : inLevelNodes) {
            prevLevelMapping.push_back((uint32_t)outLevelNodes.size());

            uint32_t childMask = 0;
            for (uint32_t i = 0; i < 8; ++i) {
                if (node.children[i] != SymmetryPointer<ExtendedInvariance>::sentinel())
                    childMask |= 1u << i;
            }
            outLevelNodes.push_back(childMask);

            for (uint32_t i = 0; i < 8; ++i) {
                const auto symmetryPointer = node.children[i];
                if (symmetryPointer == SymmetryPointer<ExtendedInvariance>::sentinel())
                    continue;

                const uint64_t u64Pointer = bvec3ToU64(symmetryPointer.transform) | ((uint64_t)symmetryPointer.ptr << 3);
                assert_always(u64Pointer <= (uint64_t)std::numeric_limits<uint32_t>::max());
                outLevelNodes.push_back((uint32_t)u64Pointer);
            }
        }
    }
    return out;
}

template <bool ExtendedInvariance>
SSVDAG16 constructSSVDAG16(const EditStructure<void, uint32_t>& structure)
{
    auto editSSVDAG = constructSSVDAG<ExtendedInvariance>(structure);
    // Sort subgrids/nodes by reference count.
    editSSVDAG.subGrids = sortByReferenceCount<EditSubGrid<void>, SymmetryPointer<ExtendedInvariance>>(editSSVDAG.subGrids, editSSVDAG.nodesPerLevel[editSSVDAG.subGridLevel + 1]);
    for (uint32_t level = editSSVDAG.subGridLevel + 1; level < editSSVDAG.rootLevel; ++level) {
        editSSVDAG.nodesPerLevel[level] = sortByReferenceCount<EditNode<SymmetryPointer<ExtendedInvariance>>, SymmetryPointer<ExtendedInvariance>>(editSSVDAG.nodesPerLevel[level], editSSVDAG.nodesPerLevel[level + 1]);
    }

    SSVDAG16 out {};
    out.subGrids = std::move(editSSVDAG.subGrids);
    out.resolution = editSSVDAG.resolution;
    out.rootLevel = editSSVDAG.rootLevel;

    // Traverse inner nodes from the bottom up.
    std::vector<uint32_t> prevLevelMapping(out.subGrids.size());
    std::iota(std::begin(prevLevelMapping), std::end(prevLevelMapping), 0);
    out.nodesPerLevel.resize(editSSVDAG.nodesPerLevel.size());
    for (uint32_t level = editSSVDAG.subGridLevel + 1; level <= editSSVDAG.rootLevel; ++level) {
        auto inLevelNodes = std::move(editSSVDAG.nodesPerLevel[level]);
        auto& outLevelNodes = out.nodesPerLevel[level];
        for (auto& node : inLevelNodes) {
            for (SymmetryPointer<ExtendedInvariance>& child : node.children) {
                if (child != node.EmptyChild)
                    child.ptr = prevLevelMapping[child.ptr];
            }
        }

        prevLevelMapping.clear();
        for (const auto& node : inLevelNodes) {
            prevLevelMapping.push_back((uint32_t)outLevelNodes.size());

            const size_t childMaskPtr = push<uint16_t, uint16_t>(outLevelNodes, 0);
            uint16_t childMask = 0;
            for (uint32_t i = 0; i < 8; ++i) {
                const auto symmetryPointer = node.children[i];
                if (symmetryPointer == SymmetryPointer<ExtendedInvariance>::sentinel())
                    continue;

                const uint64_t u64Pointer = bvec3ToU64(symmetryPointer.transform) | ((uint64_t)symmetryPointer.ptr << 3);
                if (u64Pointer <= std::numeric_limits<uint16_t>::max()) {
                    push<uint16_t, uint16_t>(outLevelNodes, (uint16_t)u64Pointer);
                    childMask |= 0b01 << (2 * i);
                } else if (u64Pointer <= std::numeric_limits<uint32_t>::max()) {
                    push<uint16_t, uint32_t>(outLevelNodes, (uint32_t)u64Pointer);
                    childMask |= 0b10 << (2 * i);
                } else {
                    throw std::exception();
                }
            }
            outLevelNodes[childMaskPtr] = childMask;
        }
    }
    return out;
}

template <typename T, typename BaseType>
bool SSVDAGBase<T, BaseType>::get(const glm::ivec3& voxel) const
{
    const T* pThis = (const T*)this;
    uint32_t transform = 0;
    uint32_t nodeIdx = 0;
    for (uint32_t level = this->rootLevel; level > this->subGridLevel; --level) {
        const BaseType* pNode = &this->nodesPerLevel[level][nodeIdx];

        // Bit pattern: y|x
        const int childLevel = level - 1;
        const glm::uvec3 childID = (voxel >> childLevel) & 0b1;
        const uint32_t childIdx = voxcom::morton_encode32(childID);
        if (!pThis->hasChild(pNode, childIdx, transform))
            return false;
        nodeIdx = pThis->traverseToChild(pNode, childIdx, transform);
    }

    // Flip voxel location according to symmetry.
    auto localVoxel = glm::uvec3(voxel & 0b11);
    const auto flippedVoxel = glm::uvec3(3) - localVoxel;
    localVoxel = glm::mix(localVoxel, flippedVoxel, u64ToBvec3(transform));

    const auto subGrid = this->subGrids[nodeIdx];
    const uint32_t voxelIndex = voxcom::morton_encode32(localVoxel);
    const uint64_t voxelBit = ((uint64_t)1) << voxelIndex;
    return (subGrid.bitmask & voxelBit);
}

template class SSVDAGBase<SSVDAG16, uint16_t>;
template class SSVDAGBase<SSVDAG32, uint32_t>;

bool SSVDAG32::hasChild(const BaseType* pNode, uint32_t childIndex, uint32_t transform) const
{
    childIndex ^= transform;
    assert_always(pNode[0] <= 255);
    return (pNode[0] >> childIndex) & 0b1;
}

uint32_t SSVDAG32::traverseToChild(const BaseType* pNode, uint32_t childIndex, uint32_t& transform) const
{
    // Transform child index.
    childIndex ^= transform;

    // Find start of pointer for the given childIndex.
    assert_always(pNode[0] <= 255);
    const uint32_t preMask = ~(0xFFFFFFFF << childIndex);
    const uint32_t childOffset = std::popcount(pNode[0] & preMask);
    const uint32_t pointer = pNode[1 + childOffset];
    transform ^= pointer & 0b111;
    return pointer >> 3;
}

bool SSVDAG16::hasChild(const BaseType* node, uint32_t childIndex, uint32_t transform) const
{
    childIndex ^= transform;
    return (node[0] >> (2 * childIndex)) & 0b11;
}

uint32_t SSVDAG16::traverseToChild(const BaseType* pNode, uint32_t childIndex, uint32_t& transform) const
{
    // Transform child index.
    childIndex ^= transform;

    // Find start of pointer for the given childIndex.
    const uint16_t childMask = pNode[0];
    const BaseType* pPointer = pNode + 1; // First child pointer.
    for (uint32_t i = 0; i < childIndex; ++i) {
        const uint16_t ptrBits = (childMask >> (2 * i)) & 0b11;
        if (ptrBits == 0b01)
            pPointer += 1;
        else if (ptrBits == 0b10 || ptrBits == 0b11)
            pPointer += 2;
    }

    // Check if pointer is 16 bits or 32 bits.
    const uint16_t ptrBits = (childMask >> (2 * childIndex)) & 0b11;
    assert(ptrBits != 0b00);

    // Read the 32 bit pointer.
    uint32_t u32Pointer;
    if (ptrBits == 0b01) {
        u32Pointer = pPointer[0];
    } else {
        // Must be either 0b10 or 0b11;
        std::memcpy(&u32Pointer, pPointer, sizeof(u32Pointer));
    }
    transform ^= u32Pointer & 0b111;
    return u32Pointer >> 3;
}
