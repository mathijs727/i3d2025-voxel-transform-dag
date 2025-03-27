#include "voxcom/voxel/export_ssvdag.h"
#include "voxcom/core/bounds.h"
#include "voxcom/utility/binary_reader.h"
#include "voxcom/utility/binary_writer.h"
#include "voxcom/voxel/structure.h"
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <unordered_map>
#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#define GLM_ENABLE_EXPERIMENTAL 1
#include <fmt/ranges.h>
#include <glm/gtx/component_wise.hpp>
#include <spdlog/spdlog.h>
DISABLE_WARNINGS_POP()

using namespace voxcom;

// Export functionality for files compatible with the Symmetry-Aware SVDAGf code base:
// https://dl-acm-org.tudelft.idm.oclc.org/doi/abs/10.1145/2856400.2856420

namespace voxcom {

[[maybe_unused]] static glm::bvec3 u64ToBvec3(uint64_t v)
{
    return glm::bvec3(v & 0b001, v & 0b010, v & 0b100);
}
template <template <typename, typename> typename Structure, typename Pointer>
void exportUSSVDAG(const Structure<void, Pointer>& structure, const Bounds& bounds, const std::filesystem::path& filePath)
{
    const float rootSide = glm::compMax(bounds.extent());
    const unsigned numLevels = (unsigned)structure.nodesPerLevel.size() - 1;

    // Expand 4x4x4 subgrids into nodes with 2x2x2 children.
    const auto EmptyChild = EditNode<Pointer>::EmptyChild;
    std::vector<EditNode<Pointer>> expandedSubGrids(structure.subGrids.size());
    std::transform(std::begin(structure.subGrids), std::end(structure.subGrids),
        std::begin(expandedSubGrids),
        [&](EditSubGrid<void> subGrid) {
            EditNode<Pointer> out;
            for (uint32_t i = 0; i < 8; ++i) {
                const auto child = uint32_t(subGrid.bitmask >> (i * 8)) & 0xFF;
                if (child)
                    out.children[i] = child;
                else
                    out.children[i] = EmptyChild;
            }
            return out;
        });
    // Count which 2x2x2 masks we actually use.
    std::vector<uint32_t> masksMapping2x2x2((size_t)256, false);
    for (const auto& node : expandedSubGrids) {
        for (const auto child : node.children) {
            if (child != EmptyChild) {
                masksMapping2x2x2[(uint32_t)child] = 1u;
            }
        }
    }
    assert(masksMapping2x2x2[0] == 0);
    // References to 2x2x2 masks.
    std::vector<uint32_t> childMasks2x2x2;
    for (uint32_t i = 0; i < 256; ++i) {
        const auto handle = (uint32_t)childMasks2x2x2.size();
        if (masksMapping2x2x2[i])
            childMasks2x2x2.push_back(i);
        masksMapping2x2x2[i] = handle;
    }
    // Update pointers to the collapsed vector of 2x2x2 masks.
    for (auto& subGridNode : expandedSubGrids) {
        for (auto& child : subGridNode.children) {
            if (child != EditNode<Pointer>::EmptyChild) {
                child = masksMapping2x2x2[(uint32_t)child];
            }
        }
    }

    // Compute where the output nodes/leaves will be placed.
    std::vector<std::vector<uint32_t>> truePointers(structure.nodesPerLevel.size());
    uint32_t counter = 0;
    for (int level = structure.rootLevel; level > 0; --level) {
        auto& levelTruePointers = truePointers[level];
        const auto& inLevelNodes = level == structure.subGridLevel ? expandedSubGrids : structure.nodesPerLevel[level];
        for (const auto& node : inLevelNodes) {
            levelTruePointers.push_back(counter);
            counter += (uint32_t)node.getStaticNodeSize();
        }
    }
    uint32_t firstLeafPtr = counter;
    for (size_t i = 0; i < childMasks2x2x2.size(); ++i) {
        truePointers[1].push_back(counter);
        counter += 1;
    }

    // Serialize the inner nodes.
    uint32_t numNodes = 0;
    std::vector<uint32_t> data((size_t)counter);
    for (int level = structure.rootLevel; level >= 1; --level) {
        const auto& levelNodes = (level == structure.subGridLevel ? expandedSubGrids : structure.nodesPerLevel[level]);
        const auto& levelTruePointers = truePointers[level];
        const auto& childLevelTruePointers = truePointers[level - 1];
        for (size_t nodeIdx = 0; nodeIdx < levelNodes.size(); ++nodeIdx) {
            const auto& node = levelNodes[nodeIdx];

            // SVDAG reads children in reverse order.
            uint32_t* pOutNode = &data[levelTruePointers[nodeIdx]];
            uint32_t mirrorX = 0, mirrorY = 0, mirrorZ = 0;
            for (int childIdx = 7, childOffset = 0; childIdx >= 0; --childIdx) {
                if (node.children[childIdx] != EmptyChild) {
                    const auto childPointer = node.children[childIdx];
                    if constexpr (std::is_same_v<Pointer, uint32_t>) {
                        pOutNode[1 + childOffset++] = childLevelTruePointers[childPointer];
                    } else if constexpr (std::is_same_v<Pointer, SymmetryPointer<true>> || std::is_same_v<Pointer, SymmetryPointer<false>>) {
                        pOutNode[1 + childOffset++] = childLevelTruePointers[childPointer.ptr];
                        // Store transforms
                        if (childPointer.transform.x)
                            mirrorX |= 1 << childIdx;
                        if (childPointer.transform.y)
                            mirrorY |= 1 << childIdx;
                        if (childPointer.transform.z)
                            mirrorZ |= 1 << childIdx;
                    }
                }
            }
            pOutNode[0] = node.getChildMask() | (mirrorZ << 8) | (mirrorY << 16) | (mirrorX << 24);

            ++numNodes;
        }
    }
    // Serialize the leaves.
    for (size_t leafIdx = 0; leafIdx < childMasks2x2x2.size(); ++leafIdx) {
        data[truePointers[1][leafIdx]] = childMasks2x2x2[leafIdx];
    }

    std::ofstream file { filePath, std::ios::binary };
    BinaryWriter writer { file };
    writer.write(bounds.lower);
    writer.write(bounds.upper);
    writer.write(rootSide);
    writer.write(numLevels);
    writer.write(numNodes);
    writer.write(firstLeafPtr);
    writer.write((uint32_t)data.size());
    writer.writeRange(data.data(), data.size());
}
static uint32_t traverseSVDAG(std::span<const uint32_t> svdag, uint32_t level, uint32_t nodeIdx, std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t>& nodeMapping, EditStructure<void, uint32_t>& outDAG);

EditStructure<void, uint32_t> importSVDAG(const std::filesystem::path& filePath, Bounds& outBounds)
{
    float rootSide;
    unsigned numLevels;
    uint32_t numNodes;
    uint32_t firstLeafPtr;
    std::vector<uint32_t> data;

    std::ifstream file { filePath, std::ios::binary };
    BinaryReader reader { file };
    reader.read(outBounds.lower);
    reader.read(outBounds.upper);
    reader.read(rootSide);
    reader.read(numLevels);
    reader.read(numNodes);
    reader.read(firstLeafPtr);
    data.resize(reader.read<uint32_t>());
    reader.readRange(data.data(), data.size());

    EditStructure<void, uint32_t> out(1 << numLevels);
    out.nodesPerLevel[out.rootLevel].clear(); // Remove the (empty) root node.
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t> nodeMapping;
    traverseSVDAG(data, out.rootLevel, 0, nodeMapping, out);

    return out;
}

static uint32_t traverseSVDAG(
    std::span<const uint32_t> svdag, uint32_t level, uint32_t nodeIdx,
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t>& nodeMapping, EditStructure<void, uint32_t>& outDAG)
{
    // (S)SVDAG encodes 2x2x2 bitmasks at level 1, rather than 4x4x4 at level 2.
    // Return the 16-bit (2x2x2) mask directly (rather than a pointer).
    if (level == 1)
        return svdag[nodeIdx];

    // Skip if we visited this node before.
    const std::pair key { level, nodeIdx };
    if (auto iter = nodeMapping.find(key); iter != std::end(nodeMapping))
        return iter->second;

    // Convert node into our own format.
    EditNode<uint32_t> outNode {};
    const auto inNode = svdag.subspan(nodeIdx);
    const uint32_t childMask = inNode[0] & 0xFF;
    for (int childIdx = 7, childOffset = 0; childIdx >= 0; --childIdx) {
        if (!((childMask >> childIdx) & 0b1))
            continue;

        const uint32_t inChildHandle = inNode[1 + childOffset++];
        outNode.children[childIdx] = traverseSVDAG(svdag, level - 1, inChildHandle, nodeMapping, outDAG);
    }

    if (level == outDAG.subGridLevel) {
        // (S)SVDAG encodes 2x2x2 bitmasks at level 1, rather than 4x4x4 at level 2.
        // The node, which now stores 2x2x2 bitmasks as children, is flattened into a 4x4x4 grid.
        EditSubGrid<void> subGrid {};
        assert_always(subGrid.bitmask == 0);
        for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
            if (outNode.children[childIdx] != outNode.EmptyChild)
                subGrid.bitmask |= (uint64_t)outNode.children[childIdx] << (childIdx * 8);
        }
        assert_always(subGrid.bitmask != 0);
        const uint32_t handle = (uint32_t)outDAG.subGrids.size();
        outDAG.subGrids.push_back(subGrid);
        nodeMapping[key] = handle;
        return handle;
    } else {
        auto& outLevelNodes = outDAG.nodesPerLevel[level];
        const uint32_t handle = (uint32_t)outLevelNodes.size();
        outLevelNodes.push_back(outNode);
        nodeMapping[key] = handle;
        return handle;
    }
}

template void exportUSSVDAG(const EditStructure<void, SymmetryPointer<true>>&, const Bounds&, const std::filesystem::path&);
template void exportUSSVDAG(const EditStructure<void, SymmetryPointer<false>>&, const Bounds&, const std::filesystem::path&);
template void exportUSSVDAG(const EditStructure<void, uint32_t>&, const Bounds&, const std::filesystem::path&);
template void exportUSSVDAG(const EditStructureOOC<void, SymmetryPointer<true>>&, const Bounds&, const std::filesystem::path&);
template void exportUSSVDAG(const EditStructureOOC<void, SymmetryPointer<false>>&, const Bounds&, const std::filesystem::path&);
template void exportUSSVDAG(const EditStructureOOC<void, uint32_t>&, const Bounds&, const std::filesystem::path&);

}