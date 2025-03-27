#include "svdag32_encoding.h"
#include "voxcom/utility/error_handling.h"
#include <algorithm>

using namespace voxcom;

SVDAG32 constructSVDAG32(const EditStructure<void, uint32_t>& structure, DAGEncodingStats& stats)
{
    assert_always(structure.structureType == StructureType::DAG);

    SVDAG32 out;
    out.nodesPerLevel.resize(structure.nodesPerLevel.size());
    out.subGrids = structure.subGrids;

    stats.levels.resize(structure.nodesPerLevel.size());
    stats.totalSizeInBytes = 0;
    stats.levels[structure.subGridLevel].numNodes = out.subGrids.size();
    stats.levels[structure.subGridLevel].memoryInBytes = out.subGrids.size() * sizeof(EditSubGrid<void>);
    stats.totalSizeInBytes += stats.levels[structure.subGridLevel].memoryInBytes;

    std::vector<uint32_t> prevLevelMapping(out.subGrids.size());
    std::iota(std::begin(prevLevelMapping), std::end(prevLevelMapping), 0);
    for (uint32_t level = structure.subGridLevel + 1; level < out.nodesPerLevel.size(); ++level) {
        const auto& inNodes = structure.nodesPerLevel[level];
        auto& outNodes = out.nodesPerLevel[level];
        std::vector<uint32_t> curLevelMapping;
        for (auto inNode : inNodes) {
            curLevelMapping.push_back((uint32_t)outNodes.size());
            outNodes.push_back(inNode.getChildMask());
            for (uint32_t child : inNode.children) {
                if (child != EditNode<uint32_t>::EmptyChild)
                    outNodes.push_back(prevLevelMapping[child]);
            }
        }

        auto& levelStats = stats.levels[level];
        levelStats.numNodes = inNodes.size();
        levelStats.memoryInBytes = outNodes.size() * sizeof(uint32_t);
        stats.totalSizeInBytes += levelStats.memoryInBytes;

        prevLevelMapping = std::move(curLevelMapping);
    }

    return out;
}

static void verifySVDAG32_recurse(
    const EditStructure<void, uint32_t>& editStructure, const SVDAG32& svdag,
    uint32_t level, uint32_t editNodeIdx, uint32_t dagNodeIdx)
{
    if (level == editStructure.subGridLevel) {
        const auto& editSubGrid = editStructure.subGrids[editNodeIdx];
        const auto& dagSubGrid = svdag.subGrids[dagNodeIdx];
        assert_always(editSubGrid == dagSubGrid);
    } else {
        const auto& editNode = editStructure.nodesPerLevel[level][editNodeIdx];
        const uint32_t* pDagNode = &svdag.nodesPerLevel[level][dagNodeIdx];
        for (uint32_t childIdx = 0, childOffset = 0; childIdx < 8; ++childIdx) {
            const bool editNodeHasChild = editNode.children[childIdx] != EditNode<uint32_t>::EmptyChild;
            const bool dagNodeHasChild = (*pDagNode >> childIdx) & 0b1;
            assert_always(editNodeHasChild == dagNodeHasChild);

            if (editNodeHasChild) {
                const auto editChild = editNode.children[childIdx];
                const auto dagChild = pDagNode[1 + childOffset++];
                verifySVDAG32_recurse(editStructure, svdag, level - 1, editChild, dagChild);
            }
        }
    }
}

void verifySVDAG32(const EditStructure<void, uint32_t>& editStructure, const SVDAG32& svdag)
{
    verifySVDAG32_recurse(editStructure, svdag, editStructure.rootLevel, 0, 0);
}
