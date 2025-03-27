#include "svo32_encoding.h"
#include "voxcom/utility/error_handling.h"
#include "voxcom/utility/size_of.h"
#include <algorithm>
#include <bit>
#include <stack>

using namespace voxcom;

SVO32 constructSVO32(const EditStructure<void, uint32_t>& structure, DAGEncodingStats& stats)
{
    assert_always(structure.structureType == StructureType::DAG);

    SVO32 out;
    out.nodesPerLevel.resize(structure.nodesPerLevel.size());

    struct StackItem {
        uint32_t level, inNodeIdx, outNodeIdx;
    };
    std::stack<StackItem> stack;
    stack.emplace(structure.rootLevel, 0);
    out.nodesPerLevel[structure.rootLevel].emplace_back();
    while (!stack.empty()) {
        const auto [level, inNodeIdx, outNodeIdx] = stack.top();
        stack.pop();

        if (level == structure.subGridLevel) {
            out.subGrids[outNodeIdx] = structure.subGrids[inNodeIdx];
        } else {
            const auto& inNode = structure.nodesPerLevel[level][inNodeIdx];
            const auto childMask = inNode.getChildMask();
            const auto numChildren = std::popcount(childMask);
            const auto nextLevel = level - 1;

            uint32_t firstChildIdx;
            if (nextLevel == structure.subGridLevel) {
                firstChildIdx = (uint32_t)out.subGrids.size();
                out.subGrids.resize(firstChildIdx + numChildren);
            } else {
                auto& outNextLevelNodes = out.nodesPerLevel[nextLevel];
                firstChildIdx = (uint32_t)outNextLevelNodes.size();
                outNextLevelNodes.resize(firstChildIdx + numChildren);
            }

            auto& outLevelNodes = out.nodesPerLevel[level];
            outLevelNodes[outNodeIdx] = typename SVO32::InnerNode {
                .childMask = childMask,
                .firstChildPtr = firstChildIdx
            };

            for (uint32_t childIdx = 0, childOffset = 0; childIdx < 8; ++childIdx) {
                const auto inChild = inNode.children[childIdx];
                if (inChild != EditNode<uint32_t>::EmptyChild) {
                    stack.push({ .level = nextLevel, .inNodeIdx = inChild, .outNodeIdx = firstChildIdx + childOffset++ });
                }
            }
        }
    }

    stats.levels.resize(structure.nodesPerLevel.size());
    stats.totalSizeInBytes = 0;
    stats.levels[structure.subGridLevel].numNodes = out.subGrids.size();
    stats.levels[structure.subGridLevel].memoryInBytes = sizeOfVector(out.subGrids);
    for (uint32_t level = structure.subGridLevel+1; level <= structure.rootLevel; ++level) {
        stats.levels[level].numNodes = out.nodesPerLevel[level].size();
        stats.levels[level].memoryInBytes = sizeOfVector(out.nodesPerLevel[level]); 
    }
    for (const auto& levelStats : stats.levels)
        stats.totalSizeInBytes += levelStats.memoryInBytes;

    return out;
}

static void verifySVO32_recurse(
    const EditStructure<void, uint32_t>& editStructure, const SVO32& svo,
    uint32_t level, uint32_t editNodeIdx, uint32_t svoNodeIdx)
{
    if (level == editStructure.subGridLevel) {
        const auto& editSubGrid = editStructure.subGrids[editNodeIdx];
        const auto& dagSubGrid = svo.subGrids[svoNodeIdx];
        assert_always(editSubGrid == dagSubGrid);
    } else {
        const auto& editNode = editStructure.nodesPerLevel[level][editNodeIdx];
        const typename SVO32::InnerNode& svoNode = svo.nodesPerLevel[level][svoNodeIdx];
        for (uint32_t childIdx = 0, childOffset = 0; childIdx < 8; ++childIdx) {
            const bool editNodeHasChild = editNode.children[childIdx] != EditNode<uint32_t>::EmptyChild;
            const bool dagNodeHasChild = (svoNode.childMask >> childIdx) & 0b1;
            assert_always(editNodeHasChild == dagNodeHasChild);

            if (editNodeHasChild) {
                const auto editChild = editNode.children[childIdx];
                const auto svoChild = svoNode.firstChildPtr + childOffset++;
                verifySVO32_recurse(editStructure, svo, level - 1, editChild, svoChild);
            }
        }
    }
}

void verifySVO32(const EditStructure<void, uint32_t>& editStructure, const SVO32& svo)
{
    verifySVO32_recurse(editStructure, svo, editStructure.rootLevel, 0, 0);
}
