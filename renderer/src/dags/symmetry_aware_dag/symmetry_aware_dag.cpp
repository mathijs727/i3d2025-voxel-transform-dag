#include "symmetry_aware_dag.h"
#include "binary_reader.h"
#include "dag_info.h"
#include "utils.h"
#include <algorithm>
#include <span>
#include <vector>

static void verifyTraversal_recurse(
    const SymmetryAwareDAG16& dag, uint32_t level, uint32_t nodeIdx, std::vector<bool>& leafAccessed)
{
    const uint32_t childLevel = level - 1;
    if (level == dag.leafLevel) {
        checkAlways(nodeIdx < dag.leaves.size());
        leafAccessed[nodeIdx] = true;
    } else {
        uint16_t const* pNode = &dag.nodes[dag.levelStarts[level] + nodeIdx];
        check(*pNode != 0);
        const uint32_t bitmask = SymmetryAwareDAG16::convert_child_mask(*pNode);
        for (uint8_t childIdx = 0; childIdx < 8; ++childIdx) {
            if ((bitmask >> childIdx) & 0b1) {
                const auto tmp = dag.get_child_index(childLevel, pNode, *pNode, childIdx);
                verifyTraversal_recurse(dag, childLevel, tmp.index, leafAccessed);
            }
        }
    }
}

static void verifyTraversal(const SymmetryAwareDAG16& dag)
{
    std::vector<bool> leavesVisited(dag.leaves.size(), false);
    verifyTraversal_recurse(dag, dag.levels, dag.get_first_node_index(), leavesVisited);
    for (bool leafVisited : leavesVisited)
        checkAlways(leafVisited);
}

// For some inexplicable reason the bitmask is in a matrix order rather than in a Morton order like a sane person would write.
static Leaf fixLeafBitOrder(Leaf inLeaf)
{
    const uint64_t inBitMask = inLeaf.to_64();
    uint64_t outBitMask = 0;
    uint32_t inIdx = 0;
    for (uint32_t z = 0; z < 4; ++z) {
        for (uint32_t y = 0; y < 4; ++y) {
            for (uint32_t x = 0; x < 4; ++x) {
                const auto inVoxel = (inBitMask >> (inIdx++)) & 0b1;
                const auto outIdx = Utils::morton3D(z, y, x);
                if (inVoxel)
                    outBitMask |= 1llu << outIdx;
            }
        }
    }
    Leaf outLeaf;
    outLeaf.low = uint32_t(outBitMask);
    outLeaf.high = uint32_t(outBitMask >> 32);
    return outLeaf;
}

void SymmetryAwareDAGFactory::load_dag_from_file(DAGInfo& outInfo, SymmetryAwareDAG16& outDag, const std::filesystem::path& path, EMemoryType memoryType)
{

    PROFILE_FUNCTION();
    checkAlways(!outDag.is_valid());

    float3 boundsMin, boundsMax;
    float rootSide;
    uint32_t numLevels;
    uint32_t numElements;
    std::vector<uint16_t> nodes;
    std::vector<Leaf> leaves;
    std::vector<uint32_t> levelStartsInverted;

    BinaryReader reader { path };
    reader.read(boundsMin);
    reader.read(boundsMax);
    reader.read(rootSide);
    reader.read(numLevels);
    checkAlways(numLevels == SCENE_DEPTH);
    reader.read(numElements);

    // Inner nodes.
    uint32_t count;
    reader.read(count);
    nodes.resize(count);
    reader.readRange<uint16_t>(nodes);
    // Leaves.
    reader.read(count);
    checkAlways(count % sizeof(Leaf) == 0);
    count /= sizeof(Leaf);
    leaves.resize(count);
    reader.readRange<Leaf>(leaves);
    // Level offset table.
    reader.read(count);
    levelStartsInverted.resize(count);
    reader.readRange<uint32_t>(levelStartsInverted);

    std::transform(std::begin(leaves), std::end(leaves), std::begin(leaves), fixLeafBitOrder);

    std::vector<uint32_t> levelStarts;
    levelStarts.push_back(0); // L = 0 (1^3)
    levelStarts.push_back(0); // L = 1 (2^3)
    levelStarts.push_back(0); // L = 2 (4^3)
    for (int i = (int)levelStartsInverted.size() - 1; i >= 0; --i)
        levelStarts.push_back(levelStartsInverted[i]);

    outInfo.boundsAABBMin = Vector3(-1.0f, -1.0f, -1.0f);
    outInfo.boundsAABBMax = Vector3(+1.0f, +1.0f, +1.0f);

    outDag.nodes = StaticArray<uint16_t>::allocate("SymmetryAwareDAG16::nodes", nodes, memoryType);
    outDag.leaves = StaticArray<Leaf>::allocate("SymmetryAwareDAG16::leaves", leaves, memoryType);
    outDag.levelStarts = StaticArray<uint32_t>::allocate("SymmetryAwareDAG16::levelStarts", levelStarts, memoryType);

    if (memoryType == EMemoryType::CPU || memoryType == EMemoryType::GPU_Managed)
        verifyTraversal(outDag);
}

SymmetryAwareDAG16::TraversalConstants SymmetryAwareDAG16::createTraversalConstants() const
{
    TraversalConstants out {};
    for (uint32_t symmetry = 0; symmetry < 8; ++symmetry) {
        for (uint32_t inMask = 0; inMask < 256; ++inMask) {
            uint8_t outMask = 0;
            for (uint32_t bitIdx = 0; bitIdx < 8; ++bitIdx) {
                // const uint32_t outBitIdx = bitIdx ^ SymmetryAwareDAG16::SymmetryPointer::invSymmetry(symmetry);
                const uint32_t outBitIdx = bitIdx ^ symmetry;
                if ((inMask >> bitIdx) & 0b1)
                    outMask |= 1u << outBitIdx;
            }
            // outMask = (uint8_t)inMask;
            out.symmetryChildMask[symmetry][inMask] = outMask;
        }
    }
    return out;
}
