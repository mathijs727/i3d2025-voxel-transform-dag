#include "voxcom/voxel/export_hashdag.h"
#include "voxcom/core/bounds.h"
#include "voxcom/utility/binary_writer.h"
#include "voxcom/utility/error_handling.h"
#include "voxcom/voxel/morton.h"
#include <bit>
#include <cassert>
#include <fstream>
#include <span>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <fmt/format.h>
DISABLE_WARNINGS_POP()

// Export functionality for files compatible with the Symmetry-Aware SVDAGf code base:
// https://dl-acm-org.tudelft.idm.oclc.org/doi/abs/10.1145/2856400.2856420

using namespace voxcom;

namespace voxcom {

struct SubHashDAG {
    uint32_t rootNodeIdx;
    uint64_t leafCount; // Number of leafs in subtree.
};

static constexpr uint32_t enclosedLeafLevel = 7;

// TODO: replace by iterative method which doesn't expand a DAG into a tree.
template <typename Attribute>
static SubHashDAG encodeOctreeHashDAG(
    const StaticStructure<Attribute, uint32_t>& structure, std::span<uint32_t> fullyFilledNodes, uint32_t level, uint64_t nodeIdx,
    std::vector<uint32_t>& outTree, std::vector<uint64_t>& outEnclosedLeaves, std::vector<glm::u8vec4>& outLeaves)
{
    uint32_t currentNodeIdx = (uint32_t)outTree.size();
    uint64_t numVoxels = 0;

    if (level == structure.subGridLevel) { // Leaf node.
        const auto& subGrid = structure.subGrids[nodeIdx];
        numVoxels = std::popcount(subGrid.bitmask);
        if constexpr (!std::is_void_v<Attribute>) {
            for (uint32_t voxelIdx = 0; voxelIdx < 64; voxelIdx++) {
                if ((subGrid.bitmask >> voxelIdx) & 0b1) {
                    const auto rgb = subGrid.attributes[voxelIdx];
                    outLeaves.push_back({ rgb.r, rgb.g, rgb.b, 255 });
                }
            }
        }
        outTree.push_back((uint32_t)subGrid.bitmask);
        outTree.push_back((uint32_t)(subGrid.bitmask >> 32));
        return { currentNodeIdx, numVoxels };
    }

    const StaticNode node { &structure.nodesPerLevel[level][nodeIdx] };
    const uint32_t bitmask = node.getChildMask();
    assert_always(bitmask <= 0xFF);
    assert_always(bitmask != 0x00);
    // Upper 24 bits store leaf count in this subtree.
    outTree.push_back(bitmask);
    outTree.resize(outTree.size() + node.getNumChildren());

    // Allocate child pointers.
    for (uint32_t childOffset = 0; childOffset < node.getNumChildren(); childOffset++) {
        auto [childNodeRef, childVoxelCount] = encodeOctreeHashDAG(
            structure, fullyFilledNodes, level - 1, node.getChildHandleAtOffset(childOffset), outTree, outEnclosedLeaves, outLeaves);
        outTree[currentNodeIdx + 1 + childOffset] = childNodeRef;
        numVoxels += childVoxelCount;
    }

    if (level > enclosedLeafLevel) {
        // Write leaf count in this subtree to separate array and store index into that array in the upper 24 bits of the bitmask.
        assert_always(std::bit_width(outEnclosedLeaves.size()) <= 24);
        outTree[currentNodeIdx] |= (uint32_t)outEnclosedLeaves.size() << 8;
        outEnclosedLeaves.push_back(numVoxels);
    } else {
        // Write leaf count in this subtree to the upper 24 bits of the bitmask.
        assert_always(std::bit_width(numVoxels) <= 24);
        outTree[currentNodeIdx] |= (uint32_t)numVoxels << 8;
    }

    return { currentNodeIdx, numVoxels };
}

[[maybe_unused]] static void validate(std::span<const uint32_t> tree, uint32_t leafLevel, uint32_t level, uint32_t nodeIdx)
{
    assert_always(nodeIdx < tree.size());
    if (level == leafLevel)
        return;

    uint32_t const* pNode = &tree[nodeIdx];
    const auto bitmask = pNode[0] & 0xFF;
    for (int i = 0; i < std::popcount(bitmask); ++i) {
        validate(tree, leafLevel, level + 1, pNode[1 + i]);
    }
}

// TODO: replace by iterative method which doesn't expand a DAG into a tree.
template <template <typename, typename> typename Structure, typename Attribute, typename F>
static void collectAttributesDepthFirst(
    const Structure<Attribute, uint32_t>& structure, uint32_t level, uint32_t nodeIdx, std::vector<glm::u8vec4>& out, F&& f)
{
    if (level == structure.subGridLevel) {
        const auto& subGrid = structure.subGrids[nodeIdx];
        for (uint32_t idx = 0; idx < 64; ++idx) { // assume voxels are stored in depth first-order.
            const auto optAttribute = subGrid.get(idx);
            if (optAttribute) {
                out.push_back(f(optAttribute.value()));
            }
        }
    }

    const auto& node = structure.nodesPerLevel[level][nodeIdx];
    for (uint32_t child : node.children) {
        if (child != node.EmptyChild)
            collectAttributesDepthFirst(structure, level - 1, child, out, f);
    }
}

// HashDAG uses 0bXYZ Morton code rather than 0bZYX.
// This code changes the order of the voxels to match the HashDAG format.
static uint64_t fixMortonSubGrid(uint64_t inU64)
{
    uint64_t outU64 = 0;
    for (uint32_t outVoxelIdx = 0; outVoxelIdx < 64; ++outVoxelIdx) {
        const uint32_t x2 = (outVoxelIdx >> 5) & 0b1;
        const uint32_t y2 = (outVoxelIdx >> 4) & 0b1;
        const uint32_t z2 = (outVoxelIdx >> 3) & 0b1;
        const uint32_t x1 = (outVoxelIdx >> 2) & 0b1;
        const uint32_t y1 = (outVoxelIdx >> 1) & 0b1;
        const uint32_t z1 = (outVoxelIdx >> 0) & 0b1;
        const uint32_t inVoxelIdx = (x1 << 0) | (y1 << 1) | (z1 << 2) | (x2 << 3) | (y2 << 4) | (z2 << 5);
        if ((inU64 >> inVoxelIdx) & 0b1) {
            outU64 |= (uint64_t)1 << outVoxelIdx;
        }
    }
    return outU64;
}
static EditNode<uint32_t> fixMortonNode(EditNode<uint32_t> inNode, uint32_t& outChildMask)
{
    EditNode<uint32_t> outNode;
    for (uint32_t outChildIdx = 0; outChildIdx < 8; ++outChildIdx) {
        const uint32_t x = (outChildIdx >> 2) & 0b1;
        const uint32_t y = (outChildIdx >> 1) & 0b1;
        const uint32_t z = (outChildIdx >> 0) & 0b1;
        const uint32_t inChildIdx = (x + 2 * y + 4 * z);
        outNode.children[outChildIdx] = inNode.children[inChildIdx];
        if (inNode.children[inChildIdx] != EditNode<uint32_t>::EmptyChild)
            outChildMask |= 1u << outChildIdx;
    }
    return outNode;
}

template <template <typename, typename> typename Structure, typename Attribute>
void exportHashDAG(const Structure<Attribute, uint32_t>& structure, const Bounds& bounds, const std::filesystem::path& filePath)
{
    std::vector<uint32_t> encodedTree;
    std::vector<glm::u8vec4> leaves;
    std::vector<uint64_t> enclosedLeaves;

    // HashDAG requires that the root node is stored at the start of the array (index=0).
    // Due to the bottom-to-top construction algorithm we write the ndoes in reverse order.
    // To be compatible with HashDAG we place a copy of the root node at the start of the array.
    encodedTree.resize(9); // Reserve space for a copy of the root node.
    enclosedLeaves.push_back(0);

    struct Mapping {
        uint32_t index;
        uint64_t voxelCount;
    };
    std::vector<Mapping> prevLevelMapping(structure.subGrids.size());
    std::transform(std::begin(structure.subGrids), std::end(structure.subGrids), std::begin(prevLevelMapping),
        [&](const EditSubGrid<Attribute>& subGrid) {
            Mapping mapping { .index = (uint32_t)encodedTree.size(), .voxelCount = (uint64_t)std::popcount(subGrid.bitmask) };
            const uint64_t bitmask = fixMortonSubGrid(subGrid.bitmask);
            encodedTree.push_back((uint32_t)bitmask);
            encodedTree.push_back((uint32_t)(bitmask >> 32));
            return mapping;
        });

    for (uint32_t level = structure.subGridLevel + 1; level <= structure.rootLevel; ++level) {
        const auto& inLevelNodes = structure.nodesPerLevel[level];
        std::vector<Mapping> curLevelMapping(inLevelNodes.size());
        std::transform(std::begin(inLevelNodes), std::end(inLevelNodes), std::begin(curLevelMapping),
            [&](EditNode<uint32_t> node) {
                // Write node.
                const auto currentNodeIdx = (uint32_t)encodedTree.size();
                // const uint32_t childMask = node.getChildMask();
                uint32_t childMask = 0;
                node = fixMortonNode(node, childMask);
                assert_always(childMask <= 0xFF);
                assert_always(childMask != 0x00);
                uint64_t voxelCount = 0;
                encodedTree.push_back(childMask);
                for (uint32_t child : node.children) {
                    if (child != node.EmptyChild) {
                        const auto& m = prevLevelMapping[child];
                        encodedTree.push_back(m.index);
                        voxelCount += m.voxelCount;
                    }
                }
                // Upper 24 bits store leaf count in this subtree.
                if (level > enclosedLeafLevel) {
                    // Write leaf count in this subtree to separate array and store index into that array in the upper 24 bits of the bitmask.
                    assert_always(std::bit_width(enclosedLeaves.size()) <= 24);
                    encodedTree[currentNodeIdx] |= (uint32_t)enclosedLeaves.size() << 8;
                    enclosedLeaves.push_back(voxelCount);
                } else {
                    // Write leaf count in this subtree to the upper 24 bits of the bitmask.
                    assert_always(std::bit_width(voxelCount) <= 24);
                    encodedTree[currentNodeIdx] |= (uint32_t)voxelCount << 8;
                }

                return Mapping { .index = currentNodeIdx, .voxelCount = voxelCount };
            });
        prevLevelMapping = std::move(curLevelMapping);
    }

    // Place a copy of the root node at the start of the array.
    std::copy(std::begin(encodedTree) + prevLevelMapping[0].index, std::end(encodedTree), std::begin(encodedTree));
    enclosedLeaves[0] = enclosedLeaves.back();

#ifndef NDEBUG
    validate(encodedTree, structure.subGridLevel, 0, 0);
#endif

    // Add a single dummy value to prevent HashDag from crashing.
    if constexpr (std::is_void_v<Attribute>) {
        leaves.emplace_back(glm::u8vec4(255));
    } else {
        collectAttributesDepthFirst(structure, structure.rootLevel, 0, leaves, [](const RGB& rgb) { return glm::u8vec4(rgb.r, rgb.g, rgb.b, 255); });
    }

    const std::filesystem::path parentPath = filePath.parent_path();
    const std::string fileName = fmt::format("{}{}k.basic_dag", filePath.filename().string(), structure.resolution / 1024);
    // Write DAG
    {
        const std::filesystem::path treeFilePath = parentPath / fmt::format("{}.dag.bin", fileName);
        std::ofstream file { treeFilePath, std::ios::binary };
        BinaryWriter writer { file };
        writer.write(glm::dvec3(bounds.lower));
        writer.write(glm::dvec3(bounds.upper));
        const uint32_t numLevels = (uint32_t)structure.nodesPerLevel.size() - 1;
        writer.write(numLevels);
        writer.write(encodedTree);
    }
    // Write colors.
    {
        const std::filesystem::path colorsFilePath = parentPath / fmt::format("{}.uncompressed_colors.bin", fileName);
        std::ofstream file { colorsFilePath, std::ios::binary };
        BinaryWriter writer { file };
        // Nodes encode number of leafs in subtree in 24 bit mask. Nodes near the top of the tree
        // may have more leafs in their subtree than fit in 24 bits. So the top "topLevels" can have
        // their leafs count encoded as a 64-bit uint in a separate colors file.
        const uint32_t topLevels = (uint32_t)structure.nodesPerLevel.size() - 1 - enclosedLeafLevel;
        writer.write(topLevels);
        if (enclosedLeaves.empty())
            enclosedLeaves.push_back(0); // HashDAG crashes when passing an empty array.
        writer.write(enclosedLeaves);
        writer.write(leaves);
    }
}

template void exportHashDAG(const EditStructure<void, uint32_t>& structure, const Bounds& bounds, const std::filesystem::path& filePath);
template void exportHashDAG(const EditStructureOOC<void, uint32_t>& structure, const Bounds& bounds, const std::filesystem::path& filePath);
template void exportHashDAG(const EditStructure<RGB, uint32_t>& structure, const Bounds& bounds, const std::filesystem::path& filePath);
template void exportHashDAG(const EditStructureOOC<RGB, uint32_t>& structure, const Bounds& bounds, const std::filesystem::path& filePath);
}