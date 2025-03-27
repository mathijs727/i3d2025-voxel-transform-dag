#include "voxcom/voxel/transform_dag.h"
#include "voxcom/utility/hash.h"
#include "voxcom/utility/maths.h"
#include "voxcom/utility/template_magic.h"
#include "voxcom/voxel/large_sub_grid.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
#include <execution>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <voxcom/utility/error_handling.h>
#include <voxcom/utility/fmt_glm.h>

#include "voxcom/utility/disable_all_warnings.h"
DISABLE_WARNINGS_PUSH()
#include <robin_hood.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
DISABLE_WARNINGS_POP()

static constexpr auto execution_policy = std::execution::par_unseq;
#define USE_GPU 0

namespace voxcom {
template <typename T>
struct is_static_structure : std::false_type { };
template <typename Attribute, typename Child>
struct is_static_structure<StaticStructure<Attribute, Child>> : std::true_type { };
template <typename T>
constexpr bool is_static_structure_v = is_static_structure<T>::value;

template <uint32_t RootLevel, typename Structure, typename OutGrid>
static void createTypedLargeSubGrid(const Structure& structure, uint32_t rootNodeIdx, OutGrid& out)
{
#if 1
    constexpr uint32_t Resolution = (1u << RootLevel);
    constexpr uint32_t Resolution3 = Resolution * Resolution * Resolution;
    constexpr uint32_t VoxelsPerLeaf = 64;
    using Child = typename Structure::Child;

    uint32_t prevBaseVoxelIdx = std::numeric_limits<uint32_t>::max();
    std::array<uint32_t, RootLevel + 1> traversalStack;
    traversalStack[RootLevel] = rootNodeIdx;
    for (uint32_t baseVoxelIdx = 0; baseVoxelIdx < Resolution3; baseVoxelIdx += VoxelsPerLeaf) {
        const uint32_t firstChangedBit = 32 - std::countl_zero(prevBaseVoxelIdx ^ baseVoxelIdx);
        const uint32_t startLevel = std::min(firstChangedBit / 3, RootLevel - 1u);
        prevBaseVoxelIdx = baseVoxelIdx;

        for (uint32_t level = startLevel; level >= structure.subGridLevel; --level) {
            const auto parentNodeIdx = traversalStack[level + 1];
            if (parentNodeIdx != EditNode<Child>::EmptyChild) {
                const auto childIdx = (baseVoxelIdx >> (3 * level)) & 0b111;
                if constexpr (is_static_structure_v<Structure>) {
                    const StaticNode<typename Structure::Child> staticNode { &structure.nodesPerLevel[level + 1][parentNodeIdx] };
                    if (staticNode.hasChildAtIndex(childIdx)) {
                        traversalStack[level] = (uint32_t)staticNode.getChildPointerAtIndex(childIdx);
                    } else
                        traversalStack[level] = EditNode<Child>::EmptyChild;
                } else {
                    const auto& parentNode = structure.nodesPerLevel[level + 1][parentNodeIdx];
                    traversalStack[level] = parentNode.children[childIdx];
                }
            } else {
                traversalStack[level] = EditNode<Child>::EmptyChild;
            }
        }

        const auto subGridIdx = traversalStack[structure.subGridLevel];
        const glm::uvec3 outVoxelBase = morton_decode32<3>(baseVoxelIdx);
        if (subGridIdx != EditNode<Child>::EmptyChild) {
            const EditSubGrid<void>& subGrid = structure.subGrids[subGridIdx];
            for (uint32_t z = 0; z < 4; ++z) {
                for (uint32_t y = 0; y < 4; ++y) {
                    for (uint32_t x = 0; x < 4; ++x) {
                        const glm::uvec3 voxel { x, y, z };
                        const bool filled = subGrid.get(voxel);
                        out.set(outVoxelBase + voxel, filled ? 1 : 0);
                    }
                }
            }
        } else {
            for (uint32_t z = 0; z < 4; ++z) {
                for (uint32_t y = 0; y < 4; ++y) {
                    for (uint32_t x = 0; x < 4; ++x) {
                        const glm::uvec3 voxel { x, y, z };
                        out.set(outVoxelBase + voxel, 0);
                    }
                }
            }
        }
    }

#else
    constexpr uint32_t Resolution = (1u << RootLevel);
    for (uint32_t z = 0; z < Resolution; ++z) {
        for (uint32_t y = 0; y < Resolution; ++y) {
            for (uint32_t x = 0; x < Resolution; ++x) {
                const T value = structure.get(glm::ivec3(x, y, z), RootLevel, rootNodeIdx) ? 1 : 0;
                out.set(glm::uvec3(x, y, z), value);
            }
        }
    }
#endif
}

static uint64_t basicToTransformPointer(uint32_t ptr)
{
    return std::bit_cast<uint64_t>(TransformPointer::create(ptr));
}

static EditNode<TransformPointer> toTransformEditNode(const EditNode<uint32_t>& inNode)
{
    EditNode<TransformPointer> outNode;
    std::transform(
        std::begin(inNode.children), std::end(inNode.children), std::begin(outNode.children),
        [&](uint32_t child) {
            if (child == EditNode<uint32_t>::EmptyChild)
                return TransformPointer::sentinel();
            else
                return TransformPointer::create(child);
        });
    return outNode;
}

#pragma warning(disable : 4702)
template <template <typename, typename> typename Structure>
EditStructure<void, TransformPointer> constructTransformDAG(const Structure<void, uint32_t>& inDag, const TransformDAGConfig& config)
{
    assert_always(inDag.structureType == StructureType::DAG);

    std::vector<uint32_t> translationParents;
    EditStructure<void, TransformPointer> out = constructTransformDAGHierarchical(std::move(inDag), config.maxTranslationLevel + 1, translationParents, config);
    if (!config.translation)
        return out;
    out.subGrids.clear();
    out.subGrids.shrink_to_fit();
    for (uint32_t level = inDag.subGridLevel + 1; level <= config.maxTranslationLevel; ++level) {
        out.nodesPerLevel[level].clear();
        out.nodesPerLevel[level].shrink_to_fit();
    }

    // Copy only the referenced parents from the original DAG.
    assert_always(out.nodesPerLevel[config.maxTranslationLevel + 1].size() == translationParents.size());
    std::transform(execution_policy, std::begin(translationParents), std::end(translationParents), std::begin(out.nodesPerLevel[config.maxTranslationLevel + 1]),
        [&](uint32_t nodeIdx) { return toTransformEditNode(inDag.nodesPerLevel[config.maxTranslationLevel + 1][nodeIdx]); });
    translationParents.clear();
    translationParents.shrink_to_fit();
    assert(translationParents.capacity() == 0);

    // Process the bottom levels from top-to-bottom.
    // The advantage vs bottom-to-top is that when we eliminate nodes at level i+1 we may create children at level i with no parents.
    // These can now be more easily detected and subsequently removed.
    for (uint32_t level = config.maxTranslationLevel; level >= out.subGridLevel; --level) {
        // For the bottom maxTranslationLevel's, we flatten the treelets and run the translation, symmetry, and axis permutation detection.
        assert_always(config.maxTranslationLevel <= MAX_TRANSLATION_LEVEL);

        // Find all unique nodes under translation, symmetry and axis permutation.
        std::vector<TransformPointer> nodeMapping;
        templateForLoop<2, MAX_TRANSLATION_LEVEL>(
            [&]<int Level>(std::integral_constant<int, Level>) {
                const size_t numItems = (Level == inDag.subGridLevel ? inDag.subGrids.size() : inDag.nodesPerLevel[Level].size());
                // Due to re-use, some parents may be eliminated.
                // This may cause nodes at the current level to not have any parents.
                // We remove these unused nodes to save space.
                std::vector<bool> referenced(numItems, false);
                auto& parentNodes = out.nodesPerLevel[Level + 1];
                for (const auto& node : parentNodes) {
                    for (const auto& child : node.children) {
                        if (child != EditNode<TransformPointer>::EmptyChild)
                            referenced[child.ptr] = true;
                    }
                }
                std::vector<uint32_t> indices;
                for (uint32_t i = 0; i < referenced.size(); ++i) {
                    if (referenced[i])
                        indices.push_back(i);
                }
                indices.shrink_to_fit();
                referenced.clear();
                referenced.shrink_to_fit();

// Perform search for duplicates.
#if USE_GPU
                static constexpr size_t Resolution = 1 << Level;
                std::vector<LargeSubGrid<Resolution>> flattenedGrids(indices.size());
                std::transform(
                    execution_policy, std::begin(indices), std::end(indices), std::begin(flattenedGrids),
                    [&](uint32_t idx) { return createLargeSubGrid<Level>(inDag, idx); });
                const auto remainingIndices = findDuplicatesAmongFlattenedItems_gpu<Level, uint32_t>(indices, flattenedGrids, nodeMapping, config);
#else
                FlatGridGenerator<Level, Structure<void, uint32_t>> flatGridGenerator { &inDag, indices };
                const auto remainingIndices = findDuplicatesAmongFlattenedItems_cpp2<Level, uint32_t>(indices, flatGridGenerator, nodeMapping, config);
#endif

                // Output the remaining nodes/leaves, and store a map from old->new addresses.
                if constexpr (Level == inDag.subGridLevel) {
                    out.subGrids.resize(remainingIndices.size());
                    std::transform(
                        execution_policy, std::begin(remainingIndices), std::end(remainingIndices), std::begin(out.subGrids),
                        [&](uint32_t itemIdx) { return inDag.subGrids[itemIdx]; });
                } else {
                    out.nodesPerLevel[Level].resize(remainingIndices.size());
                    std::transform(
                        execution_policy, std::begin(remainingIndices), std::end(remainingIndices), std::begin(out.nodesPerLevel[Level]),
                        [&](uint32_t itemIdx) {
                            auto node = toTransformEditNode(inDag.nodesPerLevel[Level][itemIdx]);
                            return node;
                        });
                }

                // Update the pointers in the parent level.
                std::for_each(execution_policy, std::begin(parentNodes), std::end(parentNodes),
                    [&](EditNode<TransformPointer>& node) {
                        for (TransformPointer& child : node.children) {
                            if (child == TransformPointer::sentinel())
                                continue;
                            const auto iter = std::lower_bound(std::begin(indices), std::end(indices), child.ptr);
                            assert(*iter == child.ptr);
                            child = nodeMapping[std::distance(std::begin(indices), iter)];
                        }
                    });
                nodeMapping.clear();
                nodeMapping.shrink_to_fit();
            },
            level);
    }
    return out;
}

StaticStructure<void, TransformPointer> constructStaticTransformDAG(StaticStructure<void, uint32_t>&& inDag, const TransformDAGConfig& config)
{
    // assert_always(inDag.structureType == StructureType::DAG);
    using BasicType = typename StaticNode<TransformPointer>::BasicType;

    // First construct a Transform SVDAG by hierarchically searching for symmetry + axis permutations.
    // Then we search again at the bottom levels ("config.translationLevel") for translation + symmetry + axis permutations.
    // To keep the code simple (where possible), we replace the bottom levels by the original non-transformed SVDAG nodes (making node -> grid conversion easier).
    std::vector<uint32_t> translationParents;
    StaticStructure<void, TransformPointer> out = constructStaticTransformDAGHierarchical(inDag, config.maxTranslationLevel + 1, translationParents, config);
    if (!config.translation)
        return out;

    // Release memory.
    for (uint32_t level = config.maxTranslationLevel + 2; level <= inDag.rootLevel; ++level)
        inDag.nodesPerLevel[level].clear();

    // Copy the subgrids from the original DAG.
    out.subGrids.resize(inDag.subGrids.size());
    std::copy(std::begin(inDag.subGrids), std::end(inDag.subGrids), std::begin(out.subGrids));
    inDag.subGrids.clear();
    inDag.subGrids.shrink_to_fit();

    // Copy over the bottom levels of the original input SVDAG.
    // Any nodes without a parent will automatically be removed during the top-down compression below.
    for (uint32_t level = inDag.subGridLevel + 1; level <= config.maxTranslationLevel + 1; ++level) {
        std::vector<uint32_t> nodeStarts;
        if (level == config.maxTranslationLevel + 1)
            nodeStarts = std::move(translationParents); // Copy only those nodes which are referenced by the symmetry + axis transform SVDAG.
        else
            nodeStarts = inDag.getLevelNodeStarts(level);

        // Convert from StaticNode<uint32_t> to StaticNode<TransformPointer>.
        auto& inLevelNodes = inDag.nodesPerLevel[level];
        auto& outLevelNodes = out.nodesPerLevel[level];
        const auto tsvdagLevelSize = outLevelNodes.size();
        outLevelNodes.clear();
        for (uint32_t nodeStart : nodeStarts) {
            const StaticNode<uint32_t> inNode = inDag.getNode(level, nodeStart);
            outLevelNodes.push_back(inNode.getChildMask());
            for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
                if (!inNode.hasChildAtIndex(childIdx))
                    continue;
                outLevelNodes.push_back(basicToTransformPointer(inNode.getChildPointerAtIndex(childIdx)));
            }
        }
        if (level == config.maxTranslationLevel + 1)
            assert_always(outLevelNodes.size() == tsvdagLevelSize);
        // Free memory.
        inLevelNodes.clear();
        inLevelNodes.shrink_to_fit();
    }

    // Process the bottom levels from top-to-bottom.
    // The advantage vs bottom-to-top is that when we eliminate nodes at level i+1 we may create children at level i with no parents.
    // These can now be more easily detected and subsequently removed.
    for (uint32_t level = config.maxTranslationLevel; level >= out.subGridLevel; --level) {
        // For the bottom maxTranslationLevel's, we flatten the treelets and run the translation, symmetry, and axis permutation detection.
        assert_always(config.maxTranslationLevel <= MAX_TRANSLATION_LEVEL);

        auto itemStarts = out.getLevelNodeStarts(level);

        // Having previously removed nodes at the parent level may impact items at the current level.
        // If an item loses all its parents, then we don't need to add it to the output DAG.
        // Remove all items that are unreferenced.
        if constexpr (true) {
            spdlog::info("Remove unreferenced nodes/leaves");
            // Loop over the parent nodes to check which items in the current level are referenced.
            const auto& parentLevelNodes = out.nodesPerLevel[level + 1];
            std::vector<bool> referencedItems(level == out.subGridLevel ? out.subGrids.size() : out.nodesPerLevel[level].size(), false);
            uint32_t cursor = 0;
            while (cursor < parentLevelNodes.size()) {
                const StaticNode<TransformPointer> node { &parentLevelNodes[cursor] };
                for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
                    if (node.hasChildAtIndex(childIdx))
                        referencedItems[node.getChildPointerAtIndex(childIdx).ptr] = true;
                }
                cursor += node.getSizeInBasicType();
            }

            // Compact the array of item indices to contain just the once that are referenced by the parent nodes.
            uint32_t outCursor = 0;
            for (uint32_t inCursor = 0; inCursor < itemStarts.size(); ++inCursor) {
                if (referencedItems[itemStarts[inCursor]]) {
                    itemStarts[outCursor++] = itemStarts[inCursor];
                }
            }
            itemStarts.resize(outCursor);
        }

        // Find all unique nodes under translation, symmetry and axis permutation.
        std::vector<TransformPointer> nodeMapping;
        templateForLoop<2, MAX_TRANSLATION_LEVEL>(
            [&]<int Level>(std::integral_constant<int, Level>) {
                spdlog::info("Find translations");
                StaticFlatGridGenerator<Level> flatGridGenerator { &out, itemStarts };
                if constexpr (Level == decltype(out)::subGridLevel) {
                    const auto remainingGridIndices = findDuplicatesAmongFlattenedItems_cpp2<Level, uint32_t>(itemStarts, flatGridGenerator, nodeMapping, config);
                    std::vector<EditSubGrid<void>> newOutSubGrids(remainingGridIndices.size());
                    std::transform(std::execution::par_unseq, std::begin(remainingGridIndices), std::end(remainingGridIndices), std::begin(newOutSubGrids),
                        [&](uint32_t idx) { return out.subGrids[idx]; });
                    out.subGrids = std::move(newOutSubGrids);
                } else {
                    using BasicType = typename decltype(out)::BasicType;
                    // Outputs:
                    // * editNodes      the list of unique/representative nodes
                    // * nodeMapping    maps input node index => transform pointer to editNodes
                    auto& outLevelNodes = out.nodesPerLevel[level];
                    const auto remainingNodeStarts = findDuplicatesAmongFlattenedItems_cpp2<Level, uint32_t>(itemStarts, flatGridGenerator, nodeMapping, config);

                    std::vector<uint64_t> newOutLevelNodes;
                    std::vector<uint32_t> pointerMapping(remainingNodeStarts.size());
                    std::transform(std::begin(remainingNodeStarts), std::end(remainingNodeStarts), std::begin(pointerMapping),
                        [&](uint32_t nodeStart) {
                            const auto childMask = outLevelNodes[nodeStart];
                            const uint32_t nodeSize = 1 + std::popcount(childMask);
                            const uint32_t out = (uint32_t)newOutLevelNodes.size();
                            for (uint32_t i = 0; i < nodeSize; ++i) {
                                newOutLevelNodes.push_back(outLevelNodes[nodeStart + i]);
                            }
                            return out;
                        });
                    outLevelNodes = std::move(newOutLevelNodes);
                    // Update the nodeMapping to point to the StaticNode<TransformPointer>, rather than EditNode<TransformPointer>.
                    for (auto& transformPointer : nodeMapping)
                        transformPointer.ptr = pointerMapping[transformPointer.ptr];
                }
            },
            level);

        spdlog::info("Update pointers at parent level");
        assert_always(std::is_sorted(std::begin(itemStarts), std::end(itemStarts)));
        auto& parentNodes = out.nodesPerLevel[level + 1];
        const auto parentStarts = out.getLevelNodeStarts(level + 1);
        std::for_each(execution_policy, std::begin(parentStarts), std::end(parentStarts),
            [&](uint32_t cursor) {
                const auto childMask = parentNodes[cursor++];
                const uint32_t numChildren = std::popcount(childMask);
                for (uint32_t childOffset = 0; childOffset < numChildren; ++childOffset) {
                    const auto oldPointer = std::bit_cast<TransformPointer>(parentNodes[cursor]);
                    const auto iter = std::lower_bound(std::begin(itemStarts), std::end(itemStarts), oldPointer.ptr);
                    assert_always(*iter == oldPointer.ptr);
                    const auto newPointer = nodeMapping[std::distance(std::begin(itemStarts), iter)];
                    parentNodes[cursor++] = std::bit_cast<BasicType>(newPointer);
                }
            });
    }
    return out;
}

template <uint32_t Level>
StaticFlatGridGenerator<Level>::StaticFlatGridGenerator(const StaticStructure<void, TransformPointer>* pOctree, std::span<const uint32_t> itemStarts)
    : m_pOctree(pOctree)
    , m_itemStarts(itemStarts)
{
}

template <uint32_t Level>
LargeSubGrid<(1u << Level)> StaticFlatGridGenerator<Level>::operator[](uint32_t itemIdx) const
{
    return createLargeSubGrid<Level>(*m_pOctree, m_itemStarts[itemIdx]);
}

template <uint32_t Level>
void StaticFlatGridGenerator<Level>::fillHashGrid(uint32_t itemIdx, TypedLargeSubGrid<uint32_t, (1u << Level)>& out) const
{
    createTypedLargeSubGrid<Level>(*m_pOctree, m_itemStarts[itemIdx], out);
}

template <uint32_t Level>
void StaticFlatGridGenerator<Level>::fillHashGrid(uint32_t itemIdx, TypedLargeSubGrid<uint64_t, (1u << Level)>& out) const
{
    createTypedLargeSubGrid<Level>(*m_pOctree, m_itemStarts[itemIdx], out);
}

template <uint32_t Level, typename Structure>
FlatGridGenerator<Level, Structure>::FlatGridGenerator(const Structure* pStructure, std::span<const uint32_t> indices)
    : m_pStructure(pStructure)
    , m_indices(indices)
{
}

template <uint32_t Level, typename Structure>
LargeSubGrid<(1u << Level)> FlatGridGenerator<Level, Structure>::operator[](uint32_t itemIdx) const
{
    LargeSubGrid<(1u << Level)> out;
    createTypedLargeSubGrid<Level>(*m_pStructure, m_indices[itemIdx], out);
    return out;
}

template <uint32_t Level, typename Structure>
void FlatGridGenerator<Level, Structure>::fillHashGrid(uint32_t itemIdx, TypedLargeSubGrid<uint32_t, (1u << Level)>& out) const
{
    createTypedLargeSubGrid<Level>(*m_pStructure, m_indices[itemIdx], out);
}
template <uint32_t Level, typename Structure>
void FlatGridGenerator<Level, Structure>::fillHashGrid(uint32_t itemIdx, TypedLargeSubGrid<uint64_t, (1u << Level)>& out) const
{
    createTypedLargeSubGrid<Level>(*m_pStructure, m_indices[itemIdx], out);
}

template EditStructure<void, TransformPointer> constructTransformDAG(const EditStructure<void, uint32_t>& inDag, const TransformDAGConfig& config);
template EditStructure<void, TransformPointer> constructTransformDAG(const EditStructureOOC<void, uint32_t>& inDag, const TransformDAGConfig& config);

template class FlatGridGenerator<2, EditStructure<void, TransformPointer>>;
template class FlatGridGenerator<3, EditStructure<void, TransformPointer>>;
template class FlatGridGenerator<4, EditStructure<void, TransformPointer>>;
template class FlatGridGenerator<5, EditStructure<void, TransformPointer>>;

template class FlatGridGenerator<2, EditStructureOOC<void, TransformPointer>>;
template class FlatGridGenerator<3, EditStructureOOC<void, TransformPointer>>;
template class FlatGridGenerator<4, EditStructureOOC<void, TransformPointer>>;
template class FlatGridGenerator<5, EditStructureOOC<void, TransformPointer>>;

template class FlatGridGenerator<2, EditStructure<void, uint32_t>>;
template class FlatGridGenerator<3, EditStructure<void, uint32_t>>;
template class FlatGridGenerator<4, EditStructure<void, uint32_t>>;
template class FlatGridGenerator<5, EditStructure<void, uint32_t>>;

template class FlatGridGenerator<2, EditStructureOOC<void, uint32_t>>;
template class FlatGridGenerator<3, EditStructureOOC<void, uint32_t>>;
template class FlatGridGenerator<4, EditStructureOOC<void, uint32_t>>;
template class FlatGridGenerator<5, EditStructureOOC<void, uint32_t>>;

template class StaticFlatGridGenerator<2>;
template class StaticFlatGridGenerator<3>;
template class StaticFlatGridGenerator<4>;
template class StaticFlatGridGenerator<5>;
}
