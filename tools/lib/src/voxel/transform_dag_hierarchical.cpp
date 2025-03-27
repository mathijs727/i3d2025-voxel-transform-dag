#include "voxcom/voxel/transform_dag.h"
#include <algorithm>
#include <array>
#include <execution>
#include <glm/vec3.hpp>
#include <limits>
#include <spdlog/spdlog.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "voxcom/utility/disable_all_warnings.h"
DISABLE_WARNINGS_PUSH()
#include <robin_hood.h>
DISABLE_WARNINGS_POP()

namespace voxcom {

// Bitmask storing:
//  * invariance to symmetry I, X, Y, Z, XY, XZ, YZ, XYZ in the three least significant bits (I is the identity transform).
//  * invariance to
struct TransformID {
    uint32_t symmetryID;
    uint32_t axisPermutationID;
};
static constexpr size_t NumTransformIDs = 8 * TransformPointer::NumUniqueAxisPermutations;
static uint32_t encodeTransformID(const TransformID& transformID)
{
    return transformID.symmetryID | (transformID.axisPermutationID << 3);
}
static TransformID decodeTransformID(uint32_t encodedTransformID)
{
    return TransformID {
        .symmetryID = encodedTransformID & 0b111,
        .axisPermutationID = encodedTransformID >> 3
    };
}
static uint32_t encodeSymmetryID(const glm::bvec3& symmetry)
{
    return morton_encode32(glm::uvec3(symmetry));
}
static glm::bvec3 decodeSymmetryID(uint32_t symmetryID)
{
    return glm::bvec3(morton_decode32<3>(symmetryID));
}

static uint32_t applyTransformToBitMask2x2x2(uint32_t bitmask2x2x2, const glm::bvec3& symmetry, const glm::ivec3& axisPermutation)
{
    uint32_t outBitMask2x2x2 = 0;
    for (uint32_t inVoxelIdx = 0; inVoxelIdx < 8; ++inVoxelIdx) {
        if ((bitmask2x2x2 >> inVoxelIdx) & 0b1) {
            const glm::uvec3 inVoxel = morton_decode32<3>(inVoxelIdx);
            const glm::uvec3 outVoxel = applyAxisPermutation(inVoxel, axisPermutation) ^ glm::uvec3(symmetry);
            const auto outVoxelIdx = voxcom::morton_encode32(outVoxel);
            outBitMask2x2x2 |= 1u << outVoxelIdx;
        }
    }
    return outBitMask2x2x2;
}

static TransformPointer applyTransformationToPointer(const TransformPointer& transformPointer, const glm::bvec3& symmetry, const glm::u8vec3& axisPermutation)
{
    TransformPointer out = transformPointer;
    out.setSymmetry(symmetry ^ applyAxisPermutation(transformPointer.getSymmetry(), axisPermutation));
    out.setAxisPermutation(applyAxisPermutation(transformPointer.getAxisPermutation(), axisPermutation));
    return out;
}

struct TaggedPointer {
    TransformPointer transformPointer;
    uint64_t invarianceBitMask; // Bit-mask indicating invariance for each specific transform (see encodeTransformID).

    static TaggedPointer sentinel()
    {
        return { .transformPointer = TransformPointer::sentinel(), .invarianceBitMask = std::numeric_limits<uint64_t>::max() };
    }

    bool operator==(const TaggedPointer& rhs) const
    {
        if (transformPointer.ptr != rhs.transformPointer.ptr)
            return false;
        assert_always(transformPointer.getTranslation() == rhs.transformPointer.getTranslation());
        assert_always(invarianceBitMask == rhs.invarianceBitMask);

        // Comparing lhs (*this) and rhs both referring to the same canonical representation C (transformPointer.ptr).
        // The transformation T_L(C) and T_R(C) applied to lhs & rhs respectively, is described by their transformPointer.
        // Each transformation T_i(x) consists of a symmetry S_i(x) and axis permutation P_i(x):
        // T_i(x) = S_i(P_i(x))
        //
        // The inverse of T_i(x) = T_i^-1(x) is given as:
        // T_i^-1(x) = P_i^-1(S_i^-1(x))
        // Where S_i^-1(x) and P_i^-1(x) are the inverse of the respective symmetry and permutation transformations.
        //
        // We know the invariances of the canonical representation C. In other words, we know for which transformations this holds:
        // C = S(P(C))
        //
        // To answer the question T_L(C) == T_R(C) we rewrite the equation as follows:
        // T_L(C) == T_R(C)
        // T_L^-1(T_L(C)) == T_L^-1(T_R(C))
        // C == T_L^-1(T_R(C))
        //
        // We thus need to find T=T_L^-1(T_R(C)) to tell whether C is invariant to it.
        const auto lhsInverseSymmetry = transformPointer.getSymmetry();
        const auto lhsInverseAxisPermutation = invertPermutation(transformPointer.getAxisPermutation());

#ifndef NDEBUG
        auto identity = applyTransformationToPointer(transformPointer, applyAxisPermutation(lhsInverseSymmetry, lhsInverseAxisPermutation), lhsInverseAxisPermutation);
        assert(identity.getAxisPermutation() == glm::u8vec3(0, 1, 2));
        assert(identity.getSymmetry() == glm::bvec3(false));
#endif
        const auto invLhsTimesRhsSymmetry = applyAxisPermutation(rhs.transformPointer.getSymmetry() ^ lhsInverseSymmetry, lhsInverseAxisPermutation);
        const auto invLhsTimesRhsAxisPermutation = applyAxisPermutation(rhs.transformPointer.getAxisPermutation(), lhsInverseAxisPermutation);

        const TransformID requiredInvariance {
            .symmetryID = encodeSymmetryID(invLhsTimesRhsSymmetry),
            .axisPermutationID = TransformPointer::encodeAxisPermutation(invLhsTimesRhsAxisPermutation)
        };

        const auto requiredInvarianceID = encodeTransformID(requiredInvariance);
        return (invarianceBitMask >> requiredInvarianceID) & 0b1;
    }
};

[[maybe_unused]] static std::string bitmask2x2x2_to_string(uint32_t bitmask2x2x2)
{
    std::string out = "";
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            out += "[";
            for (uint32_t x = 0; x < 2; ++x) {
                const uint32_t idx = x | (y << 1) | (z << 2);
                out += ((bitmask2x2x2 >> idx) & 0b1) ? "x" : " ";
            }
            out += "]\n";
        }
        out += "\n";
    }
    return out;
}
[[maybe_unused]] static std::string bitmask4x4x4_to_string(uint64_t bitmask4x4x4)
{
    std::string out = "";
    for (uint32_t z = 0; z < 4; ++z) {
        for (uint32_t y = 0; y < 4; ++y) {
            out += "[";
            for (uint32_t x = 0; x < 4; ++x) {
                const uint32_t idx = morton_encode32(glm::uvec3(x, y, z));
                out += ((bitmask4x4x4 >> idx) & 0b1) ? "x" : " ";
            }
            out += "]\n";
        }
        out += "\n";
    }
    return out;
}

static auto computeCanonicalLeafRepresentations(const TransformDAGConfig& config)
{
    // Loop over each potential canonical 2x2x2 bitmask.
    // Apply every possible transformation, and store it as the selected representation if
    // it has the smallest value when interpreted as a uint8_t.
    std::array<uint64_t, 256> invarianceBitMasks;
    std::fill(std::begin(invarianceBitMasks), std::end(invarianceBitMasks), 0);
    std::array<TaggedPointer, 256> out;
    std::fill(std::begin(out), std::end(out), TaggedPointer { .transformPointer = { .ptr = 0xFFFF } });
    for (uint32_t canonicalBitMask2x2x2 = 0; canonicalBitMask2x2x2 < 256; ++canonicalBitMask2x2x2) {
        //  Loop over all potential transformations of this 2x2x2 bitmask.
        for (uint32_t encodedtransformID = 0; encodedtransformID < NumTransformIDs; ++encodedtransformID) {
            const auto transformID = decodeTransformID(encodedtransformID);
            const auto axisPermutation = TransformPointer::decodeAxisPermutation(transformID.axisPermutationID);
            const glm::bvec3 symmetry = decodeSymmetryID(transformID.symmetryID);
            const auto transformedBitMask2x2x2 = applyTransformToBitMask2x2x2(canonicalBitMask2x2x2, symmetry, axisPermutation);

            // Only consider the transformations that the user requested.
            if (!config.axisPermutation && transformID.axisPermutationID != 0)
                continue;
            if (!config.symmetry && transformID.symmetryID != 0)
                continue;

            if (canonicalBitMask2x2x2 < out[transformedBitMask2x2x2].transformPointer.ptr) {
                const auto transformPointer = TransformPointer::create(canonicalBitMask2x2x2, symmetry, axisPermutation);
                out[transformedBitMask2x2x2] = TaggedPointer {
                    .transformPointer = transformPointer
                };
            }

            if (transformedBitMask2x2x2 == canonicalBitMask2x2x2)
                invarianceBitMasks[canonicalBitMask2x2x2] |= (uint64_t)1 << encodedtransformID;
        }
    }

    // Store invariances of the selected canonical representation in the TaggedPointers.
    for (TaggedPointer& pointer : out)
        pointer.invarianceBitMask = invarianceBitMasks[pointer.transformPointer.ptr];

    std::unordered_set<uint64_t> uniquePtrs;
    for (const auto& symmetryPointer : out)
        uniquePtrs.insert(symmetryPointer.transformPointer.ptr);
    spdlog::info("Num [sym+perm] canonical 2x2x2 grids: {}", uniquePtrs.size());

    return out;
}

struct TransformNode {
    std::array<TaggedPointer, 8> children;

    static TransformNode fromBitmask4x4x4(uint64_t bitmask4x4x4, const std::array<TaggedPointer, 256>& canonical2x2x2)
    {
        TransformNode out {};
        for (uint32_t index = 0; index < 8; ++index) {
            const uint32_t bitmask2x2x2 = (bitmask4x4x4 >> (8 * index)) & 0xFF;
            out.children[index] = canonical2x2x2[bitmask2x2x2];

            const TransformPointer& childPtr = out.children[index].transformPointer;
            const auto transformedBitMask2x2x2 = applyTransformToBitMask2x2x2((uint32_t)childPtr, childPtr.getSymmetry(), childPtr.getAxisPermutation());
            assert_always(transformedBitMask2x2x2 == bitmask2x2x2);
        }
        return out;
    }

    uint64_t toBitmask4x4x4() const
    {
        uint64_t out = 0;
        for (uint32_t index = 0; index < 8; ++index) {
            const auto child = children[index];
            const auto transformedBitMask2x2x2 = applyTransformToBitMask2x2x2((uint32_t)child.transformPointer.ptr, child.transformPointer.getSymmetry(), child.transformPointer.getAxisPermutation());
            out |= (uint64_t)transformedBitMask2x2x2 << (8 * index);
        }
        return out;
    }

    TransformNode transform(const glm::bvec3& symmetry, const glm::u8vec3& axisPermutation) const
    {
        TransformNode out {};
        for (uint32_t inChildIdx = 0; inChildIdx < 8; ++inChildIdx) {
            // Apply permutation then symmetry.
            const auto outChildIdx = applySymmetry(applyAxisPermutation(inChildIdx, axisPermutation), symmetry);
            out.children[outChildIdx] = children[inChildIdx];

            // Transformation are described as:
            // T(X) = S(P(X))
            // Where S is symmetry and P is axis permutation.
            //
            // We are now applying a parent transformation T_p to the child's transformation T_c:
            // T(X) = T_p(T_c(X))
            // T(X) = S_p(P_p(S_c(P_c(X))))
            // S(P(X)) = S_p(P_p(S_c(P_c(X))))
            //
            // Axis permutations are not affected by symmetry, thus:
            // P(X) = P_p(P_c(X))
            //
            // The child symmetry is affected by the parent permutation:
            // S(X) = S_p(P_p(S_c)(X))
            TransformPointer& transformPointer = out.children[outChildIdx].transformPointer;
            transformPointer = applyTransformationToPointer(transformPointer, symmetry, axisPermutation);
        }
        return out;
    }

    bool operator==(const TransformNode& rhs) const = default;
};
}

template <>
class fmt::formatter<voxcom::TaggedPointer> {
public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr inline auto format(const voxcom::TaggedPointer& pointer, Context& ctx) const
    {
        return fmt::format_to(ctx.out(), "(ptr = {}, invariance = {})", pointer.transformPointer, pointer.invarianceBitMask);
    }
};
template <>
class fmt::formatter<voxcom::TransformNode> {
public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr inline auto format(const voxcom::TransformNode& node, Context& ctx) const
    {
        auto iter = ctx.out();
        iter = fmt::format_to(iter, "{{");
        for (const auto& child : node.children) {
            iter = fmt::format_to(iter, "\t{}\n", child);
        }
        iter = fmt::format_to(iter, "}}");
        return iter;
    }
};

namespace voxcom {
struct TransformNodeHash {
    inline size_t operator()(const TransformNode& node) const noexcept
    {
        size_t seed = 0;
        for (const auto& child : node.children) {
            voxcom::hash_combine(seed, child.transformPointer.ptr);
        }
        return seed;
    }
};

template <template <typename, typename> typename Structure>
EditStructure<void, TransformPointer> constructTransformDAGHierarchical(const Structure<void, uint32_t>& inDag, uint32_t midLevel, std::vector<uint32_t>& outMidLevelInvMapping, const TransformDAGConfig& config)
{
    EditStructure<void, TransformPointer> out;
    out.resolution = inDag.resolution;
    out.rootLevel = inDag.rootLevel;

    // Store a mapping from new nodeIdx to old nodeIdx.
    const auto storeMidLevelMapping = [&](size_t numOutNodes, std::span<const TaggedPointer> prevLevelMapping) {
        outMidLevelInvMapping.resize(numOutNodes);
        for (uint32_t oldNodeIdx = 0; oldNodeIdx < prevLevelMapping.size(); ++oldNodeIdx) {
            const auto& transformPointer = prevLevelMapping[oldNodeIdx].transformPointer;
            if (transformPointer.hasTransform()) // Skip if not a canonical node.
                continue;
            outMidLevelInvMapping[transformPointer.ptr] = oldNodeIdx;
        }
    };

    using NodeMap = std::unordered_map<TransformNode, std::pair<uint32_t, uint64_t>, TransformNodeHash>;
    // using NodeMap = robin_hood::unordered_flat_map<ShiftNode, std::pair<uint32_t, uint64_t>, ShiftNodeHash>;

    // Find a matching node.
    const auto findTransformNode = [&](const TransformNode& node, NodeMap& inOutUniqueNodes, uint64_t& outInvarianceBitMask) -> std::optional<TaggedPointer> {
        // Find matching canonical node which can be transformed into this one.
        outInvarianceBitMask = 0;
        for (uint32_t encodedTransformID = 0; encodedTransformID < NumTransformIDs; ++encodedTransformID) {
            const auto transformID = decodeTransformID(encodedTransformID);
            const auto symmetry = decodeSymmetryID(transformID.symmetryID);
            const auto axisPermutation = TransformPointer::decodeAxisPermutation(transformID.axisPermutationID);

            // Only consider transformations that the user requested.
            if (!config.axisPermutation && transformID.axisPermutationID != 0)
                continue;
            if (!config.symmetry && transformID.symmetryID != 0)
                continue;

            const auto transformedNode = node.transform(symmetry, axisPermutation);
            if (auto iter = inOutUniqueNodes.find(transformedNode); iter != std::end(inOutUniqueNodes)) {
                // We computed how to go from node => matchingNode (WORLD => LOCAL) but we need to store how to go from matchingNode => node (LOCAL => WORLD).
                const auto invAxisPermutation = invertPermutation(axisPermutation);
                const auto transformPointer = TransformPointer::create(iter->second.first, applyAxisPermutation(symmetry, invAxisPermutation), invAxisPermutation);
                outInvarianceBitMask = iter->second.second;

                const auto reconstructedNode = iter->first.transform(transformPointer.getSymmetry(), transformPointer.getAxisPermutation());
                assert_always(reconstructedNode == node);

                return TaggedPointer { .transformPointer = transformPointer, .invarianceBitMask = iter->second.second };
            }

            if (transformedNode == node)
                outInvarianceBitMask |= (uint64_t)1 << encodedTransformID;
        }
        return {};
    };

    const auto processLevel = [&findTransformNode]<typename T>(std::span<const T> inItems, std::span<const TransformNode> inNodes, std::vector<T>& outItems, std::span<TaggedPointer> outPrevLevelMapping) {
        assert_always(outItems.size() == 0);
        assert_always(inItems.size() == inNodes.size());
        assert_always(outPrevLevelMapping.size() == inNodes.size());
#if 1 // Parallel vs easy to understand.
      // Create groups of TransformNodes with the same children pointers (order independent).
        std::vector<std::pair<size_t, uint32_t>> hashKeys(inItems.size());
        {
            std::vector<uint32_t> indices(inItems.size());
            std::iota(std::begin(indices), std::end(indices), 0);
            std::transform(std::execution::par_unseq, std::begin(inNodes), std::end(inNodes), std::begin(indices), std::begin(hashKeys),
                [](TransformNode node, uint32_t index) {
                    std::sort(std::begin(node.children), std::end(node.children), [](const TaggedPointer& lhs, const TaggedPointer& rhs) { return lhs.transformPointer.ptr < rhs.transformPointer.ptr; });
                    size_t seed = 0;
                    for (const auto& child : node.children)
                        hash_combine(seed, child.transformPointer.ptr);
                    return std::pair { seed, index };
                });
        }
        std::sort(std::execution::par_unseq, std::begin(hashKeys), std::end(hashKeys));

        // Sequentially loop over the sorted array to create groups/clusters.
        size_t prevKey = hashKeys[0].first;
        uint32_t prevI = 0;
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        for (uint32_t i = 0; i < hashKeys.size(); ++i) {
            const size_t key = hashKeys[i].first;
            if (key != prevKey) {
                ranges.push_back({ prevI, i });
                prevI = i;
                prevKey = key;
            }
        }
        ranges.push_back({ prevI, (uint32_t)hashKeys.size() });

        // Search within each group of potentially matching nodes (in parallel).
        std::vector<TaggedPointer> uniqueIndices(inItems.size());
        std::vector<uint64_t> invarianceMasks(inItems.size());
        std::for_each(std::execution::par_unseq, std::begin(ranges), std::end(ranges),
            [&](const std::pair<size_t, size_t>& range) {
                const auto [begin, end] = range;
                NodeMap uniqueNodesLUT;
                uniqueNodesLUT.reserve(end - begin);

                for (size_t j = begin; j < end; ++j) {
                    const auto index = hashKeys[j].second;
                    const auto& node = inNodes[index];
                    uint64_t invarianceBitMask;
                    if (auto optExistingNode = findTransformNode(node, uniqueNodesLUT, invarianceBitMask); optExistingNode.has_value()) {
                        uniqueIndices[index] = optExistingNode.value();
                    } else {
                        uniqueNodesLUT[node] = { (uint32_t)index, invarianceBitMask };
                        uniqueIndices[index] = TaggedPointer::sentinel();
                    }
                    invarianceMasks[index] = invarianceBitMask;
                }
            });

        for (const auto& [_, index] : hashKeys) {
            auto taggedPointer = uniqueIndices[index];
            const auto invarianceBitMask = invarianceMasks[index];
            if (taggedPointer != TaggedPointer::sentinel()) {
                taggedPointer.transformPointer.ptr = outPrevLevelMapping[taggedPointer.transformPointer.ptr].transformPointer.ptr;
                outPrevLevelMapping[index] = taggedPointer;
            } else {
                const auto handle = (uint32_t)outItems.size();
                outItems.push_back(inItems[index]);
                taggedPointer.transformPointer = TransformPointer::create(handle);
                taggedPointer.invarianceBitMask = invarianceBitMask;
                outPrevLevelMapping[index] = taggedPointer;
            }
        }
#else
        NodeMap uniqueNodesLUT;
        uniqueNodesLUT.reserve(inItems.size());
        for (size_t i = 0; i < inItems.size(); ++i) {
            const auto& node = inNodes[i];
            uint64_t invarianceBitMask;
            if (auto optExistingNode = findTransformNode(node, uniqueNodesLUT, invarianceBitMask); optExistingNode.has_value()) {
                outPrevLevelMapping[i] = optExistingNode.value();
            } else {
                const auto handle = (uint32_t)outItems.size();
                uniqueNodesLUT[node] = { handle, invarianceBitMask };
                const auto transformPointer = TransformPointer::create(handle);
                outPrevLevelMapping[i] = TaggedPointer { .transformPointer = transformPointer, .invarianceBitMask = invarianceBitMask };
                outItems.push_back(inItems[i]);
            }
        }
#endif
    };

    // Find unique leaf subgrids under symmetry and axis permutations.
    std::vector<TaggedPointer> prevLevelMapping(inDag.subGrids.size());
    {
        const auto canonicalRepresentations = computeCanonicalLeafRepresentations(config);
        std::vector<TransformNode> inTransformNodes(inDag.subGrids.size());
        std::transform(std::execution::par_unseq, std::begin(inDag.subGrids), std::end(inDag.subGrids), std::begin(inTransformNodes),
            [&](const EditSubGrid<void>& subGrid) { return TransformNode::fromBitmask4x4x4(subGrid.bitmask, canonicalRepresentations); });
        processLevel(std::span<const EditSubGrid<void>>(inDag.subGrids), inTransformNodes, out.subGrids, prevLevelMapping);
        spdlog::info("[{}] Reduced SVO leaves from {} to {} using symmetry + axis permutation", out.subGridLevel, inDag.subGrids.size(), out.subGrids.size());
        if (midLevel == out.subGridLevel)
            storeMidLevelMapping(out.subGrids.size(), prevLevelMapping);
    }

    // Traverse inner nodes from the bottom up.
    out.nodesPerLevel.resize(inDag.nodesPerLevel.size());
    for (uint32_t level = inDag.subGridLevel + 1; level <= inDag.rootLevel; ++level) {
        const auto& inLevelNodes = inDag.nodesPerLevel[level];

        // Update child pointers.
        std::vector<TransformNode> inTransformNodes(inLevelNodes.size());
        std::transform(std::execution::par_unseq, std::begin(inLevelNodes), std::end(inLevelNodes), std::begin(inTransformNodes),
            [&](const EditNode<uint32_t>& inNode) {
                TransformNode outNode {};
                for (size_t i = 0; i < 8; ++i) {
                    const auto inChild = inNode.children[i];
                    if (inChild == inNode.EmptyChild)
                        outNode.children[i] = TaggedPointer::sentinel();
                    else
                        outNode.children[i] = prevLevelMapping[inChild];
                }
                return outNode;
            });

        prevLevelMapping.resize(inTransformNodes.size());
        auto& outLevelNodes = out.nodesPerLevel[level];
#if 1
        // Sort indices to reduce transient memory usage.
        std::vector<uint32_t> inIndices(inLevelNodes.size()), outIndices;
        std::iota(std::begin(inIndices), std::end(inIndices), 0);
        processLevel(std::span<const uint32_t>(inIndices), inTransformNodes, outIndices, prevLevelMapping);
        inIndices.clear();
        inIndices.shrink_to_fit();
        assert(inIndices.capacity() == 0);

        outLevelNodes.resize(outIndices.size());
        std::transform(std::execution::par_unseq, std::begin(outIndices), std::end(outIndices), std::begin(outLevelNodes),
            [&](uint32_t index) {
                const auto& inNode = inTransformNodes[index];
                EditNode<TransformPointer> outNode;
                std::transform(std::begin(inNode.children), std::end(inNode.children), std::begin(outNode.children),
                    [](const TaggedPointer& ptr) { return ptr.transformPointer; });
                return outNode;
            });
#else
        // Remove tag from child pointers.
        std::vector<EditNode<TransformPointer>> inEditNodes(inLevelNodes.size());
        std::transform(std::execution::par_unseq, std::begin(inTransformNodes), std::end(inTransformNodes), std::begin(inEditNodes),
            [](const TransformNode& node) {
                std::array<TransformPointer, 8> children;
                std::transform(std::begin(node.children), std::end(node.children), std::begin(children), [](const TaggedPointer& ptr) { return ptr.transformPointer; });
                return EditNode<TransformPointer> { .children = children };
            });

        processLevel(inEditNodes, inTransformNodes, outLevelNodes, prevLevelMapping);
#endif
        spdlog::info("[{}] Reduced SVO nodes from {} to {} using symmetry + axis permutation", level, inLevelNodes.size(), outLevelNodes.size());
        inTransformNodes.clear();

        if (level == midLevel)
            storeMidLevelMapping(outLevelNodes.size(), prevLevelMapping);
    }
    prevLevelMapping.clear();
    return out;
}
template <template <typename, typename> typename Structure>
EditStructure<void, TransformPointer> constructTransformDAGHierarchical(const Structure<void, uint32_t>&& inDag, const TransformDAGConfig& config)
{
    std::vector<uint32_t> dummy;
    return constructTransformDAGHierarchical(inDag, std::numeric_limits<uint32_t>::max(), dummy, config);
}

StaticStructure<void, TransformPointer> constructStaticTransformDAGHierarchical(const StaticStructure<void, uint32_t>& inDag, uint32_t midLevel, std::vector<uint32_t>& outMidLevelRemainingNodeStarts, const TransformDAGConfig& config)
{
    StaticStructure<void, TransformPointer> out;
    out.resolution = inDag.resolution;
    out.rootLevel = inDag.rootLevel;

    // Store a mapping from new nodeIdx to old nodeIdx.
    const auto storeMidLevelMapping = [&](const std::unordered_map<uint32_t, TaggedPointer>& prevLevelMapping) {
        std::vector<std::pair<uint32_t, TaggedPointer>> prevLevelMappingCanonical;
        std::copy_if(std::begin(prevLevelMapping), std::end(prevLevelMapping), std::back_inserter(prevLevelMappingCanonical),
            [](const std::pair<uint32_t, TaggedPointer>& oldStartAndNewPointer) { return !oldStartAndNewPointer.second.transformPointer.hasTransform(); });
        std::sort(std::execution::par_unseq, std::begin(prevLevelMappingCanonical), std::end(prevLevelMappingCanonical), [](const auto& lhs, const auto& rhs) { return lhs.second.transformPointer.ptr < rhs.second.transformPointer.ptr; });
        outMidLevelRemainingNodeStarts.resize(prevLevelMappingCanonical.size());
        std::transform(std::execution::par_unseq, std::begin(prevLevelMappingCanonical), std::end(prevLevelMappingCanonical), std::begin(outMidLevelRemainingNodeStarts),
            [&](const std::pair<uint32_t, TaggedPointer>& oldStartAndNewPointer) { return oldStartAndNewPointer.first; });
        return outMidLevelRemainingNodeStarts;
    };

    using NodeMap = std::unordered_map<TransformNode, std::pair<uint32_t, uint64_t>, TransformNodeHash>;
    // using NodeMap = robin_hood::unordered_flat_map<ShiftNode, std::pair<uint32_t, uint64_t>, TransformNodeHash>;

    // Find a matching node.
    const auto findTransformNode = [&](const TransformNode& node, NodeMap& inOutUniqueNodes, uint64_t& outInvarianceBitMask) -> std::optional<TaggedPointer> {
        // Find matching canonical node which can be transformed into this one.
        outInvarianceBitMask = 0;
        for (uint32_t encodedTransformID = 0; encodedTransformID < NumTransformIDs; ++encodedTransformID) {
            const auto transformID = decodeTransformID(encodedTransformID);
            const auto symmetry = decodeSymmetryID(transformID.symmetryID);
            const auto axisPermutation = TransformPointer::decodeAxisPermutation(transformID.axisPermutationID);

            // Only consider transformations that the user requested.
            if (!config.axisPermutation && transformID.axisPermutationID != 0)
                continue;
            if (!config.symmetry && transformID.symmetryID != 0)
                continue;

            const auto transformedNode = node.transform(symmetry, axisPermutation);
            if (auto iter = inOutUniqueNodes.find(transformedNode); iter != std::end(inOutUniqueNodes)) {
                // We computed how to go from node => matchingNode (WORLD => LOCAL) but we need to store how to go from matchingNode => node (LOCAL => WORLD).
                const auto invAxisPermutation = invertPermutation(axisPermutation);
                const auto transformPointer = TransformPointer::create(iter->second.first, applyAxisPermutation(symmetry, invAxisPermutation), invAxisPermutation);
                outInvarianceBitMask = iter->second.second;

                const auto reconstructedNode = iter->first.transform(transformPointer.getSymmetry(), transformPointer.getAxisPermutation());
                assert_always(reconstructedNode == node);

                return TaggedPointer { .transformPointer = transformPointer, .invarianceBitMask = iter->second.second };
            }

            if (transformedNode == node)
                outInvarianceBitMask |= (uint64_t)1 << encodedTransformID;
        }
        return {};
    };

    const auto processLevel = [&findTransformNode]<typename T>(std::span<const T> inItems, auto itemToTransformNodeinNode, std::vector<T>& outItems, std::span<TaggedPointer> outNextLeveLMapping) {
        assert_always(outItems.size() == 0);
        // assert_always(inItems.size() == inNodes.size());
        assert_always(outNextLeveLMapping.size() == inItems.size());

        // if constexpr (!std::is_same_v<T, EditSubGrid<void>>) {
        //     outItems.resize(inItems.size());
        //     std::copy(std::begin(inItems), std::end(inItems), std::begin(outItems));
        //     for (size_t i = 0; i < inItems.size(); ++i) {
        //         outNextLeveLMapping[i] = TaggedPointer { .transformPointer = TransformPointer::create(i) };
        //     }
        //     return;
        // }

        // Create groups of TransformNodes with the same child pointer addresses (in any order).
        std::vector<std::pair<size_t, uint32_t>> hashKeys(inItems.size());
        {
            std::vector<uint32_t> indices(inItems.size());
            std::iota(std::begin(indices), std::end(indices), 0);
            std::transform(std::execution::par_unseq, std::begin(inItems), std::end(inItems), std::begin(indices), std::begin(hashKeys),
                [&](const T& item, uint32_t index) {
                    TransformNode node = itemToTransformNodeinNode(item);
                    std::sort(std::begin(node.children), std::end(node.children), [](const TaggedPointer& lhs, const TaggedPointer& rhs) { return lhs.transformPointer.ptr < rhs.transformPointer.ptr; });
                    size_t seed = 0;
                    for (const auto& child : node.children)
                        hash_combine(seed, child.transformPointer.ptr);
                    return std::pair { seed, index };
                });
        }
        std::sort(std::execution::par_unseq, std::begin(hashKeys), std::end(hashKeys));
        // Sequentially loop over the sorted array to create groups/clusters.
        size_t prevHash = hashKeys[0].first;
        uint32_t prevIndex = 0;
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        for (uint32_t i = 0; i < hashKeys.size(); ++i) {
            const size_t hash = hashKeys[i].first;
            if (hash != prevHash) {
                ranges.push_back({ prevIndex, i });
                prevIndex = i;
                prevHash = hash;
            }
        }
        ranges.push_back({ prevIndex, (uint32_t)hashKeys.size() });

        // Search within each group of potentially matching nodes (in parallel).
        std::vector<TaggedPointer> uniqueIndices(inItems.size());
        std::vector<uint64_t> invarianceMasks(inItems.size());
        std::for_each(std::execution::par_unseq, std::begin(ranges), std::end(ranges),
            [&](const std::pair<uint32_t, uint32_t>& range) {
                const auto [begin, end] = range;
                NodeMap uniqueNodesLUT;
                uniqueNodesLUT.reserve(end - begin);

                for (uint32_t j = begin; j < end; ++j) {
                    const auto index = hashKeys[j].second;
                    const auto& node = itemToTransformNodeinNode(inItems[index]);
                    uint64_t invarianceBitMask;
                    if (auto optExistingNode = findTransformNode(node, uniqueNodesLUT, invarianceBitMask); optExistingNode.has_value()) {
                        uniqueIndices[index] = optExistingNode.value();
                    } else {
                        uniqueNodesLUT[node] = { index, invarianceBitMask };
                        uniqueIndices[index] = TaggedPointer::sentinel();
                    }
                    invarianceMasks[index] = invarianceBitMask;
                }
            });

        // Store results.
        for (const auto& [_, index] : hashKeys) {
            auto taggedPointer = uniqueIndices[index];
            const auto invarianceBitMask = invarianceMasks[index];
            if (taggedPointer != TaggedPointer::sentinel()) {
                taggedPointer.transformPointer.ptr = outNextLeveLMapping[taggedPointer.transformPointer.ptr].transformPointer.ptr;
                outNextLeveLMapping[index] = taggedPointer;
            } else {
                const auto handle = (uint32_t)outItems.size();
                outItems.push_back(inItems[index]);
                taggedPointer.transformPointer = TransformPointer::create(handle);
                taggedPointer.invarianceBitMask = invarianceBitMask;
                outNextLeveLMapping[index] = taggedPointer;
            }
        }
    };

    // Find unique leaf subgrids under symmetry and axis permutations.
    std::unordered_map<uint32_t, TaggedPointer> prevLevelMapping;
    {
        const auto canonicalRepresentations = computeCanonicalLeafRepresentations(config);
        const auto gridToTransformNode = [&](const EditSubGrid<void>& subGrid) {
            return TransformNode::fromBitmask4x4x4(subGrid.bitmask, canonicalRepresentations);
        };
        std::vector<TaggedPointer> prevLevelMappingVector(inDag.subGrids.size());
        processLevel(std::span<const EditSubGrid<void>>(inDag.subGrids), gridToTransformNode, out.subGrids, prevLevelMappingVector);
        spdlog::info("[{}] Reduced SVO leaves from {} to {} using symmetry + axis permutation", out.subGridLevel, inDag.subGrids.size(), out.subGrids.size());

        prevLevelMapping.reserve(prevLevelMappingVector.size());
        for (uint32_t i = 0; i < inDag.subGrids.size(); ++i)
            prevLevelMapping[i] = prevLevelMappingVector[i];

        if (midLevel == out.subGridLevel)
            storeMidLevelMapping(prevLevelMapping);
    }

    // Traverse inner nodes from the bottom up.
    out.nodesPerLevel.resize(inDag.nodesPerLevel.size());
    for (uint32_t level = inDag.subGridLevel + 1; level <= inDag.rootLevel; ++level) {
        auto nodeStarts = inDag.getLevelNodeStarts(level);

        const auto toTransformNode = [&](uint32_t nodeStart) {
            const StaticNode<uint32_t> inNode = inDag.getNode(level, nodeStart);
            TransformNode outNode {};
            for (uint32_t i = 0; i < 8; ++i) {
                if (inNode.hasChildAtIndex(i))
                    outNode.children[i] = prevLevelMapping.find(inNode.getChildPointerAtIndex(i))->second;
                else
                    outNode.children[i] = EditNode<TaggedPointer>::EmptyChild;
            }
            return outNode;
        };
        std::vector<TaggedPointer> nextLevelMappingVector(nodeStarts.size());
        std::vector<uint32_t> uniqueNodeStarts;
        processLevel(std::span<const uint32_t>(nodeStarts), toTransformNode, uniqueNodeStarts, nextLevelMappingVector);

        spdlog::info("[{}] Reduced SVO nodes from {} to {} using symmetry + axis permutation", level, nodeStarts.size(), uniqueNodeStarts.size());

        // Write the unique nodes to the output.
        auto& outLevelNodes = out.nodesPerLevel[level];
        outLevelNodes.reserve(uniqueNodeStarts.size() * 4);
        std::transform(
            std::begin(uniqueNodeStarts), std::end(uniqueNodeStarts), std::begin(uniqueNodeStarts),
            [&](uint32_t nodeStart) {
                const auto transformNode = toTransformNode(nodeStart);
                const StaticNode<uint32_t> inNode = inDag.getNode(level, nodeStart);
                const uint32_t nodeHandle = (uint32_t)outLevelNodes.size();
                outLevelNodes.push_back(inNode.getChildMask());
                for (TaggedPointer childPointer : transformNode.children) {
                    if (childPointer != EditNode<TaggedPointer>::EmptyChild)
                        outLevelNodes.push_back(std::bit_cast<typename StaticNode<TransformPointer>::BasicType>(childPointer.transformPointer));
                }
                return nodeHandle;
            });

        // Update the updated pointers to point to the newly output'd nodes.
        std::unordered_map<uint32_t, TaggedPointer> nextLevelMapping;
        nextLevelMapping.reserve(nodeStarts.size());
        for (size_t i = 0; i < nodeStarts.size(); ++i) {
            auto transformPointer = nextLevelMappingVector[i];
            transformPointer.transformPointer.ptr = uniqueNodeStarts[transformPointer.transformPointer.ptr];
            nextLevelMapping[nodeStarts[i]] = transformPointer;
        }

        if (level == midLevel)
            storeMidLevelMapping(nextLevelMapping);

        prevLevelMapping = std::move(nextLevelMapping);
    }
    return out;
}
StaticStructure<void, TransformPointer> constructStaticTransformDAGHierarchical(const StaticStructure<void, uint32_t>&& inDag, const TransformDAGConfig& config)
{
    std::vector<uint32_t> dummy;
    return constructStaticTransformDAGHierarchical(inDag, std::numeric_limits<uint32_t>::max(), dummy, config);
}

template EditStructure<void, TransformPointer> constructTransformDAGHierarchical(const EditStructure<void, uint32_t>& inDag, const TransformDAGConfig& config);
template EditStructure<void, TransformPointer> constructTransformDAGHierarchical(const EditStructureOOC<void, uint32_t>& inDag, const TransformDAGConfig& config);
template EditStructure<void, TransformPointer> constructTransformDAGHierarchical(const EditStructure<void, uint32_t>& inDag, uint32_t midLevel, std::vector<uint32_t>& outMidLevelInvMapping, const TransformDAGConfig& config);
template EditStructure<void, TransformPointer> constructTransformDAGHierarchical(const EditStructureOOC<void, uint32_t>& inDag, uint32_t midLevel, std::vector<uint32_t>& outMidLevelInvMapping, const TransformDAGConfig& config);

}