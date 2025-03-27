#include "voxcom/voxel/ssvdag.h"
#include "voxcom/voxel/morton.h"
#include "voxcom/voxel/transform_dag.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <execution>
#include <fmt/ranges.h>
#include <glm/vec3.hpp>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace voxcom {

static glm::bvec3 decodeInvarianceID(uint32_t invarianceID)
{
    return glm::bvec3(morton_decode32<3>(invarianceID));
}
static uint32_t encodeInvarianceID(const glm::bvec3& invariance)
{
    return morton_encode32(glm::uvec3(invariance));
}

template <bool ExtendedInvariance>
bool SymmetryPointer<ExtendedInvariance>::operator==(const SymmetryPointer& rhs) const
{
    if (ptr != rhs.ptr)
        return false;
    assert_always(invariance == rhs.invariance);

    if constexpr (ExtendedInvariance) {
        for (uint32_t invarianceID = 0; invarianceID < 8; ++invarianceID) {
            if (!((this->invariance >> invarianceID) & 0b1))
                continue;

            const auto symmetryTransform = this->transform ^ decodeInvarianceID(invarianceID);
            if (glm::all(glm::equal(symmetryTransform, rhs.transform)))
                return true;
        }
        return false;
    } else {
        // For some reason bvec3::operator~() is a no-op, rather than negation.
        const auto negatedInvariance = invariance ^ glm::bvec3(true);
        const auto maskedTransformLhs = negatedInvariance & transform;
        const auto maskedTransformRhs = negatedInvariance & rhs.transform;
        return glm::all(glm::equal(maskedTransformLhs, maskedTransformRhs));
    }
}
template <bool ExtendedInvariance>
SymmetryPointer<ExtendedInvariance>::operator uint32_t() const { return (uint32_t)ptr; }
template <bool ExtendedInvariance>
SymmetryPointer<ExtendedInvariance>& SymmetryPointer<ExtendedInvariance>::operator=(uint32_t p)
{
    this->ptr = p;
    return *this;
}

static uint32_t transformBitmask2x2x2(uint32_t bitmask2x2x2, glm::bvec3 transform)
{
    uint32_t flippedBitmask = 0;
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                const uint32_t newX = transform.x ? 1 - x : x;
                const uint32_t newY = transform.y ? 1 - y : y;
                const uint32_t newZ = transform.z ? 1 - z : z;
                const auto oldIndex = voxcom::morton_encode32(glm::uvec3(x, y, z));
                const auto newIndex = voxcom::morton_encode32(glm::uvec3(newX, newY, newZ));
                if ((bitmask2x2x2 >> oldIndex) & 0b1)
                    flippedBitmask |= 1u << newIndex;
            }
        }
    }
    return flippedBitmask;
}
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
template <bool ExtendedInvariance>
static auto computeCanonicalLeafRepresentations()
{
    std::array<SymmetryPointer<ExtendedInvariance>, 256> out;
    for (uint32_t bitmask2x2x2 = 0; bitmask2x2x2 < 256; ++bitmask2x2x2) {
        uint32_t bestValue = 0;
        glm::bvec3 bestTransform { false };
        auto invariance = invariance_mask<ExtendedInvariance>::no_invariance;
        for (uint32_t z = 0; z < 2; ++z) {
            for (uint32_t y = 0; y < 2; ++y) {
                for (uint32_t x = 0; x < 2; ++x) {
                    const glm::bvec3 transform { (bool)x, (bool)y, (bool)z };
                    const uint32_t transformedBitmask2x2x2 = transformBitmask2x2x2(bitmask2x2x2, transform);
                    if (transformedBitmask2x2x2 > bestValue) {
                        bestTransform = transform;
                        bestValue = transformedBitmask2x2x2;
                    }

                    if constexpr (ExtendedInvariance) {
                        if (transformedBitmask2x2x2 == bitmask2x2x2)
                            invariance |= 1u << encodeInvarianceID(transform);
                    } else {
                        if ((x + y + z) == 1 && transformedBitmask2x2x2 == bitmask2x2x2)
                            invariance |= glm::bvec3(x, y, z);
                    }
                }
            }
        }
        out[bitmask2x2x2] = SymmetryPointer<ExtendedInvariance> { .ptr = bestValue, .transform = bestTransform, .invariance = invariance };
    }

    std::unordered_set<uint64_t> uniquePtrs;
    for (const auto& SymmetryPointer : out)
        uniquePtrs.insert(SymmetryPointer.ptr);

    return out;
}

template <bool ExtendedInvariance>
struct CanonicalRepresentations { };

template <>
struct CanonicalRepresentations<true> {
    static inline const auto value = computeCanonicalLeafRepresentations<true>();
};
template <>
struct CanonicalRepresentations<false> {
    static inline const auto value = computeCanonicalLeafRepresentations<false>();
};

template <bool ExtendedInvariance>
SymmetryNode<ExtendedInvariance> SymmetryNode<ExtendedInvariance>::fromBitmask4x4x4(uint64_t bitmask4x4x4)
{
    SymmetryNode out {};
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                const auto index = voxcom::morton_encode32(glm::uvec3(x, y, z));
                const uint32_t bitmask2x2x2 = (bitmask4x4x4 >> (8 * index)) & 0xFF;
                out.children[index] = CanonicalRepresentations<ExtendedInvariance>::value[bitmask2x2x2];
            }
        }
    }
    return out;
}

template <bool ExtendedInvariance>
uint64_t SymmetryNode<ExtendedInvariance>::toBitmask4x4x4() const
{
    uint64_t out = 0;
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                const auto index = voxcom::morton_encode32(glm::uvec3(x, y, z));
                const auto child = children[index];
                const auto transformedChild = transformBitmask2x2x2(child.ptr, child.transform);
                out |= (uint64_t)transformedChild << (8 * index);
            }
        }
    }
    return out;
}

template <bool ExtendedInvariance>
SymmetryNode<ExtendedInvariance> SymmetryNode<ExtendedInvariance>::mirror(bool mirrorX, bool mirrorY, bool mirrorZ) const
{
    SymmetryNode<ExtendedInvariance> out {};
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                const uint32_t newX = mirrorX ? 1 - x : x;
                const uint32_t newY = mirrorY ? 1 - y : y;
                const uint32_t newZ = mirrorZ ? 1 - z : z;
                const auto oldIndex = voxcom::morton_encode32(glm::uvec3(x, y, z));
                const auto newIndex = voxcom::morton_encode32(glm::uvec3(newX, newY, newZ));
                out.children[newIndex] = children[oldIndex];
                out.children[newIndex].transform ^= glm::bvec3(mirrorX, mirrorY, mirrorZ);
            }
        }
    }
    return out;
}

}

template <bool ExtendedInvariance>
class fmt::formatter<voxcom::SymmetryPointer<ExtendedInvariance>> {
public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr inline auto format(const voxcom::SymmetryPointer<ExtendedInvariance>& pointer, Context& ctx) const
    {
        // return fmt::format_to(ctx.out(), "(ptr = {}, invariance = {})", voxcom::transformBitmask2x2x2(pointer.ptr, pointer.transform), pointer.invariance);
        return fmt::format_to(ctx.out(), "(ptr = {}, transform = {}, invariance = {})", pointer.ptr, pointer.transform, pointer.invariance);
    }
};
template <bool ExtendedInvariance>
class fmt::formatter<voxcom::SymmetryNode<ExtendedInvariance>> {
public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr inline auto format(const voxcom::SymmetryNode<ExtendedInvariance>& node, Context& ctx) const
    {
        auto outIter = ctx.out();
        for (const auto& child : node.children) {
            outIter = fmt::format_to(outIter, "{}\n", child);
        }
        return outIter;
    }
};

namespace voxcom {
struct SymmetryNodeHash {
    template <bool ExtendedInvariance>
    inline size_t operator()(const SymmetryNode<ExtendedInvariance>& node) const noexcept
    {
        size_t seed = 0;
        for (const auto child : node.children) {
            voxcom::hash_combine(seed, child.ptr);
            if constexpr (ExtendedInvariance) {
                uint32_t maxTransformMorton = 0;
                for (uint32_t invarianceID = 0; invarianceID < 8; ++invarianceID) {
                    if (!((child.invariance >> invarianceID) & 0b1))
                        continue;

                    const auto transform = child.transform ^ decodeInvarianceID(invarianceID);
                    maxTransformMorton = std::max(maxTransformMorton, (uint32_t)morton_encode32(glm::uvec3(transform)));
                }
                voxcom::hash_combine(seed, maxTransformMorton);
            } else {
                // WARNING: ~glm::bvec3(...) is a no-op and not a negation!
                const auto negatedInvariance = child.invariance ^ glm::bvec3(true);
                const auto maskedTransform = negatedInvariance & child.transform;
                voxcom::hash_combine(seed, maskedTransform.x);
                voxcom::hash_combine(seed, maskedTransform.y);
                voxcom::hash_combine(seed, maskedTransform.z);
            }
        }
        return seed;
    }
};

template <size_t Level>
static LargeSubGrid<(1u << Level)> createLargeSubGrid2(const EditStructure<void, uint32_t>& structure, uint32_t nodeIdx)
{
    constexpr uint32_t resolution = (1u << Level);
    LargeSubGrid<resolution> out;
    for (uint32_t z = 0; z < resolution; ++z) {
        for (uint32_t y = 0; y < resolution; ++y) {
            for (uint32_t x = 0; x < resolution; ++x) {
                if (structure.get(glm::ivec3(x, y, z), Level, nodeIdx))
                    out.set(glm::uvec3(x, y, z));
            }
        }
    }
    return out;
}

template <bool ExtendedInvariance, template <typename, typename> typename Structure>
EditStructure<void, SymmetryPointer<ExtendedInvariance>> constructSSVDAG(const Structure<void, uint32_t>& octree)
{
    EditStructure<void, SymmetryPointer<ExtendedInvariance>> out;
    out.resolution = octree.resolution;
    out.rootLevel = octree.rootLevel;

    const auto findSymmetryNode = [&](const SymmetryNode<ExtendedInvariance>& node, std::unordered_map<SymmetryNode<ExtendedInvariance>, uint32_t, SymmetryNodeHash>& inOutUniqueNodes, typename invariance_mask<ExtendedInvariance>::type& outInvariance) -> std::optional<SymmetryPointer<ExtendedInvariance>> {
        // Compute invariance.
        outInvariance = invariance_mask<ExtendedInvariance>::no_invariance;
        for (uint32_t transformID = 0; transformID < 8; ++transformID) { // Start from 1; 0 is no transform.
            const auto transform = decodeInvarianceID(transformID);
            const auto transformedNode = node.mirror(transform.x, transform.y, transform.z);
            if constexpr (ExtendedInvariance) {
                // Count all invariances, including combinations of axis (XY, XZ, XZ,XYZ).
                if (transformedNode == node)
                    outInvariance |= 1u << transformID;
            } else {
                // Only count invariance for X, Y, Z.
                if (transform.x && !transform.y && !transform.z && transformedNode == node) {
                    outInvariance.x = true;
                } else if (!transform.x && transform.y && !transform.z && transformedNode == node) {
                    outInvariance.y = true;
                } else if (!transform.x && !transform.y && transform.z && transformedNode == node) {
                    outInvariance.z = true;
                }
            }
        }

        // Find matching node (if any).
        for (uint32_t transformID = 1; transformID < 8; ++transformID) { // Start from 1; 0 is no transform.
            const auto transform = decodeInvarianceID(transformID);
            const auto transformedNode = node.mirror(transform.x, transform.y, transform.z);
            if (auto iter = inOutUniqueNodes.find(transformedNode); iter != std::end(inOutUniqueNodes)) {
                return SymmetryPointer<ExtendedInvariance> { .ptr = iter->second, .transform = transform, .invariance = outInvariance };
            }
        }
        return {};
    };

    assert_always(SymmetryPointer<ExtendedInvariance>::sentinel() == SymmetryPointer<ExtendedInvariance>::sentinel());

    std::vector<SymmetryPointer<ExtendedInvariance>> prevLevelMapping(octree.subGrids.size());
    {
        // Find unique leaf subgrids under symmetry.
        std::unordered_map<SymmetryNode<ExtendedInvariance>, uint32_t, SymmetryNodeHash> uniqueNodesLUT;
        for (size_t i = 0; i < octree.subGrids.size(); ++i) {
            const uint64_t bitmask4x4x4 = octree.subGrids[i].bitmask;
            const SymmetryNode<ExtendedInvariance> node = SymmetryNode<ExtendedInvariance>::fromBitmask4x4x4(bitmask4x4x4);
            // const uint64_t recoveredBitmask4x4x4 = node.toBitmask4x4x4();
            // assert_always(recoveredBitmask4x4x4 == bitmask4x4x4);

            typename invariance_mask<ExtendedInvariance>::type invariance;
            if (auto optExistingNode = findSymmetryNode(node, uniqueNodesLUT, invariance); optExistingNode.has_value()) {
                // const DAG transform { .transform = optExistingNode->transform };
                //  auto transformedSubGrid = transform.transformSubGrid(out.subGrids[optExistingNode->ptr]);
                //  assert_always(transformedSubGrid.bitmask == bitmask4x4x4);
                prevLevelMapping[i] = optExistingNode.value();
            } else {
                const auto handle = (uint32_t)out.subGrids.size();
                uniqueNodesLUT[node] = handle;
                prevLevelMapping[i] = SymmetryPointer<ExtendedInvariance> { .ptr = handle, .transform = glm::bvec3(false), .invariance = invariance };
                out.subGrids.push_back(EditSubGrid<void> { .bitmask = node.toBitmask4x4x4() });
            }
        }
        spdlog::info("[{}] Reduced SVO leaves from {} to {} using symmetry", out.subGridLevel, octree.subGrids.size(), out.subGrids.size());
    }

    // Traverse inner nodes from the bottom up.
    out.nodesPerLevel.resize(octree.nodesPerLevel.size());
    for (uint32_t level = octree.subGridLevel + 1; level <= octree.rootLevel; ++level) {
        const auto& inLevelNodes = octree.nodesPerLevel[level];

        // Update child pointers.
        std::vector<SymmetryNode<ExtendedInvariance>> inSymmetryNodes(inLevelNodes.size());
        std::transform(std::begin(inLevelNodes), std::end(inLevelNodes), std::begin(inSymmetryNodes),
            [&](const EditNode<uint32_t>& inNode) {
                SymmetryNode<ExtendedInvariance> outNode {};
                for (size_t i = 0; i < 8; ++i) {
                    const auto inChild = inNode.children[i];
                    if (inChild == inNode.EmptyChild)
                        outNode.children[i] = SymmetryPointer<ExtendedInvariance>::sentinel();
                    else
                        outNode.children[i] = prevLevelMapping[inChild];
                }
                return outNode;
            });

        // Find duplicates according to symmetry.
        prevLevelMapping.resize(inLevelNodes.size());
        std::unordered_map<SymmetryNode<ExtendedInvariance>, uint32_t, SymmetryNodeHash> uniqueNodesLUT;
        auto& outLevelNodes = out.nodesPerLevel[level];
        for (size_t i = 0; i < inLevelNodes.size(); ++i) {
            const auto& node = inSymmetryNodes[i];
            typename invariance_mask<ExtendedInvariance>::type invariance;
            if (auto optExistingNode = findSymmetryNode(node, uniqueNodesLUT, invariance); optExistingNode.has_value()) {
                prevLevelMapping[i] = optExistingNode.value();
            } else {
                const auto handle = (uint32_t)outLevelNodes.size();
                uniqueNodesLUT[node] = handle;
                prevLevelMapping[i] = SymmetryPointer<ExtendedInvariance> { .ptr = handle, .transform = glm::bvec3(false), .invariance = invariance };
                outLevelNodes.push_back({ .children = node.children });
            }
        }
        spdlog::info("[{}] Reduced SVO nodes from {} to {} using symmetry", level, inLevelNodes.size(), outLevelNodes.size());
    }
    return out;
}

static uint32_t applyMirrorToChildIdx(uint32_t childIdx, const glm::bvec3& mirror)
{
    if (mirror.x)
        childIdx ^= 1;
    if (mirror.y)
        childIdx ^= 2;
    if (mirror.z)
        childIdx ^= 4;
    return childIdx;
}
static glm::uvec3 applyMirrorToVoxelPos(glm::uvec3 voxelPos, const glm::bvec3& mirror)
{
    if (mirror.x)
        voxelPos.x ^= 0b11;
    if (mirror.y)
        voxelPos.y ^= 0b11;
    if (mirror.z)
        voxelPos.z ^= 0b11;
    return voxelPos;
}

template <bool ExtendedInvariance, template <typename, typename> typename Structure>
static void verifySSVDAG_recurse(
    const Structure<void, uint32_t>& editStructure, const EditStructure<void, SymmetryPointer<ExtendedInvariance>>& ssvdag,
    uint32_t level, uint32_t editNodeIdx, uint32_t ssvdagNodeIdx, const glm::bvec3& mirror)
{
    if (level == editStructure.subGridLevel) {
        const auto& editSubGrid = editStructure.subGrids[editNodeIdx];
        const auto& ssvdagSubGrid = ssvdag.subGrids[ssvdagNodeIdx];
        for (uint32_t voxelIdx = 0; voxelIdx < 64; ++voxelIdx) {
            const glm::uvec3 voxelPos = morton_decode32<3>(voxelIdx);
            const glm::uvec3 mirrorVoxelPos = applyMirrorToVoxelPos(voxelPos, mirror);
            const auto expected = editSubGrid.get(voxelPos);
            const auto got = ssvdagSubGrid.get(mirrorVoxelPos);
            assert_always(expected == got);
        }
    } else {
        const auto& editNode = editStructure.nodesPerLevel[level][editNodeIdx];
        const auto& ssvdagNode = ssvdag.nodesPerLevel[level][ssvdagNodeIdx];
        for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
            const uint32_t mirrorChildIdx = applyMirrorToChildIdx(childIdx, mirror);
            const bool editNodeHasChild = editNode.children[childIdx] != EditNode<uint32_t>::EmptyChild;
            const bool dagNodeHasChild = ssvdagNode.children[mirrorChildIdx] != EditNode<SymmetryPointer<ExtendedInvariance>>::EmptyChild;
            assert_always(editNodeHasChild == dagNodeHasChild);

            if (editNodeHasChild) {
                const auto editChild = editNode.children[childIdx];
                const auto ssvdagChild = ssvdagNode.children[mirrorChildIdx];
                verifySSVDAG_recurse<ExtendedInvariance, Structure>(editStructure, ssvdag, level - 1, editChild, ssvdagChild.ptr, mirror ^ ssvdagChild.transform);
            }
        }
    }
}

template <bool ExtendedInvariance, template <typename, typename> typename Structure>
void verifySSVDAG(const Structure<void, uint32_t>& editStructure, const EditStructure<void, SymmetryPointer<ExtendedInvariance>>& ssvdag)
{
    verifySSVDAG_recurse(editStructure, ssvdag, editStructure.rootLevel, 0, 0, glm::bvec3(false));
}

template EditStructure<void, SymmetryPointer<false>> constructSSVDAG(const EditStructure<void, uint32_t>&);
template EditStructure<void, SymmetryPointer<false>> constructSSVDAG(const EditStructureOOC<void, uint32_t>&);
template EditStructure<void, SymmetryPointer<true>> constructSSVDAG(const EditStructure<void, uint32_t>&);
template EditStructure<void, SymmetryPointer<true>> constructSSVDAG(const EditStructureOOC<void, uint32_t>&);

template void verifySSVDAG(const EditStructure<void, uint32_t>&, const EditStructure<void, SymmetryPointer<false>>&);
template void verifySSVDAG(const EditStructure<void, uint32_t>&, const EditStructure<void, SymmetryPointer<true>>&);
template void verifySSVDAG(const EditStructureOOC<void, uint32_t>&, const EditStructure<void, SymmetryPointer<false>>&);
template void verifySSVDAG(const EditStructureOOC<void, uint32_t>&, const EditStructure<void, SymmetryPointer<true>>&);

template struct SymmetryPointer<false>;
template struct SymmetryPointer<true>;

}
