#pragma once
#include "morton.h"
#include "voxcom/core/image.h"
#include "voxcom/utility/arg_sort.h"
#include "voxcom/utility/binary_reader.h"
#include "voxcom/utility/binary_writer.h"
#include "voxcom/utility/error_handling.h"
#include "voxcom/utility/hash.h"
#include "voxcom/utility/my_cuda.h"
#include "voxcom/voxel/voxel_grid.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <execution>
#include <filesystem>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream>
#include <span>
#include <stack>
#include <type_traits>
#include <variant>
#include <vector>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <mio/mmap.hpp>
#include <robin_hood.h>
#include <spdlog/spdlog.h>
DISABLE_WARNINGS_POP()

namespace voxcom {

namespace detail {
    template <typename T>
    constexpr T emptyChild() { return T::sentinel(); }
    template <std::integral T>
    constexpr T emptyChild() { return std::numeric_limits<T>::max(); }
}

template <typename Child>
struct EditNode {
    // Can't make constexpr because glm::ivec3 doesn't want to play nicely (be constexpr) in CUDA it seems...
    inline static const Child EmptyChild = detail::emptyChild<Child>();
    // static constexpr Child EmptyChild = detail::emptyChild<Child>();
    std::array<Child, 8> children { EmptyChild, EmptyChild, EmptyChild, EmptyChild, EmptyChild, EmptyChild, EmptyChild, EmptyChild };

    constexpr auto operator<=>(const EditNode&) const noexcept = default;

    inline size_t getStaticNodeSize() const
    {
        size_t nodeSize = 1; // Header.
        for (auto child : children)
            nodeSize += (child != EmptyChild);
        return nodeSize;
    }
    inline uint32_t getChildMask() const
    {
        uint32_t bitmask = 0;
        for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
            if (children[childIdx] != EmptyChild)
                bitmask |= 1u << childIdx;
        }
        return bitmask;
    }
};

struct Dummy {
    int tmp;
};
template <typename T>
struct EditSubGrid {
    uint64_t bitmask = 0;
    std::array<T, 64> attributes;

    inline constexpr bool operator==(const EditSubGrid<T>& other) const noexcept
    {
        if (bitmask != other.bitmask)
            return false;

        for (uint64_t voxelIdx = 0, voxelBit = 1; voxelIdx < 64; ++voxelIdx, voxelBit <<= 1) {
            if ((bitmask & voxelBit) && attributes[voxelIdx] != other.attributes[voxelIdx])
                return false;
        }

        return true;
    }
    inline constexpr bool operator<(const EditSubGrid<T>& other) const noexcept
    {
        return std::hash<EditSubGrid>()(*this) < std::hash<EditSubGrid>()(other);
    }

    inline void set(uint32_t idx, T attribute)
    {
        bitmask |= (uint64_t)1 << idx;
        attributes[idx] = attribute;
    }
    inline void set(const glm::ivec3& voxel, T attribute)
    {
        const auto idx = morton_encode32(glm::uvec3(voxel));
        bitmask |= (uint64_t)1 << idx;
        attributes[idx] = attribute;
    }
    inline std::optional<T> get(uint32_t idx) const
    {
        if (bitmask & ((uint64_t)1 << idx))
            return attributes[idx];
        else
            return {};
    }
    inline auto get(const glm::ivec3& voxel) const { return get(morton_encode32(glm::uvec3(voxel))); }
};
template <>
struct EditSubGrid<void> {
    uint64_t bitmask = 0;

    constexpr auto operator<=>(const EditSubGrid<void>&) const noexcept = default;
    inline void set(uint32_t idx) { bitmask |= (uint64_t)1 << idx; }
    inline void set(const glm::ivec3& voxel) { return set(morton_encode32(glm::uvec3(voxel))); }
    inline bool get(uint32_t idx) const { return bitmask & ((uint64_t)1 << idx); }
    inline bool get(const glm::ivec3& voxel) const { return get(morton_encode32(glm::uvec3(voxel))); }
};

enum class StructureType {
    Tree,
    DAG,
    // MultiDAG // DAG with multiple root nodes
};

template <typename Attribute_, typename Child_>
struct EditStructureOOC {
public:
    using Attribute = Attribute_;
    using Child = Child_;

    std::vector<mio::mmap_source> nodesPerLevelMmaps;
    std::vector<std::span<const EditNode<Child>>> nodesPerLevel;

    mio::mmap_source subGridsMmap;
    std::span<const EditSubGrid<Attribute>> subGrids;

    static constexpr uint32_t subGridLevel = 2; // lvl0 = 1x1x1, lvl1 = 2x2x2, lvl2 = 4x4x4 subgrids
    unsigned resolution = 0;
    uint32_t rootLevel;
    StructureType structureType = StructureType::Tree;

public:
    EditStructureOOC(BinaryReader& reader, const std::filesystem::path& filePath);
    inline bool get(const glm::ivec3&, uint32_t level, uint32_t root) const
        requires std::is_void_v<Attribute>;
    void writeTo(BinaryWriter& writer) const;

    size_t computeVoxelCount() const;
};

template <typename Attribute_, typename Child_>
struct EditStructure {
public:
    using Attribute = Attribute_;
    using Child = Child_;

    std::vector<std::vector<EditNode<Child>>> nodesPerLevel;
    std::vector<EditSubGrid<Attribute>> subGrids;

    static constexpr uint32_t subGridLevel = 2; // lvl0 = 1x1x1, lvl1 = 2x2x2, lvl2 = 4x4x4 subgrids
    unsigned resolution = 0;
    uint32_t rootLevel;
    StructureType structureType = StructureType::Tree;

public:
    EditStructure() = default;
    EditStructure(const EditStructure&) = default;
    EditStructure(EditStructure&&) = default;
    EditStructure(unsigned resolution);
    EditStructure(const VoxelGrid<Attribute>&);
    EditStructure& operator=(const EditStructure&) = default;
    EditStructure& operator=(EditStructure&&) = default;

    bool operator==(const EditStructure<Attribute, Child>&) const = default;

    void set(const glm::ivec3&, std::conditional_t<std::is_void_v<Attribute>, Dummy, Attribute>);
    void set(const glm::ivec3&)
        requires std::is_void_v<Attribute>;
    void set(const glm::ivec3& position, const EditStructure<Attribute, Child>& other);

    // https://brevzin.github.io/c++/2021/11/21/conditional-members/
    // "requires" does not prevent the compiler from skipping over the function completely.
    // Thus having "Attribute&" as an argument would be ill-formed when Attribute is void.
    inline bool get(const glm::ivec3&, std::conditional_t<std::is_void_v<Attribute>, Dummy, Attribute>&, uint32_t root = 0) const;
    inline bool get(const glm::ivec3&, uint32_t root = 0) const
        requires std::is_void_v<Attribute>;
    inline bool get(const glm::ivec3&, uint32_t level, uint32_t root) const
        requires std::is_void_v<Attribute>;

    void toDAG();

    size_t computeInnerNodeCount() const;
    size_t computeLeafCount() const;
    size_t computeItemCount() const;
    size_t computeVoxelCount() const;

    void writeTo(BinaryWriter& writer) const;
    void readFrom(BinaryReader& reader);

private:
    uint32_t constructOctreeFromVoxelGridRecurse(
        const VoxelGrid<Attribute>& voxelGrid, int level, const glm::ivec3& xyz);
};

template <size_t SizeInBytes>
struct static_basic_type { };
template <>
struct static_basic_type<4> {
    using type = uint32_t;
};
template <>
struct static_basic_type<8> {
    using type = uint64_t;
};
template <size_t SizeInBytes>
using static_basic_type_t = typename static_basic_type<SizeInBytes>::type;

namespace impl {
    template <typename Child_, bool immutable>
    struct StaticNodeImpl {
        using BasicType = static_basic_type_t<sizeof(Child_)>;
        std::conditional_t<immutable, const BasicType*, BasicType*> pNode;

        inline uint32_t getSizeInBasicType() const { return 1 + getNumChildren(); }
        inline uint32_t getChildMask() const { return (uint32_t)pNode[0]; }
        inline uint32_t getNumChildren() const { return std::popcount(getChildMask()); }
        inline bool hasChildAtIndex(uint32_t childIndex) const { return (getChildMask() >> childIndex) & 0b1; }
        inline Child_ getChildPointerAtIndex(uint32_t childIndex) const
        {
            assert(hasChildAtIndex(childIndex));
            const uint32_t bitmask = getChildMask();
            const uint32_t preMask = ~(0xFFFFFFFF << childIndex);
            const uint32_t childOffset = std::popcount(bitmask & preMask);
            return getChildPointerAtOffset(childOffset);
        }
        inline Child_ getChildPointerAtOffset(uint32_t childOffset) const
        {
            return std::bit_cast<Child_>(pNode[1 + childOffset]);
        }
    };
}
template <typename Child_>
struct StaticNode : public impl::StaticNodeImpl<Child_, true> {
    using ParentType = impl::StaticNodeImpl<Child_, true>;
    using ParentType::BasicType;

    inline uint32_t getSizeInBasicType() const { return 1 + getNumChildren(); }
    inline uint32_t getChildMask() const { return (uint32_t)this->pNode[0]; }
    inline uint32_t getNumChildren() const { return std::popcount(getChildMask()); }
    inline bool hasChildAtIndex(uint32_t childIndex) const { return (getChildMask() >> childIndex) & 0b1; }
    inline Child_ getChildPointerAtIndex(uint32_t childIndex) const
    {
        assert(this->hasChildAtIndex(childIndex));
        const uint32_t bitmask = getChildMask();
        const uint32_t preMask = ~(0xFFFFFFFF << childIndex);
        const uint32_t childOffset = std::popcount(bitmask & preMask);
        return getChildPointerAtOffset(childOffset);
    }
    inline Child_ getChildPointerAtOffset(uint32_t childOffset) const
    {
        return std::bit_cast<Child_>(this->pNode[1 + childOffset]);
    }
};
template <typename Child_>
struct MutStaticNode : public impl::StaticNodeImpl<Child_, false> {
    using ParentType = impl::StaticNodeImpl<Child_, false>;
    using ParentType::BasicType;

    inline void copyFrom(const StaticNode<Child_>& other)
    {
        const auto nodeSize = other.getSizeInBasicType();
        for (uint32_t i = 0; i < nodeSize; ++i)
            this->pNode[i] = other.pNode[i];
    }

    inline Child_ setChildPointerAtIndex(uint32_t childIndex, Child_ child)
    {
        assert(this->hasChildAtIndex(childIndex));
        const uint32_t bitmask = this->getChildMask();
        const uint32_t preMask = ~(0xFFFFFFFF << childIndex);
        const uint32_t childOffset = std::popcount(bitmask & preMask);
        setChildPointerAtOffset(childOffset, child);
    }
    inline void setChildPointerAtOffset(uint32_t childOffset, Child_ child)
    {
        this->pNode[1 + childOffset] = std::bit_cast<BasicType>(child);
    }
};

template <typename Attribute_, typename Child_>
struct StaticStructure {
public:
    using Attribute = Attribute_;
    using Child = Child_;
    using BasicType = static_basic_type_t<sizeof(Child_)>;

    std::vector<std::vector<BasicType>> nodesPerLevel;
    std::vector<EditSubGrid<Attribute>> subGrids;

    static constexpr uint32_t subGridLevel = 2; // lvl0 = 1x1x1, lvl1 = 2x2x2, lvl2 = 4x4x4 subgrids.
    unsigned resolution = 0;
    uint32_t rootLevel;
    StructureType structureType = StructureType::Tree;

public:
    explicit StaticStructure(const EditStructure<Attribute, Child_>&);
    explicit StaticStructure() = default;

    StaticNode<Child_> getNode(uint32_t level, uint32_t index) const { return { &nodesPerLevel[level][index] }; }
    MutStaticNode<Child_> getMutableNode(uint32_t level, uint32_t index) { return { &nodesPerLevel[level][index] }; }
    std::vector<uint32_t> getLevelNodeStarts(uint32_t level) const;

    // https://brevzin.github.io/c++/2021/11/21/conditional-members/
    // "requires" does not prevent the compiler from skipping over the function completely.
    // Thus having "Attribute&" as an argument would be ill-formed when Attribute is void.
    inline bool get(const glm::ivec3&, std::conditional_t<std::is_void_v<Attribute>, Dummy, Attribute>&, uint32_t root = 0) const;
    inline bool get(const glm::ivec3&, uint32_t root = 0) const
        requires std::is_void_v<Attribute>;
    inline bool get(const glm::ivec3&, uint32_t level, uint32_t root) const
        requires std::is_void_v<Attribute>;

    void writeTo(BinaryWriter& writer) const;
    void readFrom(BinaryReader& reader);
};

template <typename Attribute_, typename Child_>
inline bool EditStructure<Attribute_, Child_>::get(const glm::ivec3& voxel, uint32_t traversalRootLevel, uint32_t root) const
    requires std::is_void_v<Attribute>
{
    uint32_t nodeIdx = root;
    for (uint32_t level = traversalRootLevel; level > subGridLevel; --level) {
        const EditNode<Child_>& node = nodesPerLevel[level][nodeIdx];

        // Bit pattern: y|x
        const int childLevel = level - 1;
        const glm::uvec3 childOffset = (voxel >> childLevel) & 0b1;
        const uint32_t childIdx = morton_encode32(childOffset);

        nodeIdx = node.children[childIdx];
        if (nodeIdx == EditNode<uint32_t>::EmptyChild)
            return false;
    }

    const uint32_t voxelIndex = morton_encode32(glm::uvec3(voxel & 0b11));
    const uint64_t voxelBit = ((uint64_t)1) << voxelIndex;
    return (subGrids[nodeIdx].bitmask & voxelBit);
}

template <typename Attribute_, typename Child_>
EditStructureOOC<Attribute_, Child_>::EditStructureOOC(BinaryReader& reader, const std::filesystem::path& filePath)
{
    reader.readMmap(nodesPerLevel, nodesPerLevelMmaps, filePath);
    reader.readMmap(subGrids, subGridsMmap, filePath);
    reader.read(resolution);
    reader.read(rootLevel);
    reader.read(structureType);
}

template <typename Attribute_, typename Child_>
void EditStructureOOC<Attribute_, Child_>::writeTo(BinaryWriter& writer) const
{
    writer.write(nodesPerLevel);
    writer.write(subGrids);
    writer.write(resolution);
    writer.write(rootLevel);
    writer.write(structureType);
}

template <typename Attribute_, typename Child_>
size_t voxcom::EditStructureOOC<Attribute_, Child_>::computeVoxelCount() const
{
    size_t out = 0;

    struct StackItem {
        uint32_t level;
        uint32_t nodeIdx;
    };
    std::stack<StackItem> stack;
    stack.push({ .level = rootLevel, .nodeIdx = 0 });
    while (!stack.empty()) {
        const auto [level, nodeIdx] = stack.top();
        stack.pop();

        if (level == subGridLevel) {
            const auto& subGrid = subGrids[nodeIdx];
            out += std::popcount(subGrid.bitmask);
        } else {
            const auto& node = nodesPerLevel[level][nodeIdx];
            for (const auto& child : node.children) {
                if (child != EditNode<Child_>::EmptyChild)
                    stack.push({ .level = level - 1, .nodeIdx = (uint32_t)child });
            }
        }
    }

    return out;
}

template <typename Attribute_, typename Child_>
inline EditStructure<Attribute_, Child_>::EditStructure(unsigned resolution)
    : resolution(resolution)
    , structureType(StructureType::Tree)
{
    if (std::popcount(resolution) != 1)
        spdlog::warn("EditStructure has resolution that is not a power of 2");

    // Total number of levels including subgrid levels.
    const uint32_t numLevels = std::bit_width(resolution - 1) + 1;
    nodesPerLevel.resize(numLevels);
    // Construct empty root node.
    nodesPerLevel.back().emplace_back();
    rootLevel = numLevels - 1;
}

template <typename Attribute_, typename Child_>
EditStructure<Attribute_, Child_>::EditStructure(const VoxelGrid<Attribute>& voxelGrid)
    : EditStructure(voxelGrid.resolution)
{
    nodesPerLevel[rootLevel].clear();
    constructOctreeFromVoxelGridRecurse(voxelGrid, rootLevel, glm::ivec3(0));
}

template <typename Attribute_, typename Child_>
size_t EditStructure<Attribute_, Child_>::computeInnerNodeCount() const
{
    size_t out = 0;
    for (const auto& level : nodesPerLevel) {
        out += level.size();
    }
    return out;
}

template <typename Attribute_, typename Child_>
size_t EditStructure<Attribute_, Child_>::computeLeafCount() const
{
    return subGrids.size();
}

template <typename Attribute_, typename Child_>
size_t EditStructure<Attribute_, Child_>::computeItemCount() const
{
    return computeInnerNodeCount() + computeLeafCount();
}

template <typename Attribute_, typename Child_>
size_t EditStructure<Attribute_, Child_>::computeVoxelCount() const
{
    size_t out = 0;

    struct StackItem {
        uint32_t level;
        uint32_t nodeIdx;
    };
    std::stack<StackItem> stack;
    stack.push({ .level = rootLevel, .nodeIdx = 0 });
    while (!stack.empty()) {
        const auto [level, nodeIdx] = stack.top();
        stack.pop();

        if (level == subGridLevel) {
            const auto& subGrid = subGrids[nodeIdx];
            out += std::popcount(subGrid.bitmask);
        } else {
            const auto& node = nodesPerLevel[level][nodeIdx];
            for (const auto& child : node.children) {
                if (child != EditNode<Child_>::EmptyChild)
                    stack.push({ .level = level - 1, .nodeIdx = (uint32_t)child });
            }
        }
    }

    return out;
}

template <typename Attribute_, typename Child_>
void EditStructure<Attribute_, Child_>::readFrom(BinaryReader& reader)
{
    reader.read(nodesPerLevel);
    reader.read(subGrids);
    reader.read(resolution);
    reader.read(rootLevel);
    reader.read(structureType);
}

template <typename Attribute_, typename Child_>
void EditStructure<Attribute_, Child_>::writeTo(BinaryWriter& writer) const
{
    writer.write(nodesPerLevel);
    writer.write(subGrids);
    writer.write(resolution);
    writer.write(rootLevel);
    writer.write(structureType);
}

template <typename Attribute_, typename Child_>
void EditStructure<Attribute_, Child_>::set(const glm::ivec3& voxel, const EditStructure<Attribute, Child>& other)
{
    assert_always(other.rootLevel > 2);
    assert_always(other.rootLevel < this->rootLevel);

    // Skip empty trees.
    if (other.nodesPerLevel[other.rootLevel - 1].empty())
        return;

    uint32_t nodeIdx = 0;
    for (uint32_t level = rootLevel; level > other.rootLevel; --level) {
        EditNode<uint32_t>& node = nodesPerLevel[level][nodeIdx];

        // Bit pattern: y|x
        const uint32_t childLevel = level - 1;
        const glm::uvec3 childOffset = (voxel >> (int)childLevel) & 0b1;
        const uint32_t childIdx = morton_encode32(childOffset);

        auto& child = node.children[childIdx];
        if (child != EditNode<uint32_t>::EmptyChild) {
            nodeIdx = child;
        } else {
            // Create child node if it does not exist yet.
            auto& childLevelNodes = nodesPerLevel[childLevel];
            nodeIdx = child = (uint32_t)childLevelNodes.size();
            if (childLevel != other.rootLevel)
                childLevelNodes.emplace_back();
        }
    }

    // Copy subgrids.
    uint32_t prevLevelOffset = (uint32_t)this->subGrids.size();
    this->subGrids.resize(this->subGrids.size() + other.subGrids.size());
    std::copy(std::begin(other.subGrids), std::end(other.subGrids), std::begin(this->subGrids) + prevLevelOffset);

    // Copy nodes level-by-level while updating their pointers.
    assert_always(this->subGridLevel == other.subGridLevel);
    for (uint32_t level = other.subGridLevel + 1; level <= other.rootLevel; ++level) {
        const auto& inNodes = other.nodesPerLevel[level];
        auto& outNodes = this->nodesPerLevel[level];
        const auto currentLevelOffset = (uint32_t)outNodes.size();
        for (auto node : inNodes) {
            for (uint32_t& child : node.children) {
                if (child != node.EmptyChild)
                    child += prevLevelOffset;
            }
            outNodes.push_back(node);
        }
        prevLevelOffset = currentLevelOffset;
    }

    if (other.structureType == StructureType::DAG)
        this->structureType = StructureType::DAG;
}

template <typename Attribute_, typename Child_>
inline void EditStructure<Attribute_, Child_>::set(const glm::ivec3& voxel, std::conditional_t<std::is_void_v<Attribute>, Dummy, Attribute> value)
{
    uint32_t nodeIdx = 0;
    for (uint32_t level = rootLevel; level > subGridLevel; --level) {
        EditNode<uint32_t>& node = nodesPerLevel[level][nodeIdx];

        // Bit pattern: y|x
        const int childLevel = level - 1;
        const glm::uvec3 childOffset = (voxel >> childLevel) & 0b1;
        const uint32_t childIdx = morton_encode32(childOffset);

        auto& child = node.children[childIdx];
        if (child != EditNode<uint32_t>::EmptyChild) {
            nodeIdx = child;
        } else if (childLevel == subGridLevel) {
            // Create sub grid if it does not exist yet.
            nodeIdx = child = (uint32_t)subGrids.size();
            subGrids.emplace_back();
        } else {
            // Create child node if it does not exist yet.
            auto& childLevelNodes = nodesPerLevel[childLevel];
            nodeIdx = child = (uint32_t)childLevelNodes.size();
            childLevelNodes.emplace_back();
        }
    }

    const uint32_t voxelIndex = morton_encode32(glm::uvec3(voxel & 0b11));
    const uint64_t voxelBit = ((uint64_t)1) << voxelIndex;
    subGrids[nodeIdx].bitmask |= voxelBit;
    if constexpr (!std::is_void_v<Attribute>)
        subGrids[nodeIdx].attributes[voxelIndex] = value;
}

template <typename Attribute_, typename Child_>
inline void EditStructure<Attribute_, Child_>::set(const glm::ivec3& voxel)
    requires std::is_void_v<Attribute>
{
    set(voxel, Dummy {});
}

template <typename Attribute_, typename Child_>
inline bool EditStructure<Attribute_, Child_>::get(const glm::ivec3& voxel, std::conditional_t<std::is_void_v<Attribute>, Dummy, Attribute>& out, uint32_t root) const
{
    uint32_t nodeIdx = root;
    for (uint32_t level = rootLevel; level > subGridLevel; --level) {
        const EditNode<uint32_t>& node = nodesPerLevel[level][nodeIdx];

        // Bit pattern: y|x
        const int childLevel = level - 1;
        const glm::uvec3 childOffset = (voxel >> childLevel) & 0b1;
        const uint32_t childIdx = morton_encode32(childOffset);

        nodeIdx = node.children[childIdx];
        if (nodeIdx == EditNode<uint32_t>::EmptyChild)
            return false;
    }

    const uint32_t voxelIndex = morton_encode32(glm::uvec3(voxel & 0b11));
    const uint64_t voxelBit = ((uint64_t)1) << voxelIndex;
    if (subGrids[nodeIdx].bitmask & voxelBit) {
        if constexpr (!std::is_void_v<Attribute>)
            out = subGrids[nodeIdx].attributes[voxelIndex];
        return true;
    }
    return false;
}

template <typename Attribute_, typename Child_>
inline bool EditStructure<Attribute_, Child_>::get(const glm::ivec3& voxel, uint32_t root) const
    requires std::is_void_v<Attribute>
{
    Dummy dummy {};
    return get(voxel, dummy, root);
}

template <typename Attribute_, typename Child_>
inline bool EditStructureOOC<Attribute_, Child_>::get(const glm::ivec3& voxel, uint32_t traversalRootLevel, uint32_t root) const
    requires std::is_void_v<Attribute>
{
    uint32_t nodeIdx = root;
    for (uint32_t level = traversalRootLevel; level > subGridLevel; --level) {
        const EditNode<Child_>& node = nodesPerLevel[level][nodeIdx];

        // Bit pattern: y|x
        const int childLevel = level - 1;
        const glm::uvec3 childOffset = (voxel >> childLevel) & 0b1;
        const uint32_t childIdx = morton_encode32(childOffset);

        nodeIdx = node.children[childIdx];
        if (nodeIdx == EditNode<uint32_t>::EmptyChild)
            return false;
    }

    const uint32_t voxelIndex = morton_encode32(glm::uvec3(voxel & 0b11));
    const uint64_t voxelBit = ((uint64_t)1) << voxelIndex;
    return (subGrids[nodeIdx].bitmask & voxelBit);
}

template <typename Attribute_, typename Child_>
uint32_t EditStructure<Attribute_, Child_>::constructOctreeFromVoxelGridRecurse(const VoxelGrid<Attribute>& voxelGrid, int level, const glm::ivec3& xyz)
{
    if (level == subGridLevel) {
        // Combine the lowest two levels into a single uint64_t representing the 4x4x4 region.
        EditSubGrid<Attribute> subGrid {};
        for (uint32_t i = 0; i < 64; i++) {
            const glm::ivec3 offset = morton_decode32<3>(i);
            if constexpr (std::is_same_v<Attribute, void>) {
                if (voxelGrid.get(xyz + offset))
                    subGrid.set(i);
            } else {
                Attribute attribute;
                if (voxelGrid.get(xyz + offset, attribute)) {
                    subGrid.set(i, attribute);
                }
            }
        }

        if (subGrid.bitmask == 0)
            return EditNode<uint32_t>::EmptyChild;

        const auto childIdx = (uint32_t)subGrids.size();
        subGrids.emplace_back(subGrid);
        return childIdx;
    } else {
        const unsigned halfLevelSize = 1u << (level - 1);
        EditNode<uint32_t> node;
        for (uint32_t i = 0; i < 8; i++) {
            const glm::ivec3 offset = morton_decode32<3>(i) * halfLevelSize;
            node.children[i] = constructOctreeFromVoxelGridRecurse(voxelGrid, level - 1, xyz + offset);
        }

        // Propagate (don't subdivide) when all children are empty.
        const bool allEmpty = std::transform_reduce(
            std::begin(node.children), std::end(node.children), true, std::logical_and<bool>(),
            [](const auto& child) { return child == EditNode<uint32_t>::EmptyChild; });
        if (allEmpty)
            return EditNode<uint32_t>::EmptyChild;

        nodesPerLevel[level].push_back(node);
        return (uint32_t)nodesPerLevel[level].size() - 1;
    }
}

template <typename Attribute_, typename Child_>
void EditStructure<Attribute_, Child_>::toDAG()
{
// Remove duplicate leaves.
#if 1 // Better time complexity but higher memory usage; slower in practice than parallel sort.
    std::vector<uint32_t> prevLevelMapping;
    {
        robin_hood::unordered_flat_map<EditSubGrid<Attribute_>, uint32_t> subGridLUT;
        std::vector<EditSubGrid<Attribute_>> outSubGrids;
        for (const auto& subGrid : this->subGrids) {
            if (auto iter = subGridLUT.find(subGrid); iter != std::end(subGridLUT)) {
                prevLevelMapping.push_back(iter->second);
            } else {
                const uint32_t handle = (uint32_t)outSubGrids.size();
                outSubGrids.push_back(subGrid);
                subGridLUT[subGrid] = handle;
                prevLevelMapping.push_back(handle);
            }
        }
        this->subGrids = std::move(outSubGrids);
    }
#else
    auto prevLevelMapping = voxcom::invertIndexDirection<uint32_t>(
        voxcom::inPlaceIndexSort<uint32_t, EditSubGrid<Attribute_>>(this->subGrids));
#endif

    for (uint32_t level = subGridLevel + 1; level <= rootLevel; ++level) {
        auto& inNodes = this->nodesPerLevel[level];
        // Update child pointers to level below.
        std::for_each(std::execution::par, std::begin(inNodes), std::end(inNodes),
            [&](EditNode<Child>& node) {
                for (auto& child : node.children) {
                    if (child != EditNode<uint32_t>::EmptyChild)
                        child = prevLevelMapping[child];
                }
            });
        prevLevelMapping.clear();

        // Remove duplicate nodes.
#if 1 // Better time complexity but higher memory usage; slower in practice than parallel sort.
        std::vector<EditNode<Child>> outNodes;
        robin_hood::unordered_flat_map<EditNode<Child>, uint32_t> nodeLUT;
        for (const auto& node : inNodes) {
            if (auto iter = nodeLUT.find(node); iter != std::end(nodeLUT)) {
                prevLevelMapping.push_back(iter->second);
            } else {
                const uint32_t handle = (uint32_t)outNodes.size();
                outNodes.push_back(node);
                nodeLUT[node] = handle;
                prevLevelMapping.push_back(handle);
            }
        }
        this->nodesPerLevel[level] = std::move(outNodes);
#else
        prevLevelMapping = voxcom::invertIndexDirection<uint32_t>(
            voxcom::inPlaceIndexSort<uint32_t, EditNode<Child>>(this->nodesPerLevel[level]));
#endif
    }

    this->structureType = StructureType::DAG;
}

template <typename Attribute, typename Child_>
StaticStructure<Attribute, Child_>::StaticStructure(const EditStructure<Attribute, Child_>& other)
    : resolution(other.resolution)
    , rootLevel(other.rootLevel)
    , structureType(other.structureType)
{
    std::vector<uint32_t> prevLevelMapping(other.subGrids.size());
    std::iota(std::begin(prevLevelMapping), std::end(prevLevelMapping), 0);
    this->subGrids = other.subGrids;

    nodesPerLevel.resize(other.nodesPerLevel.size());
    for (uint32_t level = subGridLevel + 1; level <= rootLevel; ++level) {
        const auto& inNodes = other.nodesPerLevel[level];
        auto& outNodes = nodesPerLevel[level];
        std::vector<uint32_t> curLevelMapping(inNodes.size());
        for (uint32_t inNodeIdx = 0; inNodeIdx < inNodes.size(); ++inNodeIdx) {
            const EditNode<Child_>& inNode = inNodes[inNodeIdx];

            const uint32_t outNodeIdx = (uint32_t)outNodes.size();
            curLevelMapping[inNodeIdx] = outNodeIdx;
            outNodes.emplace_back(0); // Header.

            // Update child pointers in the new structure.
            for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
                const auto& child = inNode.children[childIdx];
                if (child != EditNode<Child_>::EmptyChild) {
                    outNodes.emplace_back(std::bit_cast<BasicType>(prevLevelMapping[(uint32_t)child]));
                    outNodes[outNodeIdx] |= (1u << childIdx);
                }
            }
        }
        prevLevelMapping = std::move(curLevelMapping);
    }
}

template <typename Attribute_, typename Child_>
bool StaticStructure<Attribute_, Child_>::get(const glm::ivec3& voxel, std::conditional_t<std::is_void_v<Attribute>, Dummy, Attribute>& out, uint32_t root /*= 0*/) const
{
    uint32_t nodeIdx = root;
    for (uint32_t level = rootLevel; level > subGridLevel; --level) {
        const StaticNode<Child_> node { &nodesPerLevel[level][nodeIdx] };

        // Bit pattern: y|x
        const int childLevel = level - 1;
        const glm::uvec3 childOffset = (voxel >> childLevel) & 0b1;
        const uint32_t childIdx = morton_encode32(childOffset);

        if (!node.hasChildAtIndex(childIdx))
            return false;
        nodeIdx = (uint32_t)node.getChildPointerAtIndex(childIdx);
    }

    const uint32_t voxelIndex = morton_encode32(glm::uvec3(voxel & 0b11));
    const uint64_t voxelBit = ((uint64_t)1) << voxelIndex;
    if (subGrids[nodeIdx].bitmask & voxelBit) {
        if constexpr (!std::is_void_v<Attribute>)
            out = subGrids[nodeIdx].attributes[voxelIndex];
        return true;
    }
    return false;
}

template <typename Attribute_, typename Child_>
bool StaticStructure<Attribute_, Child_>::get(const glm::ivec3& voxel, uint32_t root /*= 0*/) const
    requires std::is_void_v<Attribute>
{
    Dummy dummy {};
    return get(voxel, dummy, root);
}

template <typename Attribute_, typename Child_>
bool voxcom::StaticStructure<Attribute_, Child_>::get(const glm::ivec3& voxel, uint32_t traversalRootLevel, uint32_t root) const
    requires std::is_void_v<Attribute>
{
    uint32_t nodeIdx = root;
    for (uint32_t level = traversalRootLevel; level > subGridLevel; --level) {
        const StaticNode<Child_>& node { &nodesPerLevel[level][nodeIdx] };

        // Bit pattern: y|x
        const int childLevel = level - 1;
        const glm::uvec3 childOffset = (voxel >> childLevel) & 0b1;
        const uint32_t childIdx = morton_encode32(childOffset);

        if (!node.hasChildAtIndex(childIdx))
            return false;
        nodeIdx = (uint32_t)node.getChildPointerAtIndex(childIdx);
    }

    const uint32_t voxelIndex = morton_encode32(glm::uvec3(voxel & 0b11));
    const uint64_t voxelBit = ((uint64_t)1) << voxelIndex;
    return (subGrids[nodeIdx].bitmask & voxelBit);
}

template <typename Attribute_, typename Child_>
std::vector<uint32_t> StaticStructure<Attribute_, Child_>::getLevelNodeStarts(uint32_t level) const
{
    std::vector<uint32_t> out;
    if (level == subGridLevel) {
        out.resize(subGrids.size());
        std::iota(std::begin(out), std::end(out), 0);
    } else {
        const auto& inLevelNodes = nodesPerLevel[level];
        uint32_t cursor = 0;
        while (cursor < inLevelNodes.size()) {
            out.push_back(cursor);
            StaticNode<Child_> node { &inLevelNodes[cursor] };
            cursor += node.getSizeInBasicType();
        }
    }
    return out;
}

template <typename Attribute_, typename Child_>
void StaticStructure<Attribute_, Child_>::readFrom(BinaryReader& reader)
{
    reader.read(nodesPerLevel);
    reader.read(subGrids);
    reader.read(resolution);
    reader.read(rootLevel);
    reader.read(structureType);
}

template <typename Attribute_, typename Child_>
void StaticStructure<Attribute_, Child_>::writeTo(BinaryWriter& writer) const
{
    writer.write(nodesPerLevel);
    writer.write(subGrids);
    writer.write(resolution);
    writer.write(rootLevel);
    writer.write(structureType);
}
}

template <typename T>
class fmt::formatter<voxcom::EditSubGrid<T>> {
public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr inline auto format(const voxcom::EditSubGrid<T>& subGrid, Context& ctx) const
    {
        for (uint32_t z = 0; z < 4; ++z) {
            for (uint32_t y = 0; y < 4; ++y) {
                fmt::format_to(ctx.out(), "[");
                for (uint32_t x = 0; x < 4; ++x) {
                    const glm::uvec3 voxel { x, y, z };
                    bool voxelSet;
                    if constexpr (std::is_void_v<T>)
                        voxelSet = subGrid.get(voxel);
                    else
                        voxelSet = subGrid.get(voxel).has_value();
                    fmt::format_to(ctx.out(), "{} ", voxelSet ? "x" : " ");
                }
                fmt::format_to(ctx.out(), "]\n");
            }
            fmt::format_to(ctx.out(), "\n");
        }
        return ctx.out();
    }
};

namespace std {

template <typename T>
struct hash<voxcom::EditNode<T>> {
    inline size_t operator()(const voxcom::EditNode<T>& node) const
    {
        size_t seed = 0;
        for (const auto child : node.children)
            voxcom::hash_combine(seed, child);
        return seed;
    }
};

template <typename Attribute>
struct hash<voxcom::EditSubGrid<Attribute>> {
    inline size_t operator()(const voxcom::EditSubGrid<Attribute>& subGrid) const
    {
        size_t seed = 0;
        voxcom::hash_combine(seed, subGrid.bitmask);
        if constexpr (!std::is_void_v<Attribute>) {
            for (uint64_t voxelIdx = 0, voxelBit = 1; voxelIdx < 64; ++voxelIdx, voxelBit <<= 1) {
                if (subGrid.bitmask & voxelBit)
                    voxcom::hash_combine(seed, subGrid.attributes[voxelIdx]);
            }
        }
        return seed;
    }
};

}
