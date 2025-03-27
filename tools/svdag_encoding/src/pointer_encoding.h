#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>
#include <voxcom/utility/disable_all_warnings.h>
#include <voxcom/utility/error_handling.h>
#include <voxcom/utility/hash.h>
#include <voxcom/voxel/structure.h>
DISABLE_WARNINGS_PUSH()
#include <spdlog/spdlog.h>
DISABLE_WARNINGS_POP()

template <typename T, typename Hasher = std::hash<T>>
inline size_t printChecksum(int idx, std::span<const T> items)
{
    size_t checksum = 0;
    for (const T& item : items) {
        voxcom::hash_combine(checksum, Hasher()(item));
    }
    spdlog::info("CHECKSUM {}: {}", idx, checksum);
    return checksum;
}

template <typename T, typename Pointer, typename F>
static std::vector<T> sortByReferenceCount(std::span<const T> items, std::span<voxcom::EditNode<Pointer>> parents, bool prune, F&& filter)
{
    const auto EmptyChild = voxcom::EditNode<Pointer>::EmptyChild;

    // Compute reference counts.
    std::vector<bool> isReferenced(items.size(), false);
    std::vector<uint32_t> referenceCounts(items.size(), 0);
    for (const auto& parent : parents) {
        for (const auto child : parent.children) {
            if (child != EmptyChild) {
                auto idx = (uint32_t)child;
                isReferenced[idx] = true;
                if (filter(child))
                    ++referenceCounts[idx];
            }
        }
    }

    // Compute order based on reference count.
    std::vector<uint32_t> indices(items.size());
    std::iota(std::begin(indices), std::end(indices), 0);
    std::sort(std::begin(indices), std::end(indices), [&](uint32_t lhs, uint32_t rhs) { return referenceCounts[lhs] > referenceCounts[rhs]; });

    // Apply sorting to the items.
    std::vector<T> sortedItems;
    sortedItems.reserve(items.size());
    std::vector<uint32_t> mapping(indices.size());
    for (uint32_t i = 0; i < indices.size(); ++i) {
        // Remove items which are never referenced.
        const auto itemIdx = indices[i];
        if (prune && !isReferenced[itemIdx])
            continue;
        mapping[itemIdx] = (uint32_t)sortedItems.size();
        sortedItems.push_back(items[itemIdx]);
    }

    // Update parent pointers.
    for (auto& parent : parents) {
        for (auto& child : parent.children) {
            if (child != EmptyChild)
                child = mapping[(uint32_t)child];
        }
    }

    return sortedItems;
}

template <typename T, typename Pointer>
static std::vector<T> sortByReferenceCount(std::span<const T> items, std::span<voxcom::EditNode<Pointer>> parents, bool prune = false)
{
    return sortByReferenceCount<T, Pointer>(items, parents, prune, [](const auto&) { return true; });
}

template <typename Attribute, typename Child, typename F>
static void sortStaticStructureLevelByReferenceCount(voxcom::StaticStructure<Attribute, Child>& structure, uint32_t level, bool prune, F&& filter)
{
    // Compute reference counts.
    const auto itemStarts = structure.getLevelNodeStarts(level);
    const auto parentItemStarts = structure.getLevelNodeStarts(level + 1);

    std::vector<bool> isReferenced(itemStarts.size(), false);
    std::vector<uint32_t> referenceCounts(itemStarts.size(), 0);
    for (const auto parentStart : parentItemStarts) {
        const voxcom::StaticNode<Child> node { &structure.nodesPerLevel[level + 1][parentStart] };
        for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
            if (!node.hasChildAtIndex(childIdx))
                continue;
            const auto& child = node.getChildPointerAtIndex(childIdx);
            const auto idx = std::distance(std::begin(itemStarts), std::lower_bound(std::begin(itemStarts), std::end(itemStarts), (uint32_t)child));
            isReferenced[idx] = true;
            if (filter(child))
                ++referenceCounts[idx];
        }
    }

    // Compute order based on reference count.
    std::vector<uint32_t> indices(itemStarts.size());
    std::iota(std::begin(indices), std::end(indices), 0);
    std::sort(std::begin(indices), std::end(indices), [&](uint32_t lhs, uint32_t rhs) { return referenceCounts[lhs] > referenceCounts[rhs]; });

    // Apply sorting to the items.
    std::unordered_map<uint32_t, uint32_t> mapping(indices.size());
    if (level == structure.subGridLevel) {
        std::vector<voxcom::EditSubGrid<Attribute>> sortedItems;
        sortedItems.reserve(itemStarts.size());
        for (uint32_t itemIdx : indices) {
            // Remove items which are never referenced.
            if (prune && !isReferenced[itemIdx])
                continue;
            mapping[itemIdx] = (uint32_t)sortedItems.size();
            sortedItems.push_back(structure.subGrids[itemIdx]);
        }
        structure.subGrids = std::move(sortedItems);
    } else {
        using BasicType = typename voxcom::StaticStructure<Attribute, Child>::BasicType;
        std::vector<BasicType> sortedItems;
        sortedItems.reserve(itemStarts.size());
        for (uint32_t itemIdx : indices) {
            // Remove items which are never referenced.
            if (prune && !isReferenced[itemIdx])
                continue;

            const auto itemStart = itemStarts[itemIdx];
            mapping[itemStart] = (uint32_t)sortedItems.size();
            const auto* pNode = &structure.nodesPerLevel[level][itemStart];
            // Copy node.
            const voxcom::StaticNode<Child> node { pNode };
            for (uint32_t i = 0; i < node.getSizeInBasicType(); ++i)
                sortedItems.push_back(pNode[i]);
        }
        structure.nodesPerLevel[level] = std::move(sortedItems);
    }

    // Update parent pointers.
    auto& parentLevelNodes = structure.nodesPerLevel[level + 1];
    for (uint32_t parentStart : parentItemStarts) {
        voxcom::MutStaticNode<Child> node { &parentLevelNodes[parentStart] };
        for (uint32_t childOffset = 0; childOffset < node.getNumChildren(); ++childOffset) {
            auto childPointer = node.getChildPointerAtOffset(childOffset);
            childPointer.ptr = mapping.find((uint32_t)childPointer.ptr)->second;
            node.setChildPointerAtOffset(childOffset, childPointer);
        }
    }
}

template <typename Attribute, typename Child>
static void sortStaticStructureLevelByReferenceCount(voxcom::StaticStructure<Attribute, Child>& structure, uint32_t level, bool prune)
{
    return sortStaticStructureLevelByReferenceCount<Attribute, Child>(structure, level, prune, [](const auto&) { return true; });
}

template <typename T, typename S>
size_t push(std::vector<T>& stream, S value)
{
    static_assert(sizeof(S) % sizeof(T) == 0);
    const size_t i = stream.size();
    stream.resize(i + sizeof(S) / sizeof(T));
    std::memcpy(&stream[i], &value, sizeof(S));
    return i;
}

template <typename T, typename S>
size_t push(std::vector<T>& stream, S value, uint32_t elements)
{
    static_assert(sizeof(S) % sizeof(T) == 0);
    voxcom::assert_always(sizeof(T) * elements < sizeof(S));
    const size_t i = stream.size();
    stream.resize(i + elements);
    std::memcpy(&stream[i], &value, elements * sizeof(T));
    return i;
}
