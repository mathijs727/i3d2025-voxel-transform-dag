#include "voxcom/voxel/atomic_encoded_octree.h"
#include "voxcom/utility/error_handling.h"
#include "voxcom/voxel/encoded_octree.h"
#include <bit>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <limits>
#include <unordered_map>
#include <variant>

namespace voxcom {

template <typename Attribute>
typename AtomicEncodedOctree<Attribute>::Node AtomicEncodedOctree<Attribute>::addSubtree(const Octree<Attribute>& octree)
{
    assert_supported_tree_type<TreeType::Tree>(octree);

    // Encode leaf nodes.
    std::unordered_map<size_t, std::pair<Node, Node>> subGridMapping;
    if constexpr (!std::is_void_v<Attribute>) {
        size_t numAttributes = 0;
        for (const auto& subGrid : octree.subGrids)
            numAttributes += std::popcount(subGrid.bitmask);

        auto attributesIter = attributes.grow_by(numAttributes);
        for (uint32_t subGridIdx = 0; subGridIdx < octree.subGrids.size(); ++subGridIdx) {
            const auto& subGrid = octree.subGrids[subGridIdx];
            if (subGrid.bitmask == 0)
                continue;

            // Visit 4x4x4 region that is described by this subtree.
            const uint64_t firstAttributePtr = attributesOffset + std::distance(std::begin(attributes), attributesIter);
            for (uint64_t voxelIdx = 0, voxelBit = 1; voxelIdx < 64; ++voxelIdx, voxelBit <<= 1) {
                if (subGrid.bitmask & voxelBit)
                    *attributesIter++ = subGrid.attributes[voxelIdx];
            }

            // Store the 4x4x4 bitmask and pointer to the attributes.
            subGridMapping[subGridIdx] = std::pair {
                Node { .bitmask4x4x4 = subGrid.bitmask },
                Node { .firstAttributePtr = firstAttributePtr }
            };
        }
    } else {
        for (uint32_t subGridIdx = 0; subGridIdx < octree.subGrids.size(); ++subGridIdx) {
            const auto& subGrid = octree.subGrids[subGridIdx];
            if (subGrid.bitmask == 0)
                continue;

            // Store the 4x4x4 bitmask and pointer to the attributes.
            subGridMapping[subGridIdx] = std::pair {
                Node { .bitmask4x4x4 = subGrid.bitmask },
                Node { .firstAttributePtr = 0 }
            };
        }
    }

    // Encode inner nodes.
    std::unordered_map<uint32_t, Node> prevLevelChildNodeMapping;
    for (size_t parentLevel = subGridLevel + 1; parentLevel < octree.nodesPerLevel.size(); parentLevel++) {
        std::unordered_map<uint32_t, Node> curLevelChildNodeMapping;
        const auto& inNodes = octree.nodesPerLevel[parentLevel];
        auto& outChildNodes = nodesPerLevel[parentLevel - 1];

        // Every child is referenced exactly once.
        auto nodesIter = outChildNodes.grow_by(parentLevel == subGridLevel + 1 ? subGridMapping.size() * 2 : prevLevelChildNodeMapping.size());

        for (uint32_t inNodeIdx = 0; inNodeIdx < inNodes.size(); inNodeIdx++) {
            // Store the children consecutively in memory.
            const uint64_t firstChildPtr = offsetPerLevel[parentLevel - 1] + std::distance(std::begin(outChildNodes), nodesIter);
            const auto& inNode = inNodes[inNodeIdx];
            uint64_t bitmask = 0;
            for (uint32_t childIdx = 0; childIdx < 8; childIdx++) {
                const auto& child = inNode.children[childIdx];
                if (child != EmptyChild) {
                    if (parentLevel == octree.subGridLevel + 1) {
                        const auto [subGridBitmask, firstAttributePtr] = subGridMapping.find(child)->second;
                        *nodesIter++ = subGridBitmask;
                        *nodesIter++ = firstAttributePtr;
                    } else {
                        *nodesIter++ = prevLevelChildNodeMapping.find(child)->second;
                    }
                    bitmask |= (uint64_t)1 << childIdx;
                }
            }

            assert_always(bitmask <= 0xFF);
            assert_always(firstChildPtr < (((uint64_t)1 << 56) - 1));
            Node outNode {};
            outNode.inner.bitmask = bitmask;
            outNode.inner.firstChildPtr = firstChildPtr;
            curLevelChildNodeMapping[inNodeIdx] = outNode;
        }

        prevLevelChildNodeMapping = std::move(curLevelChildNodeMapping);
        subGridMapping.clear();
    }

    assert(prevLevelChildNodeMapping.size() == 1); // Root level has 1 node.
    return std::begin(prevLevelChildNodeMapping)->second;
}

template <typename T>
static std::vector<T> readCombine(std::optional<TempFile>& optFile, const tbb::concurrent_vector<T>& inMemory)
{
    std::vector<T> out;
    if (optFile) {
        optFile->setMode(TempFile::Mode::Read);

        // Get the size of the file.
        auto& is = optFile->fileStream;
        is.seekg(0, std::ios::beg);
        const auto fileStart = is.tellg();
        is.seekg(0, std::ios::end);
        const size_t fileSize = is.tellg() - fileStart;

        // Reset the cursor back to the beginning of the file.
        is.seekg(0, std::ios::beg);

        // Read data from the file into the output vector.
        assert_always(fileSize % sizeof(T) == 0);
        out.resize(fileSize / sizeof(T));
        is.read(reinterpret_cast<char*>(out.data()), fileSize);
    }

    // Append the in memory data to the output vector.
    const size_t sizeBefore = out.size();
    out.resize(out.size() + inMemory.size());
    std::copy(std::begin(inMemory), std::end(inMemory), std::begin(out) + sizeBefore);
    return out;
}

template <typename Attribute>
EncodedOctree<Attribute> AtomicEncodedOctree<Attribute>::toEncodedOctree()
{
    EncodedOctree<Attribute> out {};
    out.resolution = resolution;
    out.rootLevel = rootLevel;
    for (size_t level = 0; level < nodesPerLevel.size(); level++) {
        out.nodesPerLevel.emplace_back(readCombine<Node>(filePerLevel[level], nodesPerLevel[level]));
    }
    if constexpr (!std::is_void_v<Attribute>)
        out.attributes = readCombine<Attribute>(optAttributesFile, attributes);
    return out;
}

template <typename Attribute>
AtomicEncodedOctree<Attribute>::AtomicEncodedOctree(const EncodedOctree<Attribute>& octree)
    : resolution(octree.resolution)
    , rootLevel(octree.rootLevel)
{
    if constexpr (!std::is_void_v<Attribute>) {
        auto iter = attributes.grow_by(octree.attributes.size());
        std::copy(std::begin(octree.attributes), std::end(octree.attributes), iter);
    }
    filePerLevel.resize(octree.nodesPerLevel.size());
    offsetPerLevel.resize(octree.nodesPerLevel.size(), 0);
    for (const auto& levelNodes : octree.nodesPerLevel) {
        nodesPerLevel.emplace_back(std::begin(levelNodes), std::end(levelNodes));
    }
}

template <typename T>
static void write(std::optional<TempFile>& optFile, tbb::concurrent_vector<T>& vector, size_t& offset)
{
    static_assert(std::is_trivially_copyable_v<T>);

    if (!optFile)
        optFile.emplace();

    // std::copy to a separate buffer is much faster than writing to the ofstream from the tbb::concurrent_vector directly.
    static constexpr size_t blockCopySize = 1024 * 1024; // 1MiB
    std::vector<T> buffer;
    buffer.resize(blockCopySize);
    for (size_t blockStart = 0; blockStart < vector.size(); blockStart += blockCopySize) {
        const size_t blockEnd = std::min(blockStart + blockCopySize, vector.size());
        buffer.resize(blockEnd - blockStart);
        std::copy(std::begin(vector) + blockStart, std::begin(vector) + blockEnd, std::begin(buffer));
        optFile->fileStream.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(T));
    }

    assert_always(!optFile->fileStream.bad());
    offset += vector.size();
    vector.clear();
}

template <typename Attribute>
void AtomicEncodedOctree<Attribute>::moveAttributesToDiskUnsafe()
{
    if constexpr (!std::is_void_v<Attribute>)
        write(optAttributesFile, attributes, attributesOffset);
}

template <typename Attribute>
void AtomicEncodedOctree<Attribute>::moveLevelToDiskUnsafe(uint32_t level)
{
    write(filePerLevel[level], nodesPerLevel[level], offsetPerLevel[level]);
}

template struct AtomicEncodedOctree<void>;
// template struct AtomicEncodedOctree<voxcom::RGB>;

TempFile::TempFile()
{
#if WIN32
    char buffer[L_tmpnam_s];
    // Not actually part of C++ (only C11) but MSVC STL implementation will complain if you use std::tmpnam()
    tmpnam_s(buffer, L_tmpnam_s);
    const char* pFilePath = buffer;
#else
    char buffer[L_tmpnam];
    const char* pFilePath = std::tmpnam(buffer);
#endif
    filePath = pFilePath;

    spdlog::info("Creating file: {}", filePath.string());
    setMode(Mode::Write);
    assert_always(!fileStream.bad());
}

TempFile::TempFile(TempFile&& other)
    : filePath(std::move(other.filePath))
    , fileStream(std::move(other.fileStream))
{
    assert_always(!fileStream.bad());
    other.filePath = "";
}

TempFile::~TempFile()
{
    if (!filePath.empty()) {
        assert_always(!fileStream.bad());
        spdlog::info("Deleting file: {}", filePath.string());
        fileStream.close();
        std::filesystem::remove(filePath);
    }
}

void TempFile::setMode(Mode mode_)
{
    if (mode_ == this->mode)
        return;

    if (fileStream.is_open())
        fileStream.close();

    const auto iosMode = mode_ == Mode::Read ? std::ios::in : std::ios::out | std::ios::app;
    fileStream = std::fstream(filePath, std::ios::binary | iosMode);
    assert_always(fileStream.is_open());
    this->mode = mode_;
}

template struct AtomicEncodedOctree<RGB>;
template struct AtomicEncodedOctree<void>;

}