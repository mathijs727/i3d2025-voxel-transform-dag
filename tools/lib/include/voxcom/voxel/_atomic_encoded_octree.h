#pragma once
#include <cstddef>
#include <voxcom/voxel/encoded_octree.h>
#include <voxcom/voxel/octree.h>
#pragma warning(disable : 4459) // declaration of 'relation' hides global declaration (tbb_profiling.h:253)
#include <tbb/concurrent_vector.h>
#pragma warning(default : 4459)
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <vector>

namespace voxcom {

struct TempFile {
public:
    enum class Mode {
        Read,
        Write,
        Closed
    };

public:
    TempFile();
    TempFile(TempFile&&);
    ~TempFile();

    void setMode(Mode mode);

public:
    std::filesystem::path filePath;
    std::fstream fileStream;
    Mode mode { Mode::Closed };
};

template <typename T>
struct atomic_attribute_vector {
    using type = tbb::concurrent_vector<T>;
};
template <>
struct atomic_attribute_vector<void> {
    using type = int; // Random type that doesn't occupy a lot of memory.
};
template <typename T>
using atomic_attribute_vector_t = typename atomic_attribute_vector<T>::type;

// Simple encoding of an octree to reduce memory usage. This representation can easily
// be converted back to the regular Octree<Attribute> using a cast/static_cast if required.
template <typename Attribute_>
struct AtomicEncodedOctree {
public:
    using Attribute = Attribute_;
    using Node = typename EncodedOctree<Attribute>::Node;

    std::vector<tbb::concurrent_vector<Node>> nodesPerLevel;
    std::vector<size_t> offsetPerLevel;
    atomic_attribute_vector_t<Attribute> attributes;
    size_t attributesOffset = 0;

    // At high resolutions the Octree can become very big, even with an efficient encoding scheme.
    // AtomicEncodedOctree therefor has an option to move (part) of the current octree to disk so
    // the tree can keep growing without running out of memory.
    std::vector<std::optional<TempFile>> filePerLevel;
    std::optional<TempFile> optAttributesFile;

    unsigned resolution;
    constexpr static uint32_t subGridLevel = 2;
    uint32_t rootLevel;

public:
    AtomicEncodedOctree() = default;
    AtomicEncodedOctree(const EncodedOctree<Attribute>&);
    // AtomicEncodedOctree(const EncodedOctree<void>&) requires (!std::is_void_v<Attribute>);
    //  AtomicEncodedOctree(AtomicEncodedOctree<Attribute>&&) noexcept = default;

    EncodedOctree<Attribute> toEncodedOctree();

    Node addSubtree(const Octree<Attribute>&);

    // Move all of the leaves/current nodes in the given level to disk. This operation is not thread safe!
    void moveLevelToDiskUnsafe(uint32_t level);
    void moveAttributesToDiskUnsafe();
};

}