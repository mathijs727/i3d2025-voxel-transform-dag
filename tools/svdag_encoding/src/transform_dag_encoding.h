#pragma once
#include "huffman.h"
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <glm/vec3.hpp>
#include <ostream>
#include <vector>
#include <voxcom/utility/binary_reader.h>
#include <voxcom/utility/binary_writer.h>
#include <voxcom/utility/error_handling.h>
#include <voxcom/utility/size_of.h>
#include <voxcom/voxel/morton.h>
#include <voxcom/voxel/structure.h>
#include <voxcom/voxel/transform_dag.h>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <nlohmann/json_fwd.hpp>
DISABLE_WARNINGS_POP()

namespace detail {
template <typename Transform>
static constexpr bool is_complex_transform = requires(Transform state, uint32_t level, glm::uvec3 voxel, typename Transform::BaseType* pNode) {
    {
        state.hasChild(level, voxel, pNode)
    } -> std::same_as<bool>;
};
}

template <typename BaseType_>
class TransformBaseDAG {
public:
    using BaseType = BaseType_;
    std::vector<std::vector<BaseType_>> nodesPerLevel;
    std::vector<voxcom::EditSubGrid<void>> subGrids;

    static constexpr uint32_t subGridLevel = 2; // lvl0 = 1x1x1, lvl1 = 2x2x2, lvl2 = 4x4x4 subgrids
    unsigned resolution = 0;
    uint32_t rootLevel;

public:
    virtual ~TransformBaseDAG() = default;
    size_t sizeInBytes() const;

    virtual void writeTo(voxcom::BinaryWriter& writer) const;
    virtual void readFrom(voxcom::BinaryReader& reader);
};

template <typename BaseType>
size_t TransformBaseDAG<BaseType>::sizeInBytes() const
{
    size_t out = 0;
    for (const auto& levelNodes : nodesPerLevel) {
        out += voxcom::sizeOfVector(levelNodes);
    }
    out += voxcom::sizeOfVector(subGrids);
    return out;
}

template <typename BaseType_>
void TransformBaseDAG<BaseType_>::writeTo(voxcom::BinaryWriter& writer) const
{
    writer.write(nodesPerLevel);
    writer.write(subGrids);
    writer.write(resolution);
    writer.write(rootLevel);
}

template <typename BaseType_>
void TransformBaseDAG<BaseType_>::readFrom(voxcom::BinaryReader& reader)
{
    reader.read(nodesPerLevel);
    reader.read(subGrids);
    reader.read(resolution);
    reader.read(rootLevel);
}

template <typename TransformDAG>
inline void verifyComplexTransformDAG_recurse(
    const voxcom::EditStructure<void, uint32_t>& editStructure, const TransformDAG& transformDAG,
    uint32_t level, uint32_t nodeHandle, const glm::uvec3& voxel)
{
    if (level == editStructure.subGridLevel) {
        const auto& editSubGrid = editStructure.subGrids[nodeHandle];
        for (uint32_t i = 0; i < 64; ++i) {
            const glm::uvec3 subGridVoxel = voxcom::morton_decode32<3>(i);
            const glm::uvec3 leafVoxel = (voxel << 2u) + subGridVoxel;
            const bool expected = editSubGrid.get(subGridVoxel);
            const bool got2 = editStructure.get(leafVoxel);
            voxcom::assert_always(got2 == expected);
            const bool got = transformDAG.get(leafVoxel);
            if (got != expected) {
                const bool got3 = transformDAG.get(leafVoxel);
                voxcom::assert_always(got3 == got2);
            }
        }
    } else {
        const auto& editNode = editStructure.nodesPerLevel[level][nodeHandle];
        for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
            if (editNode.children[childIdx] != editNode.EmptyChild) {
                const auto childHandle = editNode.children[childIdx];
                const auto childVoxel = (voxel << 1u) + voxcom::morton_decode32<3>(childIdx);
                verifyComplexTransformDAG_recurse(editStructure, transformDAG, level - 1, childHandle, childVoxel);
            }
        }
    }
}
template <typename TransformDAG>
void verifyComplexTransformDAG(const voxcom::EditStructure<void, uint32_t>& editStructure, const TransformDAG& transformDAG)
{
    voxcom::assert_always(editStructure.rootLevel == transformDAG.rootLevel);
    voxcom::assert_always(editStructure.subGridLevel == transformDAG.subGridLevel);
    verifyComplexTransformDAG_recurse(editStructure, transformDAG, editStructure.rootLevel, 0, glm::uvec3(0));
}

template <typename T, typename BaseType>
class NoTransformBaseDAG : public TransformBaseDAG<BaseType> {
public:
    bool get(const glm::ivec3&) const;
};
class NoTransformDAG32 : public NoTransformBaseDAG<NoTransformDAG32, uint32_t> {
public:
    bool hasChild(const BaseType* node, uint32_t childIndex) const;
    uint32_t getChildHandle(const BaseType* node, uint32_t childIndex) const;
};
class NoTransformHybridDAG32 : public NoTransformBaseDAG<NoTransformHybridDAG32, uint32_t> {
public:
    uint32_t rootHandle;

public:
    bool hasChild(const BaseType* node, uint32_t childIndex) const;
    uint32_t getChildHandle(const BaseType* node, uint32_t childIndex) const;
};
class NoTransformDAG16 : public NoTransformBaseDAG<NoTransformDAG16, uint16_t> {
public:
    bool hasChild(const BaseType* node, uint32_t childIndex) const;
    uint32_t getChildHandle(const BaseType* pNode, uint32_t childIndex) const;
};
NoTransformDAG32 constructNoTransform32(const voxcom::EditStructure<void, uint32_t>& structure);
NoTransformHybridDAG32 constructNoTransformHybrid32(const voxcom::EditStructure<void, uint32_t>& structure);
NoTransformDAG16 constructNoTransform16(const voxcom::EditStructure<void, uint32_t>& structure);

template <typename T, typename BaseType>
class SSVDAGBase : public TransformBaseDAG<BaseType> {
public:
    bool get(const glm::ivec3&) const;
};
class SSVDAG32 : public SSVDAGBase<SSVDAG32, uint32_t> {
public:
    bool hasChild(const BaseType* node, uint32_t childIndex, uint32_t transform) const;
    uint32_t traverseToChild(const BaseType* pNode, uint32_t childIndex, uint32_t& transform) const;
};
class SSVDAG16 : public SSVDAGBase<SSVDAG16, uint16_t> {
public:
    bool hasChild(const BaseType* node, uint32_t childIndex, uint32_t transform) const;
    uint32_t traverseToChild(const BaseType* pNode, uint32_t childIndex, uint32_t& transform) const;
};
template <bool ExtendedInvariance>
SSVDAG32 constructSSVDAG32(const voxcom::EditStructure<void, uint32_t>& structure);
template <bool ExtendedInvariance>
SSVDAG16 constructSSVDAG16(const voxcom::EditStructure<void, uint32_t>& structure);

template <typename T, typename BaseType>
class TranslateTransformBaseDAG : public TransformBaseDAG<BaseType> {
public:
    voxcom::TransformDAGConfig constructionConfig;

public:
    bool get(glm::ivec3) const;

private:
    static uint32_t getChildIdx(uint32_t level, const glm::u8vec3& axisPermutation, const glm::ivec3& virtualLocation, const glm::ivec3& voxel);
};

struct TransformEncodingConfig {
    bool pointerTables = true, huffmanCode = true;
};
struct PartiallyEncodedTransformPointer {
    uint32_t index = 0;
    uint8_t transformID = 0; // 3 least significant bits store symmetry, other bits store axis permutation (6 different options).
    int8_t translation[3] = { 0, 0, 0 };
};
// class TransformDAG32 : public TranslateTransformBaseDAG<TransformDAG32, uint32_t> {
// public:
//     bool hasChild(const BaseType* node, uint32_t childIndex) const;
//     //void getChildPointer(uint32_t childLevel, const BaseType* pNode, uint32_t childIndex, voxcom::TransformPointer& out) const;
//     void getChildPointer(uint32_t childLevel, const BaseType* pNode, uint32_t childIndex, PartiallyEncodedTransformPointer& out) const;
// };
class TransformDAG16 : public TranslateTransformBaseDAG<TransformDAG16, uint16_t> {
public:
    bool hasChild(const BaseType* node, uint32_t childIndex) const;
    PartiallyEncodedTransformPointer getChildPointer(uint32_t childLevel, const BaseType* pNode, uint32_t childIndex) const;
    size_t sizeInBytes() const;

    virtual void writeTo(voxcom::BinaryWriter& writer) const override;
    virtual void readFrom(voxcom::BinaryReader& reader) override;

public:
    std::vector<std::vector<uint64_t>> pointerTables;
    std::vector<HuffmanDecodeLUT> huffmanDecoders;
    TransformEncodingConfig encodingConfig;
};

struct DAGEncodingStats {
    struct LevelStats {
        std::vector<size_t> transformHistogram;

        std::array<size_t, 3> numTablePointersOfSizeU16 { 0, 0, 0 }; // 16, 32, 48 bits
        std::array<size_t, 3> numDirectPointersOfSizeU16 { 0, 0, 0 }; // 16, 32, 48 bits

        std::vector<uint32_t> actualPointerBitSizeInTableHistogram;
        size_t tableSizeInPointers = 0;
        size_t pointerSizeInTable = 0;

        size_t numNodes = 0;
        size_t memoryInBytes = 0;

        operator nlohmann::json() const;
    };
    std::vector<LevelStats> levels;
    size_t totalSizeInBytes;
    voxcom::TransformDAGConfig dagConfig;
    TransformEncodingConfig encodingConfig;

    void init(uint32_t numLevels, const voxcom::TransformDAGConfig& dagConfig, const TransformEncodingConfig& encodingConfig);
    void write(nlohmann::json& out) const;
};

TransformDAG16 constructTransformDAG16(voxcom::EditStructure<void, uint32_t>&& structure, const voxcom::TransformDAGConfig& dagConfig, const TransformEncodingConfig& encodingConfig, DAGEncodingStats& stats);
TransformDAG16 constructTransformDAG16(voxcom::EditStructure<void, voxcom::TransformPointer> structure, const voxcom::TransformDAGConfig& dagConfig, const TransformEncodingConfig& encodingConfig, DAGEncodingStats& stats);

TransformDAG16 constructTransformDAG16(voxcom::StaticStructure<void, uint32_t>&& structure, const voxcom::TransformDAGConfig& dagConfig, const TransformEncodingConfig& encodingConfig, DAGEncodingStats& stats);
TransformDAG16 constructTransformDAG16(voxcom::StaticStructure<void, voxcom::TransformPointer> structure, const voxcom::TransformDAGConfig& dagConfig, const TransformEncodingConfig& encodingConfig, DAGEncodingStats& stats);
