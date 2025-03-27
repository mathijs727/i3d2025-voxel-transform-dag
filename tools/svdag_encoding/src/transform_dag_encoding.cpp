#include "transform_dag_encoding.h"
#include "huffman.h"
#include "pointer_encoding.h"
#include <algorithm> // std::iota
#include <execution>
#include <iostream>
#include <set>
#include <span>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <voxcom/utility/error_handling.h>
#include <voxcom/utility/fmt_glm.h>
#include <voxcom/utility/maths.h>
#include <voxcom/utility/size_of.h>
#include <voxcom/voxel/ssvdag.h>
#include <voxcom/voxel/transform_dag.h>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
#include <glm/vector_relational.hpp>
#include <nlohmann/json.hpp>
#include <robin_hood.h>
#include <spdlog/spdlog.h>
DISABLE_WARNINGS_POP()

using namespace voxcom;

#define ENABLE_ASSERTIONS 0

TransformDAG16 constructTransformDAG16(voxcom::EditStructure<void, uint32_t>&& structure, const TransformDAGConfig& dagConfig, const TransformEncodingConfig& encodingConfig, DAGEncodingStats& stats)
{
    // Convert input DAG into intermediate transform SVDAG.
    if (structure.structureType == StructureType::Tree)
        structure.toDAG();
    auto intermediateDAG = constructTransformDAG(std::move(structure), dagConfig);
    return constructTransformDAG16(std::move(intermediateDAG), dagConfig, encodingConfig, stats);
}

TransformDAG16 constructTransformDAG16(voxcom::StaticStructure<void, uint32_t>&& structure, const TransformDAGConfig& dagConfig, const TransformEncodingConfig& encodingConfig, DAGEncodingStats& stats)
{
    // Convert input DAG into intermediate transform SVDAG.
    StaticStructure<void, TransformPointer> intermediateDAG = constructStaticTransformDAG(std::move(structure), dagConfig);
    return constructTransformDAG16(std::move(intermediateDAG), dagConfig, encodingConfig, stats);
}

[[maybe_unused]] static void printHistogram(std::span<const size_t> histogram, const char* name)
{
    spdlog::info("===== HISTOGRAM {} =====", name);
    for (size_t i = 0; i < histogram.size(); ++i) {
        fmt::println("[{}] {}", i, histogram[i]);
    }
    fmt::println("\n");
}

DAGEncodingStats::LevelStats::operator nlohmann::json() const
{
    nlohmann::json out;
    // out["transform_histogram"] = transformHistogram;
    out["num_table_pointers_of_size_u16"] = numTablePointersOfSizeU16;
    out["num_direct_pointers_of_size_u16"] = numDirectPointersOfSizeU16;
    // out["actual_pointer_bit_size_in_table_histogram"] = std::span(actualPointerBitSizeInTableHistogram);
    out["table_size_in_pointers"] = tableSizeInPointers;
    out["pointer_size_in_table"] = pointerSizeInTable;
    out["num_nodes"] = numNodes;
    out["memory_in_bytes"] = memoryInBytes;
    return out;
}

void DAGEncodingStats::write(nlohmann::json& out) const
{
    out["levels"] = levels;
    out["total_size_in_bytes"] = totalSizeInBytes;
    out["pointer_tables"] = bool(encodingConfig.pointerTables);
    out["huffman_code"] = bool(encodingConfig.huffmanCode);
    out["symmetry"] = dagConfig.symmetry;
    out["axis_permutation"] = dagConfig.axisPermutation;
    out["enable_translation"] = dagConfig.translation;
    out["translation_level"] = dagConfig.maxTranslationLevel;
}

void DAGEncodingStats::init(uint32_t numLevels, const voxcom::TransformDAGConfig& dagConfig_, const TransformEncodingConfig& encodingConfig_)
{
    levels.resize(numLevels);
    this->dagConfig = dagConfig_;
    this->encodingConfig = encodingConfig_;
    for (uint32_t level = 0; level < numLevels; ++level) {
        const uint32_t symmetryBits = dagConfig.symmetry ? 3 : 0;
        const uint32_t axisPermutationBits = dagConfig.axisPermutation ? 3 : 0;
        const uint32_t translationBits = (dagConfig.translation && level <= dagConfig.maxTranslationLevel) ? (3 * level + 3) : 0;
        const uint32_t totalTransformBits = symmetryBits + axisPermutationBits + translationBits;
        const uint32_t totalHistogramSize = 1u << totalTransformBits;
        levels[level].transformHistogram.resize(totalHistogramSize);
        levels[level].actualPointerBitSizeInTableHistogram.resize(64, 0);
    }
}

// Encodes/decodes the type of transformation:
//  * non-zero translation along x, y, and/or z (8 options).
//  * symmetry along x, y, and/or z (8 options)
//  * axis permutation (6 options)
static TransformPointer decodePointerType(uint32_t pointerType, uint32_t childLevel, const TransformDAGConfig& dagConfig)
{
    glm::bvec3 symmetry { false, false, false };
    glm::u8vec3 axisPermutation { 0, 1, 2 };
    glm::bvec3 hasTranslation { false, false, false };
    if (dagConfig.translation && childLevel <= dagConfig.maxTranslationLevel) {
        hasTranslation = morton_decode32<3>(pointerType & 0b111);
        pointerType >>= 3;
    }
    if (dagConfig.symmetry) {
        symmetry = morton_decode32<3>(pointerType & 0b111);
        pointerType >>= 3;
    }
    if (dagConfig.axisPermutation) {
        axisPermutation = TransformPointer::decodeAxisPermutation(pointerType);
    }
    return TransformPointer::create(0, symmetry, axisPermutation, hasTranslation);
}
static uint32_t encodePointerType(const voxcom::TransformPointer& pointer, uint32_t childLevel, const TransformDAGConfig& dagConfig)
{
    uint32_t out = 0;
    if (dagConfig.axisPermutation) {
        out |= TransformPointer::encodeAxisPermutation(pointer.getAxisPermutation());
    }
    if (dagConfig.symmetry) {
        out <<= 3;
        out |= morton_encode32(glm::uvec3(pointer.getSymmetry()));
    }
    if (dagConfig.translation && childLevel <= dagConfig.maxTranslationLevel) {
        out <<= 3;
        out |= morton_encode32(glm::uvec3(glm::notEqual(pointer.getTranslation(), glm::ivec3(0))));
    }
    return out;
}

static PartiallyEncodedTransformPointer decodeTransformPointerHuffman(uint64_t encodedPointer, uint32_t childLevel, const TransformDAGConfig& dagConfig, const HuffmanDecodeLUT& huffmanDecoder)
{
    const bool decodeTranslation = dagConfig.translation && childLevel <= dagConfig.maxTranslationLevel;
    auto [pointerType, prefixLength] = huffmanDecoder.decode((uint32_t)encodedPointer);
    encodedPointer >>= prefixLength;

    uint32_t translationAxis = 0;
    if (decodeTranslation) {
        translationAxis = pointerType & 0b111;
        pointerType >>= 3;
    }

    PartiallyEncodedTransformPointer out {};
    out.translation[0] = out.translation[1] = out.translation[2] = 0;
    if (translationAxis) {
        const int8_t offset = 1 << childLevel;
        const uint32_t numTranslationBits = childLevel + 1;
        const uint64_t translationMask = (1llu << numTranslationBits) - 1llu;
        if (translationAxis & 0b100) {
            out.translation[2] = int8_t(encodedPointer & translationMask) - offset;
            encodedPointer >>= numTranslationBits;
        }
        if (translationAxis & 0b010) {
            out.translation[1] = int8_t(encodedPointer & translationMask) - offset;
            encodedPointer >>= numTranslationBits;
        }
        if (translationAxis & 0b001) {
            out.translation[0] = int8_t(encodedPointer & translationMask) - offset;
            encodedPointer >>= numTranslationBits;
        }
    }

    out.transformID = 0;
    if (dagConfig.symmetry) {
        out.transformID |= pointerType & 0b111;
        pointerType >>= 3;
    }
    if (dagConfig.axisPermutation) {
        out.transformID |= (pointerType & 0b111) << 3;
        pointerType >>= 3;
    }
    out.index = (uint32_t)encodedPointer;
    return out;
}
static PartiallyEncodedTransformPointer decodeTransformPointerWithoutHuffman(uint64_t encodedPointer, uint32_t childLevel, const TransformDAGConfig& dagConfig)
{
    PartiallyEncodedTransformPointer out;
    bool hasTransform = false;
    if (dagConfig.axisPermutation || dagConfig.symmetry || dagConfig.translation) {
        hasTransform = encodedPointer & 0b1;
        encodedPointer >>= 1;
    }
    if (hasTransform) {
        if (dagConfig.translation && childLevel <= dagConfig.maxTranslationLevel) {
            const uint32_t translationBits = childLevel + 1;
            const uint64_t translationMask = (1llu << translationBits) - 1llu;
            const int8_t offset = 1 << childLevel;

            out.translation[0] = int8_t(encodedPointer & translationMask) - offset;
            encodedPointer >>= translationBits;
            out.translation[1] = int8_t(encodedPointer & translationMask) - offset;
            encodedPointer >>= translationBits;
            out.translation[2] = int8_t(encodedPointer & translationMask) - offset;
            encodedPointer >>= translationBits;
        } else {
            out.translation[0] = out.translation[1] = out.translation[2] = 0;
        }

        out.transformID = 0;
        if (dagConfig.symmetry) {
            out.transformID |= encodedPointer & 0b111;
            encodedPointer >>= 3;
        }
        if (dagConfig.axisPermutation) {
            out.transformID |= (encodedPointer & 0b111) << 3;
            encodedPointer >>= 3;
        }
        assert_always(out.transformID < 48);
    }
    out.index = (uint32_t)encodedPointer;
    return out;
}

// https://stackoverflow.com/questions/7818371/printing-to-nowhere-with-ostream
struct NulStreambuf : public std::streambuf {
    size_t sizeInBytes = 0;

protected:
    virtual int overflow(int c)
    {
        sizeInBytes += 1;
        return c;
    }
};
class NulOStream : public NulStreambuf, public std::ostream {
public:
    NulOStream()
        : std::ostream(this)
    {
    }
};

TransformDAG16 constructTransformDAG16(voxcom::EditStructure<void, voxcom::TransformPointer> intermediateDAG, const voxcom::TransformDAGConfig& dagConfig, const TransformEncodingConfig& encodingConfig, DAGEncodingStats& stats)
{
    stats.init((uint32_t)intermediateDAG.nodesPerLevel.size(), dagConfig, encodingConfig);

    // Sort nodes by reference count to reduce pointer size.
    // By performing this process from top-to-bottom we can also eliminate nodes which are never referenced.
    for (uint32_t level = intermediateDAG.rootLevel - 1; level > intermediateDAG.subGridLevel; --level) {
        auto& levelNodes = intermediateDAG.nodesPerLevel[level];
        auto& parentNodes = intermediateDAG.nodesPerLevel[level + 1];
        levelNodes = sortByReferenceCount<EditNode<TransformPointer>, TransformPointer>(levelNodes, parentNodes, true);
    }
    intermediateDAG.subGrids = sortByReferenceCount<EditSubGrid<void>, TransformPointer>(
        intermediateDAG.subGrids, intermediateDAG.nodesPerLevel[intermediateDAG.subGridLevel + 1], true);

    // Convert from intermediate DAG to final output.
    TransformDAG16 out {};
    out.constructionConfig = dagConfig;
    out.encodingConfig = encodingConfig;
    out.subGrids = std::move(intermediateDAG.subGrids);
    out.resolution = intermediateDAG.resolution;
    out.rootLevel = intermediateDAG.rootLevel;
    out.nodesPerLevel.resize(intermediateDAG.nodesPerLevel.size());
    out.pointerTables.resize(out.nodesPerLevel.size());
    if (encodingConfig.huffmanCode)
        out.huffmanDecoders.resize(out.nodesPerLevel.size());
    spdlog::info("[{}] {} KiB ({} leaves)", out.subGridLevel, (out.subGrids.size() * sizeof(EditSubGrid<void>)) >> 10, out.subGrids.size());

    stats.levels[out.subGridLevel].numNodes = out.subGrids.size();
    stats.levels[out.subGridLevel].memoryInBytes = sizeOfVector(out.subGrids);

    std::vector<size_t> prevLevelMapping(out.subGrids.size());
    std::iota(std::begin(prevLevelMapping), std::end(prevLevelMapping), 0);
    for (uint32_t level = out.subGridLevel + 1; level <= out.rootLevel; ++level) {
        const auto childLevel = level - 1;
        auto& levelStats = stats.levels[level];
        auto& inLevelNodes = intermediateDAG.nodesPerLevel[level];
        auto& outLevelNodes = out.nodesPerLevel[level];

        std::vector<std::pair<uint32_t, TransformPointer>> childPointerRefCounts;
        {
            // Update child pointers
            std::unordered_map<TransformPointer, uint32_t> childPointerRefCountsLUT;
            for (auto& node : inLevelNodes) {
                for (auto& child : node.children) {
                    if (child != TransformPointer::sentinel()) {
                        child.ptr = prevLevelMapping[child.ptr];
                        childPointerRefCountsLUT[child]++;
                    }
                }
            }

            // Sort children based on how often they are referenced.
            for (const auto& [pointer, refCount] : childPointerRefCountsLUT)
                childPointerRefCounts.push_back({ refCount, pointer });
            std::sort(std::execution::par, std::begin(childPointerRefCounts), std::end(childPointerRefCounts), [](const auto& lhs, const auto& rhs) { return std::get<0>(lhs) > std::get<0>(rhs); });
        }

        std::vector<Code> huffmanCodes;
        if (encodingConfig.huffmanCode) {
            size_t numPointerTypes = 1;
            if (dagConfig.symmetry)
                numPointerTypes *= 8;
            if (dagConfig.axisPermutation)
                numPointerTypes *= 6;
            if (dagConfig.translation)
                numPointerTypes *= 8;
            assert_always(numPointerTypes >= 1); // In this case Huffman code will always add 1 bit of overhead. Instead run program with Huffman disabled.
            std::vector<size_t> transformationTypeHistogram(numPointerTypes);
            std::fill(std::begin(transformationTypeHistogram), std::end(transformationTypeHistogram), 0);

            for (const auto& [refCount, childPointer] : childPointerRefCounts) {
                const auto pointerType = encodePointerType(childPointer, childLevel, dagConfig);
                transformationTypeHistogram[pointerType]++;
            }

            huffmanCodes = createLengthLimitedHuffmanCodeTable(transformationTypeHistogram, 16);
            convertHuffmanToLSB(huffmanCodes);
            out.huffmanDecoders[childLevel] = HuffmanDecodeLUT(huffmanCodes);
        }

        const auto encodePointer = [&](const TransformPointer& transformPointer) {
            if (encodingConfig.huffmanCode) {
                const auto pointerType = encodePointerType(transformPointer, childLevel, dagConfig);
                const auto pointerTypeCode = huffmanCodes[pointerType];

                uint64_t encodedTransformPointer = transformPointer.ptr;

                // Offset the translation so that it's always positive (>= 0).
                const auto translation = transformPointer.getTranslation();
                const glm::uvec3 offsetTranslation = glm::uvec3(translation + glm::ivec3(1 << childLevel));
                const auto translationBits = childLevel + 1;

                // Add translation along x/y/z axis if non-zero.
                if (translation.x) {
                    encodedTransformPointer <<= translationBits;
                    encodedTransformPointer |= offsetTranslation.x;
                }
                if (translation.y) {
                    encodedTransformPointer <<= translationBits;
                    encodedTransformPointer |= offsetTranslation.y;
                }
                if (translation.z) {
                    encodedTransformPointer <<= translationBits;
                    encodedTransformPointer |= offsetTranslation.z;
                }

                // Encode the pointer type (symmetry + axis permutation + non-zero translation axis).
                encodedTransformPointer <<= pointerTypeCode.bits;
                encodedTransformPointer |= pointerTypeCode.code;
                return encodedTransformPointer;
            } else {
                // Pointer and symmetry.
                const bool hasTransform = transformPointer.hasTransform();
                uint64_t out = transformPointer.ptr;
                if (dagConfig.axisPermutation && hasTransform) {
                    out <<= 3;
                    out |= TransformPointer::encodeAxisPermutation(transformPointer.getAxisPermutation());
                }
                if (dagConfig.symmetry && hasTransform) {
                    out <<= 3;
                    out |= bvec3ToU64(transformPointer.getSymmetry());
                }
                const uint32_t translationBits = childLevel + 1;
                if (dagConfig.translation && childLevel <= dagConfig.maxTranslationLevel && hasTransform) {
                    const glm::uvec3 offsetShift = glm::uvec3(transformPointer.getTranslation() + glm::ivec3(1 << childLevel));
                    out <<= translationBits;
                    out |= offsetShift.z;
                    out <<= translationBits;
                    out |= offsetShift.y;
                    out <<= translationBits;
                    out |= offsetShift.x;
                }

                // Prefix pointer with 0b0 or 0b1 depending on whether it contains any type of transformation...
                if (dagConfig.axisPermutation || dagConfig.symmetry || dagConfig.translation) {
                    out <<= 1;
                    out |= transformPointer.hasTransform() ? 0b1 : 0b0;
                }
                return out;
            }
        };

        std::unordered_map<TransformPointer, std::pair<uint64_t, uint32_t>> childPointersLUT;
        [[maybe_unused]] auto& pointerTable = out.pointerTables[childLevel];
        for (const auto& [refCount, childPointer] : childPointerRefCounts) {
            uint64_t encodedPointer = encodePointer(childPointer);

            if (encodingConfig.pointerTables) {
                uint32_t inlinePointerSizeInU16, tablePointerSizeInU16;
                uint64_t encodedInlinePointer, encodedTablePointer;
                // Stored in at least 32-bits, where the first bit indicates:
                // 1 pointer into table (1) or direct (0)
                encodedInlinePointer = (encodedPointer << 1) | 0b0;
                // 16-bit pointers are reserved for indices into the table.
                inlinePointerSizeInU16 = std::max(2u, divideRoundUp((uint32_t)std::bit_width(encodedInlinePointer), 16u));

                // Detect the switch-over point at which it is cheaper to store pointers inline vs in a table.
                // While more items are added the reference count goes down (inlineCost goes down) and table index size goes up (tableCost goes up).
                constexpr auto tableEntrySizeInU16 = 4;
                uint32_t inlineCost = refCount * inlinePointerSizeInU16;
                // Pointers that cannot fit in U48 MUST be stored in a table (which uses U64 pointers).
                if (inlinePointerSizeInU16 > 3)
                    inlineCost = std::numeric_limits<uint32_t>::max();
                if (pointerTable.size() <= std::numeric_limits<uint16_t>::max()) {
                    encodedTablePointer = pointerTable.size();
                    tablePointerSizeInU16 = 1;
                } else {
                    // Stored in at least 32-bits, where the first bit indicates:
                    // 1 pointer into table (1) or direct (0)
                    encodedTablePointer = (static_cast<uint64_t>(pointerTable.size()) << 1) | 0b1;
                    tablePointerSizeInU16 = divideRoundUp((uint32_t)std::bit_width(pointerTable.size()), 16u);
                }
                const auto tableCost = refCount * tablePointerSizeInU16 + tableEntrySizeInU16;
                if (tableCost < inlineCost) {
                    // Store encoded pointer in table; reference to table index.
                    pointerTable.push_back(encodedPointer);
                    levelStats.numTablePointersOfSizeU16[tablePointerSizeInU16 - 1]++;
                    levelStats.actualPointerBitSizeInTableHistogram[std::bit_width(encodedTablePointer)]++;
                    childPointersLUT[childPointer] = { encodedTablePointer, tablePointerSizeInU16 };
                } else {
                    // Store encoded pointer directly inside DAG.
                    levelStats.numDirectPointersOfSizeU16[inlinePointerSizeInU16 - 1]++;
                    childPointersLUT[childPointer] = { encodedInlinePointer, inlinePointerSizeInU16 };
                }
            } else { // encodingConfig.pointerTables
                const uint32_t pointerSizeInU16 = std::max(1u, divideRoundUp((uint32_t)std::bit_width(encodedPointer), 16u));
                assert_always(pointerSizeInU16 <= 3);
                childPointersLUT[childPointer] = { encodedPointer, pointerSizeInU16 };
                levelStats.numDirectPointersOfSizeU16[pointerSizeInU16 - 1]++;
            } // encodingConfig.pointerTables
        }

        if (encodingConfig.pointerTables) {
            levelStats.pointerSizeInTable = sizeof(uint64_t);
            levelStats.tableSizeInPointers = pointerTable.size();
        } else {
            levelStats.pointerSizeInTable = 0;
        }

        // Write nodes
        prevLevelMapping.clear();
        for (auto& node : inLevelNodes) {
            prevLevelMapping.push_back(outLevelNodes.size());

            // Encode node.
            const size_t childMaskPtr = push<uint16_t, uint16_t>(outLevelNodes, 0);
            uint16_t childMask = 0;
            for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
                const auto transformPointer = node.children[childIdx];
                if (transformPointer == TransformPointer::sentinel())
                    continue;

                const auto [pointer, pointerSizeInU16] = childPointersLUT.find(transformPointer)->second;
                if (pointerSizeInU16 == 1) {
                    push<uint16_t, uint16_t>(outLevelNodes, (uint16_t)pointer);
                    childMask |= 0b01 << (2 * childIdx);
                } else if (pointerSizeInU16 == 2) {
                    push<uint16_t, uint32_t>(outLevelNodes, (uint32_t)pointer);
                    childMask |= 0b10 << (2 * childIdx);
                } else if (pointerSizeInU16 <= 3) {
                    push<uint16_t, uint64_t>(outLevelNodes, (uint64_t)pointer, 3);
                    childMask |= 0b11 << (2 * childIdx);
                } else {
                    spdlog::info("pointer = {}; size = {}", pointer, pointerSizeInU16);
                    assert_always(false);
                }
            }
            outLevelNodes[childMaskPtr] = childMask;
            for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
                const auto transformPointer = node.children[childIdx];
                if (transformPointer == TransformPointer::sentinel())
                    continue;
                const auto ptr = out.getChildPointer(level, &outLevelNodes[childMaskPtr], childIdx);
                assert_always(ptr.index == transformPointer.ptr);
            }
        }
        levelStats.numNodes = inLevelNodes.size();
        levelStats.memoryInBytes = sizeOfVector(outLevelNodes);
        spdlog::info("[{}] {} KiB for {} nodes ({:.2f} bytes / node)", level, levelStats.memoryInBytes >> 10, inLevelNodes.size(), (double)levelStats.memoryInBytes / inLevelNodes.size());
    }

#if 0
    size_t pointersInPointerTable = 0, pointerTableWastedBits = 0;
    std::array<size_t, 64> pointerTableSizeHistogram;
    std::fill(std::begin(pointerTableSizeHistogram), std::end(pointerTableSizeHistogram), 0);
    for (size_t level = 0; level < out.pointerTables.size(); ++level) {
        for (const uint64_t pointer : out.pointerTables[level]) {
            const auto pointerSizeInBits = std::bit_width(pointer);
            ++pointerTableSizeHistogram[pointerSizeInBits];
            pointerTableWastedBits += 64 - pointerSizeInBits;
        }
        pointersInPointerTable += out.pointerTables[level].size();
    }
    spdlog::info("Pointer table contains {} entries ({} KiB)", pointersInPointerTable, pointersInPointerTable * sizeof(uint64_t));
    spdlog::info("Pointer table wasted {} KiB on zero bits", (pointerTableWastedBits / 8) >> 10);
    spdlog::info(" === POINTER TABLE HISTOGRAM ===");
    for (size_t i = 0; i < pointerTableSizeHistogram.size(); ++i) {
        const auto numPointersOfSize = pointerTableSizeHistogram[i];
        spdlog::info("[{}] {} pointers ({} bits; {} wasted bytes)", i, numPointersOfSize, numPointersOfSize * i, numPointersOfSize * (64 - i));
    }
#endif

    {
        NulOStream nulStream {};
        BinaryWriter writer { nulStream };
        out.writeTo(writer);
        stats.totalSizeInBytes = nulStream.sizeInBytes;
    }

    if constexpr (false) {
        nlohmann::json statsJSON;
        stats.write(statsJSON);
        for (auto& levelStatsJSON : statsJSON["levels"])
            levelStatsJSON.erase("transform_histogram");
        std::cout << std::setfill(' ') << std::setw(4) << statsJSON << std::endl;
    }

    return out;
}

TransformDAG16 constructTransformDAG16(voxcom::StaticStructure<void, voxcom::TransformPointer> intermediateDAG, const voxcom::TransformDAGConfig& dagConfig, const TransformEncodingConfig& encodingConfig, DAGEncodingStats& stats)
{
    stats.init((uint32_t)intermediateDAG.nodesPerLevel.size(), dagConfig, encodingConfig);

    // Sort nodes by reference count to reduce pointer size.
    // By performing this process from top-to-bottom we can also eliminate nodes which are never referenced.
    for (uint32_t level = intermediateDAG.rootLevel - 1; level >= intermediateDAG.subGridLevel; --level) {
        sortStaticStructureLevelByReferenceCount<void, voxcom::TransformPointer>(intermediateDAG, level, true);
    }

    // Convert from intermediate DAG to final output.
    TransformDAG16 out {};
    out.constructionConfig = dagConfig;
    out.encodingConfig = encodingConfig;
    out.subGrids = std::move(intermediateDAG.subGrids);
    out.resolution = intermediateDAG.resolution;
    out.rootLevel = intermediateDAG.rootLevel;
    out.nodesPerLevel.resize(intermediateDAG.nodesPerLevel.size());
    out.pointerTables.resize(out.nodesPerLevel.size());
    if (encodingConfig.huffmanCode)
        out.huffmanDecoders.resize(out.nodesPerLevel.size());
    spdlog::info("[{}] {} KiB ({} leaves)", out.subGridLevel, (out.subGrids.size() * sizeof(EditSubGrid<void>)) >> 10, out.subGrids.size());

    std::unordered_map<uint32_t, uint32_t> prevLevelMapping;
    for (uint32_t i = 0; i < out.subGrids.size(); ++i)
        prevLevelMapping[i] = i;
    for (uint32_t level = out.subGridLevel + 1; level <= out.rootLevel; ++level) {
        const auto childLevel = level - 1;
        auto& levelStats = stats.levels[childLevel];
        auto& inLevelNodes = intermediateDAG.nodesPerLevel[level];
        auto& outLevelNodes = out.nodesPerLevel[level];

        std::vector<std::pair<uint32_t, TransformPointer>> childPointerRefCounts;
        {
            // Update child pointers
            std::unordered_map<TransformPointer, uint32_t> childPointerRefCountsLUT;
            for (size_t cursor = 0; cursor < inLevelNodes.size();) {
                voxcom::MutStaticNode<TransformPointer> node { &inLevelNodes[cursor] };
                for (uint32_t childOffset = 0; childOffset < node.getNumChildren(); ++childOffset) {
                    auto childPointer = node.getChildPointerAtOffset(childOffset);
                    childPointer.ptr = prevLevelMapping.find((uint32_t)childPointer.ptr)->second;
                    node.setChildPointerAtOffset(childOffset, childPointer);
                    childPointerRefCountsLUT[childPointer]++;
                }
                cursor += node.getSizeInBasicType();
            }

            // Sort children based on how often they are referenced.
            for (const auto& [pointer, refCount] : childPointerRefCountsLUT)
                childPointerRefCounts.push_back({ refCount, pointer });
            std::sort(std::execution::par, std::begin(childPointerRefCounts), std::end(childPointerRefCounts), [](const auto& lhs, const auto& rhs) { return std::get<0>(lhs) > std::get<0>(rhs); });
        }

        std::vector<Code> huffmanCodes;
        if (encodingConfig.huffmanCode) {
            size_t numPointerTypes = 1;
            if (dagConfig.symmetry)
                numPointerTypes *= 8;
            if (dagConfig.axisPermutation)
                numPointerTypes *= 6;
            if (dagConfig.translation)
                numPointerTypes *= 8;
            assert_always(numPointerTypes >= 1); // In this case Huffman code will always add 1 bit of overhead. Instead run program with Huffman disabled.
            std::vector<size_t> transformationTypeHistogram(numPointerTypes);
            std::fill(std::begin(transformationTypeHistogram), std::end(transformationTypeHistogram), 0);

            for (const auto& [refCount, childPointer] : childPointerRefCounts) {
                const auto pointerType = encodePointerType(childPointer, childLevel, dagConfig);
                transformationTypeHistogram[pointerType]++;
            }

            huffmanCodes = createLengthLimitedHuffmanCodeTable(transformationTypeHistogram, 16);
            convertHuffmanToLSB(huffmanCodes);
            out.huffmanDecoders[childLevel] = HuffmanDecodeLUT(huffmanCodes);
        }

        const auto encodePointer = [&](const TransformPointer& transformPointer) {
            if (encodingConfig.huffmanCode) {
                const auto pointerType = encodePointerType(transformPointer, childLevel, dagConfig);
                const auto pointerTypeCode = huffmanCodes[pointerType];

                uint64_t encodedTransformPointer = transformPointer.ptr;

                // Offset the translation so that it's always positive (>= 0).
                const auto translation = transformPointer.getTranslation();
                const glm::uvec3 offsetTranslation = glm::uvec3(translation + glm::ivec3(1 << childLevel));
                const auto translationBits = childLevel + 1;

                // Add translation along x/y/z axis if non-zero.
                if (translation.x) {
                    encodedTransformPointer <<= translationBits;
                    encodedTransformPointer |= offsetTranslation.x;
                }
                if (translation.y) {
                    encodedTransformPointer <<= translationBits;
                    encodedTransformPointer |= offsetTranslation.y;
                }
                if (translation.z) {
                    encodedTransformPointer <<= translationBits;
                    encodedTransformPointer |= offsetTranslation.z;
                }

                // Encode the pointer type (symmetry + axis permutation + non-zero translation axis).
                encodedTransformPointer <<= pointerTypeCode.bits;
                encodedTransformPointer |= pointerTypeCode.code;
                return encodedTransformPointer;
            } else {
                // Pointer and symmetry.
                const bool hasTransform = transformPointer.hasTransform();
                uint64_t out = transformPointer.ptr;
                if (dagConfig.axisPermutation && hasTransform) {
                    out <<= 3;
                    out |= TransformPointer::encodeAxisPermutation(transformPointer.getAxisPermutation());
                }
                if (dagConfig.symmetry && hasTransform) {
                    out <<= 3;
                    out |= bvec3ToU64(transformPointer.getSymmetry());
                }
                const uint32_t translationBits = childLevel + 1;
                if (dagConfig.translation && childLevel <= dagConfig.maxTranslationLevel && hasTransform) {
                    const glm::uvec3 offsetShift = glm::uvec3(transformPointer.getTranslation() + glm::ivec3(1 << childLevel));
                    out <<= translationBits;
                    out |= offsetShift.z;
                    out <<= translationBits;
                    out |= offsetShift.y;
                    out <<= translationBits;
                    out |= offsetShift.x;
                }

                // Prefix pointer with 0b0 or 0b1 depending on whether it contains any type of transformation...
                if (dagConfig.axisPermutation || dagConfig.symmetry || dagConfig.translation) {
                    out <<= 1;
                    out |= transformPointer.hasTransform() ? 0b1 : 0b0;
                }
                return out;
            }
        };

        std::unordered_map<TransformPointer, std::pair<uint64_t, uint32_t>> childPointersLUT;
        [[maybe_unused]] auto& pointerTable = out.pointerTables[childLevel];
        for (const auto& [refCount, childPointer] : childPointerRefCounts) {
            uint64_t encodedPointer = encodePointer(childPointer);

            if (encodingConfig.pointerTables) {
                uint32_t inlinePointerSizeInU16, tablePointerSizeInU16;
                uint64_t encodedInlinePointer, encodedTablePointer;
                // Stored in at least 32-bits, where the first bit indicates:
                // 1 pointer into table (1) or direct (0)
                encodedInlinePointer = (encodedPointer << 1) | 0b0;
                // 16-bit pointers are reserved for indices into the table.
                inlinePointerSizeInU16 = std::max(2u, divideRoundUp((uint32_t)std::bit_width(encodedInlinePointer), 16u));

                // Detect the switch-over point at which it is cheaper to store pointers inline vs in a table.
                // While more items are added the reference count goes down (inlineCost goes down) and table index size goes up (tableCost goes up).
                constexpr auto tableEntrySizeInU16 = 4;
                uint32_t inlineCost = refCount * inlinePointerSizeInU16;
                // Pointers that cannot fit in U48 MUST be stored in a table (which uses U64 pointers).
                if (inlinePointerSizeInU16 > 3)
                    inlineCost = std::numeric_limits<uint32_t>::max();
                if (pointerTable.size() <= std::numeric_limits<uint16_t>::max()) {
                    encodedTablePointer = pointerTable.size();
                    tablePointerSizeInU16 = 1;
                } else {
                    // Stored in at least 32-bits, where the first bit indicates:
                    // 1 pointer into table (1) or direct (0)
                    encodedTablePointer = (static_cast<uint64_t>(pointerTable.size()) << 1) | 0b1;
                    tablePointerSizeInU16 = divideRoundUp((uint32_t)std::bit_width(pointerTable.size()), 16u);
                }
                const auto tableCost = refCount * tablePointerSizeInU16 + tableEntrySizeInU16;
                if (tableCost < inlineCost) {
                    // Store encoded pointer in table; reference to table index.
                    pointerTable.push_back(encodedPointer);
                    levelStats.numTablePointersOfSizeU16[tablePointerSizeInU16 - 1]++;
                    levelStats.actualPointerBitSizeInTableHistogram[std::bit_width(encodedTablePointer)]++;
                    childPointersLUT[childPointer] = { encodedTablePointer, tablePointerSizeInU16 };
                } else {
                    // Store encoded pointer directly inside DAG.
                    levelStats.numDirectPointersOfSizeU16[inlinePointerSizeInU16 - 1]++;
                    childPointersLUT[childPointer] = { encodedInlinePointer, inlinePointerSizeInU16 };
                }
            } else { // encodingConfig.pointerTables
                const uint32_t pointerSizeInU16 = std::max(1u, divideRoundUp((uint32_t)std::bit_width(encodedPointer), 16u));
                assert_always(pointerSizeInU16 <= 3);
                childPointersLUT[childPointer] = { encodedPointer, pointerSizeInU16 };
                levelStats.numDirectPointersOfSizeU16[pointerSizeInU16 - 1]++;
            } // encodingConfig.pointerTables
        }

        if (encodingConfig.pointerTables) {
            levelStats.pointerSizeInTable = sizeof(uint64_t);
            levelStats.tableSizeInPointers = pointerTable.size();
        } else {
            levelStats.pointerSizeInTable = 0;
        }

        // Write nodes
        prevLevelMapping.clear();
        for (uint32_t cursor = 0; cursor < inLevelNodes.size();) {
            prevLevelMapping[cursor] = (uint32_t)outLevelNodes.size();
            const StaticNode<TransformPointer> inNode { &inLevelNodes[cursor] };

            // Encode node.
            const size_t childMaskPtr = push<uint16_t, uint16_t>(outLevelNodes, 0);
            uint16_t childMask = 0;
            for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
                if (!inNode.hasChildAtIndex(childIdx))
                    continue;
                const auto transformPointer = inNode.getChildPointerAtIndex(childIdx);

                const auto [pointer, pointerSizeInU16] = childPointersLUT.find(transformPointer)->second;
                if (pointerSizeInU16 == 1) {
                    push<uint16_t, uint16_t>(outLevelNodes, (uint16_t)pointer);
                    childMask |= 0b01 << (2 * childIdx);
                } else if (pointerSizeInU16 == 2) {
                    push<uint16_t, uint32_t>(outLevelNodes, (uint32_t)pointer);
                    childMask |= 0b10 << (2 * childIdx);
                } else if (pointerSizeInU16 <= 3) {
                    push<uint16_t, uint64_t>(outLevelNodes, (uint64_t)pointer, 3);
                    childMask |= 0b11 << (2 * childIdx);
                } else {
                    spdlog::info("pointer = {}; size = {}", pointer, pointerSizeInU16);
                    assert_always(false);
                }
            }
            outLevelNodes[childMaskPtr] = childMask;

            cursor += inNode.getSizeInBasicType();
        }
        levelStats.numNodes = inLevelNodes.size();
        levelStats.memoryInBytes = outLevelNodes.size() * sizeof(uint16_t);
        spdlog::info("[{}] {} KiB for {} nodes ({:.2f} bytes / node)", level, levelStats.memoryInBytes >> 10, inLevelNodes.size(), (double)levelStats.memoryInBytes / inLevelNodes.size());
    }

    {
        NulOStream nulStream {};
        BinaryWriter writer { nulStream };
        out.writeTo(writer);
        stats.totalSizeInBytes = nulStream.sizeInBytes;
    }

    if constexpr (false) {
        nlohmann::json statsJSON;
        stats.write(statsJSON);
        for (auto& levelStatsJSON : statsJSON["levels"])
            levelStatsJSON.erase("transform_histogram");
        std::cout << std::setfill(' ') << std::setw(4) << statsJSON << std::endl;
    }

    return out;
}

bool TransformDAG16::hasChild(const BaseType* pNode, uint32_t childIndex) const
{
    return (pNode[0] >> (2 * childIndex)) & 0b11;
}

PartiallyEncodedTransformPointer TransformDAG16::getChildPointer(uint32_t level, const BaseType* pNode, uint32_t childIndex) const
{
    // Find start of pointer for the given childIndex.
    const uint32_t childLevel = level - 1;
    const uint16_t childMask = pNode[0];
    const BaseType* pPointer = pNode + 1; // First child pointer.
    for (uint32_t i = 0; i < childIndex; ++i) {
        const uint16_t ptrBits = (childMask >> (2 * i)) & 0b11;
        if (ptrBits == 0b01)
            pPointer += 1;
        else if (ptrBits == 0b10)
            pPointer += 2;
        else if (ptrBits == 0b11)
            pPointer += ptrBits;
    }

    const uint16_t ptrBits = (childMask >> (2 * childIndex)) & 0b11;
    assert_always(ptrBits != 0b00);

    // Read (up to) the 64-bit pointer.
    uint64_t encodedPointer = 0;
    std::memcpy(&encodedPointer, pPointer, ptrBits * sizeof(uint16_t));

    if (encodingConfig.pointerTables) {
        const auto isTablePointer = (ptrBits == 0b01 || (encodedPointer & 0b1));
        if (isTablePointer) {
            const auto& pointerTable = pointerTables[childLevel];
            // 16-bit pointer is always an index into the table. Otherwise, first (of two) bits indicates whether it's a table or not.
            const auto tableIndex = (ptrBits == 0b01) ? encodedPointer : (encodedPointer >> 1);
            encodedPointer = pointerTable[tableIndex];
        } else {
            encodedPointer = (encodedPointer >> 1);
        }
    } // encodingConfig.pointerTables

    if (encodingConfig.huffmanCode) {
        return decodeTransformPointerHuffman(encodedPointer, childLevel, constructionConfig, huffmanDecoders[childLevel]);
    } else {
        return decodeTransformPointerWithoutHuffman(encodedPointer, childLevel, constructionConfig);
    }
}

using NodeTransform = std::array<uint8_t, 8>;
static uint32_t applyTransformToBitMask2x2x2(uint32_t bitmask2x2x2, const NodeTransform& transform)
{
    uint32_t outBitMask2x2x2 = 0;
    for (uint32_t outVoxelIdx = 0; outVoxelIdx < 8; ++outVoxelIdx) {
        const auto inVoxelIdx = transform[outVoxelIdx];
        if ((bitmask2x2x2 >> inVoxelIdx) & 0b1)
            outBitMask2x2x2 |= 1u << outVoxelIdx;
    }
    return outBitMask2x2x2;
}
static uint64_t applyTransformToBitMask4x4x4(uint64_t bitmask4x4x4, NodeTransform& transform)
{
    uint64_t out = 0;
    for (uint32_t outChildIdx = 0; outChildIdx < 8; ++outChildIdx) {
        const uint32_t inChildIdx = transform[outChildIdx];
        const uint32_t bitmask2x2x2 = (bitmask4x4x4 >> (8 * inChildIdx)) & 0xFF;
        const uint64_t transformedBitmask2x2x2 = applyTransformToBitMask2x2x2(bitmask2x2x2, transform);
        out |= transformedBitmask2x2x2 << (outChildIdx * 8);
    }
    assert_always(std::popcount(out) == std::popcount(bitmask4x4x4));
    return out;
}
static NodeTransform applyTransform(const NodeTransform& lhs, const NodeTransform& rhs)
{
    NodeTransform out {};
    for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
        out[childIdx] = rhs[lhs[childIdx]];
    }
    return out;
}

static std::vector<TransformPointer> generateAllTransforms()
{
    std::vector<TransformPointer> out;
    for (uint32_t i = 0; i < 48; ++i) {
        const auto symmetry = morton_decode32<3>(i & 0b111);
        const auto axisPermutation = TransformPointer::decodeAxisPermutation(i >> 3);
        out.push_back(TransformPointer::create(0, symmetry, axisPermutation));
    }
    return out;
}
static std::vector<NodeTransform> createNodeTransforms(std::span<const TransformPointer> transformPointers)
{
    std::vector<NodeTransform> out;
    for (const auto& transformPointer : transformPointers) {
        const glm::uvec3 symmetry = transformPointer.getSymmetry();
        const auto axisPermutation = transformPointer.getAxisPermutation();

        NodeTransform transform;
        for (uint8_t inChildIdx = 0; inChildIdx < 8; ++inChildIdx) {
            const auto inChildPos = morton_decode32<3>(inChildIdx);
            const auto outChildPos = applyAxisPermutation(inChildPos, axisPermutation) ^ symmetry;
            const auto outChildIdx = morton_encode32(outChildPos);
            transform[outChildIdx] = inChildIdx;
        }
        out.push_back(transform);
    }
    return out;
}
static std::vector<uint8_t> createNodeTransformCombineTable(std::span<const NodeTransform> nodeTransforms)
{
    std::vector<uint8_t> table(nodeTransforms.size() * nodeTransforms.size());
    for (size_t lhs = 0; lhs < nodeTransforms.size(); ++lhs) {
        for (size_t rhs = 0; rhs < nodeTransforms.size(); ++rhs) {
            const auto result = applyTransform(nodeTransforms[lhs], nodeTransforms[rhs]);
            const auto resultIdx = std::distance(std::begin(nodeTransforms), std::find(std::begin(nodeTransforms), std::end(nodeTransforms), result));
            table[lhs * nodeTransforms.size() + rhs] = (uint8_t)resultIdx;
        }
    }
    return table;
}

static auto allTransformPointers = generateAllTransforms();
static auto allTransforms = createNodeTransforms(allTransformPointers);
static auto transformCombineTable = createNodeTransformCombineTable(allTransforms);

#if 1
template <typename T, typename BaseType>
bool TranslateTransformBaseDAG<T, BaseType>::get(glm::ivec3 voxel) const
{
    const T* pThis = (const T*)this;

    const auto ogVoxel = voxel;
    uint32_t transformIdx = 0;
    uint32_t nodeIdx = 0;
    for (uint32_t level = this->rootLevel; level > this->subGridLevel; --level) {
        const uint32_t childLevel = level - 1u;
        const BaseType* pNode = &this->nodesPerLevel[level][nodeIdx];
        uint32_t childIndex = morton_encode32(glm::uvec3(voxel) >> childLevel) & 0b111;
        childIndex = allTransforms[transformIdx][childIndex];

        const uint32_t childBits = (pNode[0] >> (2 * childIndex)) & 0b11;
        if (!childBits)
            return false;

        // This solution allows us to forward declare TransformPointer in the header file.
        auto transformPointer = pThis->getChildPointer(level, pNode, childIndex);
        nodeIdx = transformPointer.index;

        if (transformPointer.translation[0] || transformPointer.translation[1] || transformPointer.translation[2]) {
            glm::ivec3 translation { transformPointer.translation[0], transformPointer.translation[1], transformPointer.translation[2] };

            // auto globalTransform = std::find_if(std::begin(transforms), std::end(transforms), [&](const std::pair<TransformPointer, NodeTransform>& xxx) { return xxx.second == nodeOrder; })->first;
            const auto globalTransform = allTransformPointers[transformIdx];
            translation = applyAxisPermutation(translation, globalTransform.getAxisPermutation());
            translation = glm::mix(translation, -translation, globalTransform.getSymmetry());

            const auto childLevelPosBefore = glm::uvec3(voxel) >> childLevel;
            voxel -= translation;
            const auto childLevelPosAfter = glm::uvec3(voxel) >> childLevel;

            // Empty outside of the child.
            if (glm::any(glm::notEqual(childLevelPosBefore, childLevelPosAfter)))
                return false;
        }

        transformIdx = transformCombineTable[transformIdx * 48 + transformPointer.transformID];
        // nodeOrder = applyTransform(nodeOrder, transforms[transformPointer.transformID].second);
    }

    const uint64_t childMask = applyTransformToBitMask4x4x4(this->subGrids[nodeIdx].bitmask, allTransforms[transformIdx]);
    return (childMask >> morton_encode32(glm::uvec3(voxel) & 3u)) & 0b1;
}
#else
template <typename T, typename BaseType>
bool TranslateTransformBaseDAG<T, BaseType>::get(glm::ivec3 voxel) const
{
    const T* pThis = (const T*)this;
    glm::ivec3 virtualLocation { 0 };
    glm::u8vec3 axisPermutation { 0, 1, 2 };
    uint32_t nodeIdx = 0;
    for (uint32_t level = this->rootLevel; level > this->subGridLevel; --level) {
        const auto childLevel = level - 1;
        const BaseType* pNode = &this->nodesPerLevel[level][nodeIdx];
        const auto childIndex = getChildIdx(level, glm::u8vec3(0, 1, 2), virtualLocation, voxel);
        if (childIndex >= 8 || !pThis->hasChild(pNode, childIndex))
            return false;

        // This solution allows us to forward declare TransformPointer in the header file.
        // transformPointer = pThis->getChildPointer(pNode, childIndex);
        TransformPointer childPointer;
        pThis->getChildPointer(level, pNode, childIndex, childPointer);

        // Move Frame-Of-Reference to child.
        const auto childResolution = 1u << childLevel;
        virtualLocation += morton_decode32<3>(childIndex) * childResolution;

        // Translate Frame-Of-Reference according to pointer.
        virtualLocation += childPointer.getTranslation();

        // Flip Frame-Of-Reference in the case of symmetry.
        // localCoordinate = voxel - virtualLocation;
        // virtualLocation + localCoordinate = voxel;
        {
            const auto localCoordinate = voxel - virtualLocation;
            const auto flippedLocalCoordinate = glm::ivec3(childResolution - 1) - localCoordinate;
            const auto flippedVirtualLocation = voxel - flippedLocalCoordinate;
            virtualLocation = glm::mix(virtualLocation, flippedVirtualLocation, childPointer.getSymmetry());
        }

        // Permute the order in which the axis are processed.
        axisPermutation = invertPermutation(childPointer.getAxisPermutation());
        {
            const auto localCoordinate = voxel - virtualLocation;
            const auto permutedLocalCoordinate = applyAxisPermutation(localCoordinate, axisPermutation);
            voxel = permutedLocalCoordinate + virtualLocation;
        }

        nodeIdx = (uint32_t)childPointer.ptr;
    }

    const glm::ivec3 localCoordinate = voxel - virtualLocation;
    if (glm::any(glm::lessThan(localCoordinate, glm::ivec3(0))) || glm::any(glm::greaterThanEqual(localCoordinate, glm::ivec3(4))))
        return false;

    if (voxel.x == 966 && voxel.y == 469 && voxel.z == 63)
        spdlog::info("localCoordinate = {}", localCoordinate);
    return this->subGrids[nodeIdx].get(glm::uvec3(localCoordinate));
}
#endif

template <typename T, typename BaseType>
uint32_t TranslateTransformBaseDAG<T, BaseType>::getChildIdx(uint32_t level, const glm::u8vec3& axisPermutation, const glm::ivec3& virtualLocation, const glm::ivec3& voxel)
{
    const auto childLevel = level - 1;
    const glm::ivec3 localCoordinate = voxel - virtualLocation;
    // Prevent underflow when converting to unsigned int.
    if (glm::any(glm::lessThan(localCoordinate, glm::ivec3(0))))
        return 0xFFFF'FFFF;

    const glm::ivec3 permutedLocalCoordinate = applyAxisPermutation(localCoordinate, axisPermutation);
    return morton_encode32(glm::uvec3(permutedLocalCoordinate) >> childLevel);
}

size_t TransformDAG16::sizeInBytes() const
{
    size_t out = TransformBaseDAG<uint16_t>::sizeInBytes();
    for (const auto& pointerTable : pointerTables) {
        out += sizeOfVector(pointerTable);
    }
    return out;
}

void TransformDAG16::writeTo(voxcom::BinaryWriter& writer) const
{
    writer.write(constructionConfig);
    writer.write(encodingConfig);
    TransformBaseDAG<uint16_t>::writeTo(writer);
    writer.write(pointerTables);
    writer.write(huffmanDecoders);
}

void TransformDAG16::readFrom(voxcom::BinaryReader& reader)
{
    reader.read(constructionConfig);
    reader.read(encodingConfig);
    TransformBaseDAG<uint16_t>::readFrom(reader);
    reader.read(pointerTables);
    reader.read(huffmanDecoders);
}

template class TranslateTransformBaseDAG<TransformDAG16, uint16_t>;
