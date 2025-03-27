#pragma once
#include "array.h"
#include "configuration/transform_dag_definitions.h"
#include "dags/base_dag.h"
#include "my_units.h"
#include "typedefs.h"
#include "utils.h"
#include <array>
#include <cstdint>
#include <filesystem>

struct small_int3 {
    int8_t xyz[3];

    HOST_DEVICE small_int3 operator+(const small_int3& other) const
    {
        return small_int3 { {
            int8_t((int)xyz[0] + (int)other.xyz[0]),
            int8_t((int)xyz[1] + (int)other.xyz[1]),
            int8_t((int)xyz[2] + (int)other.xyz[2]),
        } };
    }
};

class BinaryReader;
struct HuffmanCodeWord {
    uint32_t code : 27;
    uint32_t numBits : 5;
};

struct TransformDAG16 {
    StaticArray<Leaf> leaves;

    StaticArray<uint32_t> levelStarts;
    StaticArray<uint16_t> nodes;

#if TRANSFORM_DAG_USE_POINTER_TABLES
    StaticArray<uint32_t> tableStarts;
    StaticArray<uint64_t> pointerTable;
#endif

#if TRANSFORM_DAG_USE_HUFFMAN_CODE
    struct HuffmanDecoderLUT {
        uint32_t codeWordStart;
        uint32_t bitMask;
    };
    StaticArray<HuffmanCodeWord> huffmanCodeWords;
    StaticArray<HuffmanDecoderLUT> huffmanDecoders;
#endif

    static constexpr uint32_t levels = MAX_LEVELS;
    static constexpr uint32_t leafLevel = 2; // lvl0 = 1x1x1, lvl1 = 2x2x2, lvl2 = 4x4x4 leaves

    struct HuffmanCodePoint {
        uint32_t code : 27;
        uint32_t numBits : 5;
    };
    struct Transform {
        uint8_t symmetry;
        uint8_t axis0;
        uint8_t axis1;
        uint8_t axis2;
    };
    struct TraversalConstants {
#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
        static constexpr size_t NumTransforms = 48;
        std::array<Transform, NumTransforms> transforms;
        std::array<std::array<uint8_t, 8>, NumTransforms> transformChildMappingWorldToLocal;
        std::array<std::array<uint8_t, 256>, NumTransforms> transformMaskMappingLocalToWorld;
        std::array<std::array<uint8_t, NumTransforms>, NumTransforms> transformCombineTable;
#endif

        std::array<std::array<uint8_t, 8>, 256> nextChildLUT;
    };
    TraversalConstants createTraversalConstants() const;

    struct TransformPointer {
        uint32_t index;
#if TRANSFORM_DAG_USE_TRANSLATION
        small_int3 translation;
#endif

#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
        uint8_t transformID;
#endif

        static HOST_DEVICE TransformPointer create(uint64_t ptr)
        {
            return TransformPointer
            {
                .index = (uint32_t)ptr,
#if TRANSFORM_DAG_USE_TRANSLATION
                .translation = small_int3 { 0, 0, 0 },
#endif
#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
                .transformID = 0
#endif
            };
        }

#if TRANSFORM_DAG_USE_HUFFMAN_CODE
        static HOST_DEVICE TransformPointer decode(uint64_t encodedPointer, uint32_t childLevel, HuffmanCodeWord const* pHuffmanCodes, uint32_t huffmanMask)
        {
            auto codeWord = pHuffmanCodes[(uint32_t)encodedPointer & huffmanMask];
            encodedPointer >>= codeWord.numBits;

            TransformPointer out;
#if TRANSFORM_DAG_USE_TRANSLATION
            uint32_t hasTranslationAxis = 0;
            if (childLevel <= TRANSFORM_DAG_MAX_TRANSLATION_LEVEL) {
                hasTranslationAxis = codeWord.code & 0b111;
                codeWord.code >>= 3;
            }
#endif

#if TRANSFORM_DAG_USE_SYMMETRY && TRANSFORM_DAG_USE_AXIS_PERMUTATION
            out.transformID = 0;
            out.transformID |= codeWord.code & 0b111;
            codeWord.code >>= 3;
            out.transformID |= (codeWord.code & 0b111) << 3;
            codeWord.code >>= 3;
            check(out.transformID < 48);
#elif TRANSFORM_DAG_USE_SYMMETRY
            out.transformID = codeWord.code & 0b111;
#elif TRANSFORM_DAG_USE_AXIS_PERMUTATION
            out.transformID = (codeWord.code & 0b111) << 3;
            check(out.transformID < 48);
#endif

#if TRANSFORM_DAG_USE_TRANSLATION
            out.translation = small_int3 { { 0, 0, 0 } };
            if (hasTranslationAxis) {
                const uint32_t translationBits = childLevel + 1;
                const uint64_t translationMask = (1llu << translationBits) - 1llu;
                const int8_t offset = 1 << childLevel;

                if (hasTranslationAxis & 0b100) {
                    out.translation.xyz[2] = int8_t(encodedPointer & translationMask) - offset;
                    encodedPointer >>= translationBits;
                }
                if (hasTranslationAxis & 0b010) {
                    out.translation.xyz[1] = int8_t(encodedPointer & translationMask) - offset;
                    encodedPointer >>= translationBits;
                }
                if (hasTranslationAxis & 0b001) {
                    out.translation.xyz[0] = int8_t(encodedPointer & translationMask) - offset;
                    encodedPointer >>= translationBits;
                }
            }
#endif

            out.index = (uint32_t)encodedPointer;
            return out;
        }

#else // TRANSFORM_DAG_USE_HUFFMAN_CODE

        static HOST_DEVICE TransformPointer decode(uint64_t encodedPointer, uint32_t childLevel)
        {
#if TRANSFORM_DAG_HAS_TRANSFORMATIONS
            const bool hasTransform = encodedPointer & 0b1;
            encodedPointer >>= 1;

            TransformPointer out;

            if (hasTransform) {
#if TRANSFORM_DAG_USE_TRANSLATION
                if (childLevel <= TRANSFORM_DAG_MAX_TRANSLATION_LEVEL) {
                    const uint32_t translationBits = childLevel + 1;
                    const uint64_t translationMask = (1llu << translationBits) - 1llu;
                    const int8_t offset = 1 << childLevel;

                    out.translation.xyz[0] = int8_t(encodedPointer & translationMask) - offset;
                    encodedPointer >>= translationBits;
                    out.translation.xyz[1] = int8_t(encodedPointer & translationMask) - offset;
                    encodedPointer >>= translationBits;
                    out.translation.xyz[2] = int8_t(encodedPointer & translationMask) - offset;
                    encodedPointer >>= translationBits;
                } else {
                    out.translation = small_int3 { { 0, 0, 0 } };
                }
#endif

#if TRANSFORM_DAG_USE_SYMMETRY && TRANSFORM_DAG_USE_AXIS_PERMUTATION
                out.transformID = encodedPointer & 0b111111;
                encodedPointer >>= 6;
                check(out.transformID < 48);
#elif TRANSFORM_DAG_USE_SYMMETRY
                out.transformID = encodedPointer & 0b111;
                encodedPointer >>= 3;
#elif TRANSFORM_DAG_USE_AXIS_PERMUTATION
                out.transformID = (encodedPointer & 0b111) << 3;
                encodedPointer >>= 3;
                check(out.transformID < 48);
#endif
            } else {
                out.transformID = 0;
#if TRANSFORM_DAG_USE_TRANSLATION
                out.translation.xyz[0] = out.translation.xyz[1] = out.translation.xyz[2] = 0;
#endif
            }

            out.index = (uint32_t)encodedPointer;
            return out;
#else
            return TransformPointer::create(encodedPointer);
#endif
        }

#endif // TRANSFORM_DAG_USE_HUFFMAN_CODE
    };
    static_assert(sizeof(TransformPointer) <= sizeof(uint64_t));

    HOST_DEVICE bool is_valid() const
    {
        return nodes.is_valid();
    }

    HOST void upload_to_gpu()
    {
        leaves.upload_to_gpu();
        levelStarts.upload_to_gpu();
        nodes.upload_to_gpu();
#if TRANSFORM_DAG_USE_POINTER_TABLES
        tableStarts.upload_to_gpu();
        pointerTable.upload_to_gpu();
#endif
#if TRANSFORM_DAG_USE_HUFFMAN_CODE
        huffmanCodeWords.upload_to_gpu();
        huffmanDecoders.upload_to_gpu();
#endif
    }
    HOST TransformDAG16 copy(EMemoryType newMemoryType) const
    {
        auto out = *this;
        out.leaves = leaves.copy(newMemoryType);
        out.nodes = nodes.copy(newMemoryType);
        out.levelStarts = levelStarts.copy(newMemoryType);
#if TRANSFORM_DAG_USE_POINTER_TABLES
        out.tableStarts = tableStarts.copy(newMemoryType);
        out.pointerTable = pointerTable.copy(newMemoryType);
#endif
#if TRANSFORM_DAG_USE_HUFFMAN_CODE
        out.huffmanCodeWords = huffmanCodeWords.copy(newMemoryType);
        out.huffmanDecoders = huffmanDecoders.copy(newMemoryType);
#endif
        return out;
    }

    HOST_DEVICE uint32 get_first_node_index() const
    {
        return 0;
    }

    static HOST_DEVICE uint8_t convert_child_mask(uint16_t header)
    {
        const uint32_t interleavedHeader = header | (header >> 1);
        uint32_t fakeHeader = interleavedHeader & 0b1;
        fakeHeader |= (interleavedHeader >> 1) & 0b10;
        fakeHeader |= (interleavedHeader >> 2) & 0b100;
        fakeHeader |= (interleavedHeader >> 3) & 0b1000;
        fakeHeader |= (interleavedHeader >> 4) & 0b10000;
        fakeHeader |= (interleavedHeader >> 5) & 0b100000;
        fakeHeader |= (interleavedHeader >> 6) & 0b1000000;
        fakeHeader |= (interleavedHeader >> 7) & 0b10000000;
        // for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
        //     if ((header >> (2 * childIdx)) & 0b11)
        //         fakeHeader |= 1 << childIdx;
        // }
        return (uint8_t)fakeHeader;
    }

    static HOST_DEVICE uint64_t memcpy64(uint16_t const* pData, uint32_t numWords)
    {
        const auto copySizeInBytes = numWords * sizeof(uint16_t);

        const auto pointer = (uintptr_t)pData;
        const auto basePointer = pointer & 0xFFFF'FFFF'FFFF'FFF8;
        const auto byteOffset = pointer - basePointer;
        const auto bytesPart0 = sizeof(uint64_t) - byteOffset;

        const uint64_t part0 = *((const uint64_t*)basePointer);
        uint64_t out = part0 >> (8 * byteOffset);
        if (copySizeInBytes > bytesPart0) {
            const uint64_t part1 = *((const uint64_t*)basePointer + 1);
            out |= part1 << (8 * bytesPart0);
        }

        const auto copySizeInBits = copySizeInBytes * 8;
        const uint64_t mask = 0xFFFF'FFFF'FFFF'FFFF >> (64 - copySizeInBits);
        return out & mask;
    }

    HOST_DEVICE TransformPointer get_child_index(uint32_t childLevel, uint16_t const* pNode, uint16_t header, uint8 childIndex) const
    {
        // Find start of pointer for the given childIndex.
        // const uint16_t header = pNode[0];
        const uint16_t* pPointer = pNode + 1; // First child pointer.
        // for (uint32_t i = 0; i < childIndex; ++i) {
        //     const uint16_t ptrBits = (header >> (2 * i)) & 0b11;
        //     pPointer += ptrBits;
        // }
        const uint16_t preMask = 0xFFFF >> (16 - 2 * childIndex);
        uint16_t preChildMask = header & preMask;
        const uint16_t partialSum1 = (preChildMask & 0b0011001100110011) + ((preChildMask >> 2) & 0b0011001100110011);
        const uint16_t partialSum2 = (partialSum1 & 0b0000111100001111) + ((partialSum1 >> 4) & 0b0000111100001111);
        const uint16_t partialSum3 = (partialSum2 & 0b0000000011111111) + ((partialSum2 >> 8) & 0b0000000011111111);
        pPointer += partialSum3;

        const uint16_t ptrBits = (header >> (2 * childIndex)) & 0b11;
        check(ptrBits != 0b00);

        // Read (up to) the 64-bit pointer.
        // uint64_t encodedPointer = 0;
        // std::memcpy(&encodedPointer, pPointer, ptrBits * sizeof(uint16_t));
        uint64_t encodedPointer = memcpy64(pPointer, ptrBits);

#if TRANSFORM_DAG_USE_POINTER_TABLES
        if (ptrBits == 0b01 || (encodedPointer & 0b1)) {
            // 16-bit pointer is always an index into the table. Otherwise, first (of two) bits indicates whether it's a table or not.
            const auto tableIndex = (ptrBits == 0b01) ? encodedPointer : (encodedPointer >> 1);
            encodedPointer = pointerTable[tableStarts[childLevel] + tableIndex];
        } else {
            encodedPointer = (encodedPointer >> 1);
        }
#endif

#if TRANSFORM_DAG_USE_HUFFMAN_CODE
        const HuffmanDecoderLUT huffmanDecoder = huffmanDecoders[childLevel];
        return TransformPointer::decode(encodedPointer, childLevel, &huffmanCodeWords[huffmanDecoder.codeWordStart], huffmanDecoder.bitMask);
#else
        return TransformPointer::decode(encodedPointer, childLevel);
#endif
    }
    HOST_DEVICE Leaf get_leaf(uint32 index) const
    {
        return leaves[index];
    }

    HOST my_units::bytes memory_used() const
    {
        size_t out = leaves.memory_used() + levelStarts.memory_used() + nodes.memory_used();
#if TRANSFORM_DAG_USE_POINTER_TABLES
        out += tableStarts.memory_used() + pointerTable.memory_used();
#endif
#if TRANSFORM_DAG_USE_HUFFMAN_CODE
        out += huffmanCodeWords.memory_used() + huffmanDecoders.memory_used();
#endif
        return out;
    }
    HOST void print_stats() const
    {
        printf("Leaves data: %fMB\n", Utils::to_MB(leaves.memory_used()));
        printf("Nodes data: %fMB\n", Utils::to_MB(nodes.memory_used() + levelStarts.memory_used()));
#if TRANSFORM_DAG_USE_POINTER_TABLES
        printf("Pointers data: %fMB\n", Utils::to_MB(tableStarts.memory_used() + pointerTable.memory_used()));
#endif
#if TRANSFORM_DAG_USE_HUFFMAN_CODE
        printf("Huffman decoders: %fMB\n", Utils::to_MB(huffmanCodeWords.memory_used() + huffmanDecoders.memory_used()));
#endif
    }
    HOST void free()
    {
        leaves.free();
        nodes.free();
        levelStarts.free();
#if TRANSFORM_DAG_USE_POINTER_TABLES
        tableStarts.free();
        pointerTable.free();
#endif
#if TRANSFORM_DAG_USE_HUFFMAN_CODE
        huffmanCodeWords.free();
        huffmanDecoders.free();
#endif
    }
};

struct DAGInfo;
struct TransformDAGFactory {
    static void load_dag_from_file(DAGInfo& outInfo, TransformDAG16& outDag, const std::filesystem::path& path, EMemoryType memoryType);
};