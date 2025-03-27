#pragma once
#include "array.h"
#include "dags/base_dag.h"
#include "my_units.h"
#include "typedefs.h"
#include "utils.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>

class BinaryReader;

struct SymmetryAwareDAG16 {
    StaticArray<Leaf> leaves;

    StaticArray<uint32_t> levelStarts;
    StaticArray<uint16_t> nodes;

    static constexpr uint32_t levels = MAX_LEVELS;
    static constexpr uint32_t leafLevel = 2; // lvl0 = 1x1x1, lvl1 = 2x2x2, lvl2 = 4x4x4 leaves

    struct TraversalConstants {
        std::array<std::array<uint8_t, 256>, 8> symmetryChildMask; // Child mask after a symmetry is applied.
    };
    TraversalConstants createTraversalConstants() const;

    struct SymmetryPointer {
        uint32_t index;
        uint32_t symmetry;

        static HOST_DEVICE uint32_t invSymmetry(uint32_t inSymmetry)
        {
            uint32_t out = 0;
            if (inSymmetry & 0b001)
                out |= 0b100;
            if (inSymmetry & 0b010)
                out |= 0b010;
            if (inSymmetry & 0b100)
                out |= 0b001;
            return out;
        }

        static HOST_DEVICE SymmetryPointer decode(uint16_t const* pPointer, uint32_t headerTag)
        {
            SymmetryPointer out {};
            if (headerTag == 0b01) {
                // check(encodedPointer < 0xFFFF);
                const uint32_t encodedPointer = pPointer[0];
                out.index = encodedPointer & 0b1'1111'1111'1111; // 13 bits
                out.symmetry = encodedPointer >> 13;
            } else {
                const uint32_t encodedPointer = uint32_t(pPointer[1]) + (uint32_t(pPointer[0]) << 16);
                out.index = encodedPointer & 0b1'1111'1111'1111'1111'1111'1111'1111; // 29 bits
                out.symmetry = encodedPointer >> 29;
                if (headerTag == 0b11)
                    out.index |= 1u << 29;
            }
            out.symmetry = invSymmetry(out.symmetry);
            return out;
        }
    };

    HOST_DEVICE bool is_valid() const
    {
        return nodes.is_valid();
    }

    HOST void upload_to_gpu()
    {
        leaves.upload_to_gpu();
        levelStarts.upload_to_gpu();
        nodes.upload_to_gpu();
    }
    HOST SymmetryAwareDAG16 copy(EMemoryType newMemoryType) const
    {
        auto out = *this;
        out.leaves = leaves.copy(newMemoryType);
        out.nodes = nodes.copy(newMemoryType);
        out.levelStarts = levelStarts.copy(newMemoryType);
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

    HOST_DEVICE SymmetryPointer get_child_index(uint32_t childLevel, uint16_t const* pNode, uint16_t header, uint8 childIndex) const
    {
        // Find start of pointer for the given childIndex.
        const uint16_t* pPointer = pNode + 1; // First child pointer.
        const uint16_t preMask = 0xFFFF << (2 * (childIndex + 1));

        uint16_t preChildMask = header & preMask;
        // Turn 0b11 into 0b10: generate 0b01 if 0b11, then XOR to flip the least significant bit.
        uint16_t clippedPreChildMask = preChildMask ^ (((preChildMask & 0b10101010'10101010) >> 1) & (preChildMask & 0b01010101'01010101));
        const uint16_t partialSum1 = (preChildMask & 0b0011001100110011) + ((clippedPreChildMask >> 2) & 0b0011001100110011);
        const uint16_t partialSum2 = (partialSum1 & 0b0000111100001111) + ((partialSum1 >> 4) & 0b0000111100001111);
        const uint16_t partialSum3 = (partialSum2 & 0b0000000011111111) + ((partialSum2 >> 8) & 0b0000000011111111);
        pPointer += partialSum3;

        return SymmetryPointer::decode(pPointer, (header >> (2 * childIndex)) & 0b11);
    }
    HOST_DEVICE Leaf get_leaf(uint32 index) const
    {
        return leaves[index];
    }

    HOST my_units::bytes memory_used() const
    {
        return leaves.memory_used() + levelStarts.memory_used() + nodes.memory_used();
    }
    HOST void print_stats() const
    {
        printf("Leaves data: %fMB\n", Utils::to_MB(leaves.memory_used()));
        printf("Nodes data: %fMB\n", Utils::to_MB(nodes.memory_used() + levelStarts.memory_used()));
    }
    HOST void free()
    {
        leaves.free();
        nodes.free();
        levelStarts.free();
    }
};

struct DAGInfo;
struct SymmetryAwareDAGFactory {
    static void load_dag_from_file(DAGInfo& outInfo, SymmetryAwareDAG16& outDag, const std::filesystem::path& path, EMemoryType memoryType);
};