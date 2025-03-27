#pragma once
#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

namespace voxcom {
class BinaryReader;
class BinaryWriter;
}

struct Code {
    uint64_t code;
    uint32_t bits;

    constexpr auto operator<=>(const Code&) const = default;
};

std::vector<Code> assignCodeWords(std::span<const uint32_t> codeLengths);
std::vector<Code> createHuffmanCodeTable(std::span<const size_t> frequencies);
std::vector<Code> createLengthLimitedHuffmanCodeTable(std::span<const size_t> frequencies, uint32_t maxCodeLength);

void convertHuffmanToLSB(std::span<Code> codes);

class HuffmanDecodeLUT {
public:
    HuffmanDecodeLUT() = default; // Required for readFrom(...);
    HuffmanDecodeLUT(std::span<const Code> codes);

    inline std::pair<uint32_t, uint32_t> decode(uint32_t key) const { return m_huffmanCode[key & m_bitMask]; }

    void writeTo(voxcom::BinaryWriter& writer) const;
    void readFrom(voxcom::BinaryReader& reader);

private:
    void initFromInputCodes();

private:
    std::vector<std::pair<uint32_t, uint32_t>> m_huffmanCode;
    uint32_t m_bitMask;

    std::vector<Code> m_inputCodes; // Write to file in a much more compact way.
};

float kraftSum(std::span<const uint32_t> codeLengths);

void testHuffman();
