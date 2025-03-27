#pragma once
#include <bit>
#include <cassert>
#include <cstdint>
#include <immintrin.h>
#include <memory_resource>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <tbb/concurrent_vector.h>
DISABLE_WARNINGS_POP()

namespace voxcom {

struct BitStream {
    uint8_t const* buf = nullptr;
    size_t pos = 0;
};
struct MutBitStream {
    std::pmr::vector<uint8_t> buf;
    size_t pos = 0;

    MutBitStream() = default;
    MutBitStream(std::pmr::memory_resource* pMemoryResource);
    operator BitStream() const;
};

void append_bits(MutBitStream& stream, uint32_t value, int width);
void append_bit_stream(MutBitStream& stream, BitStream other);
uint32_t consume_bits(BitStream& stream, int width);

struct BitStream2 {
    uint32_t const* buf = nullptr;
    uint32_t pos = 0, end = 0;
};
struct MutBitStream2 {
    std::pmr::vector<uint32_t> buf;
    uint32_t pos = 0;

    MutBitStream2() = default;
    MutBitStream2(std::pmr::memory_resource* pMemoryResource);
    operator BitStream2() const;
};
void append_bits(MutBitStream2& stream, uint32_t value, int width);
void append_bit_stream(MutBitStream2& stream, BitStream2 other);
// Returns starting address measured in u32's (units of 4 bytes).
[[nodiscard]] uint32_t append_bit_stream_align_u32(MutBitStream2& stream, BitStream2 other);
[[nodiscard]] uint32_t append_bit_stream_align_u32(tbb::concurrent_vector<uint32_t>& stream, BitStream2 other);
uint32_t consume_bits(BitStream2& stream, int width);

inline void append_bits(MutBitStream2& stream, uint32_t value, int width)
{
    assert(width <= 32);
    assert((int)std::bit_width(value) <= width);

    const uint32_t posInArray = stream.pos / 32;
    if (posInArray + 1 >= stream.buf.size())
        stream.buf.resize(posInArray + 8);
    const uint32_t posInUint32 = stream.pos % 32;

    // const uint32_t rotatedValue = std::rotl(value, posInUint32);
    const uint32_t rotatedValue = _rotl(value, posInUint32);
    const uint32_t mask1 = ~((1 << posInUint32) - 1);
    const uint32_t x1 = rotatedValue & mask1;
    stream.buf[posInArray] |= x1;
    if (posInUint32 + width > 32) {
        const uint32_t mask2 = ~mask1; // Mask includes extra bits but those should be 0 (if bit_width(value) <= width)
        const uint32_t x2 = rotatedValue & mask2;
        stream.buf[posInArray + 1] |= x2;
    }
    stream.pos += width;
}

}
