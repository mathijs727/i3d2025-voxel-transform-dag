#include "voxcom/utility/bitstream.h"
#include <algorithm> // std::min
#include <bit>
#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <iterator> // std::distance

namespace voxcom {

// Read/write potentially unaligned 32-bit integer from buffer.
static uint32_t read32LE(const uint8_t* buf);
static void write32LE(uint8_t* buf, uint32_t value);

void append_bits(MutBitStream& stream, uint32_t value, int width)
{
    assert(width > 0 && width < 32 - 8);
    const size_t requiredSize = (stream.pos + width + 32) >> 3;
    stream.buf.resize(requiredSize, 0);

    // Read a 32-bit little-endian number starting from the byte containing bit number "pos" (relative to "buf")
    uint32_t bits = read32LE(&stream.buf[stream.pos >> 3]);

    // Shift in the value and combine with existing bits.
    bits |= value << (stream.pos & 0b111);

    write32LE(&stream.buf[stream.pos >> 3], bits);
    stream.pos += width;
}

void append_bit_stream(MutBitStream& stream, BitStream other)
{
    const auto end = other.pos;
    other.pos = 0;
    while (other.pos < end) {
        const int width = (int)std::min((size_t)16, end - other.pos);
        const uint32_t bits = consume_bits(other, width);
        append_bits(stream, bits, width);
    }
}

// Based on the bit_extract_lsb function presented in the following blog post:
// https://fgiesen.wordpress.com/2018/02/19/reading-bits-in-far-too-many-ways-part-1/
uint32_t consume_bits(BitStream& stream, int width)
{
    assert(width > 0 && width < 32 - 8);

    // Read a 32-bit little-endian number starting from the byte containing bit number "pos" (relative to "buf")
    uint32_t bits = read32LE(&stream.buf[stream.pos >> 3]);

    // Shift out the bits inside the first byte that we've already consumed.
    bits >>= stream.pos & 0b111;

    stream.pos += width;
    // Return the low "width" bits, zeroing the rest via bit mask.
    return bits & ((1ul << width) - 1);
}

// Read 32-bit integer from potentially unaligned buffer.
static uint32_t read32LE(const uint8_t* buf)
{
    // Casting a uint8_t ptr to a uint32_t ptr and dereferencing may lead to undefined behavior because
    // the C/C++ standard say that uint32_t ptr must be 4-byte aligned as required by some hardware architectures.
    //
    // Instead we perform a memcpy which should lead to just a "mov" instruction on x86 without breaking completely
    // for other architectures.
    uint32_t out;
    std::memcpy(&out, buf, sizeof(uint32_t));
    return out;
}

// Write 32-bit integer to potentially unaligned buffer.
static void write32LE(uint8_t* buf, uint32_t value)
{
    std::memcpy(buf, &value, sizeof(uint32_t));
}

MutBitStream::operator BitStream() const
{
    return BitStream { .buf = buf.data(), .pos = pos };
}

MutBitStream::MutBitStream(std::pmr::memory_resource* pMemoryResource)
    : buf(pMemoryResource)
{
}

MutBitStream2::MutBitStream2(std::pmr::memory_resource* pMemoryResource)
    : buf(pMemoryResource)
{
}

MutBitStream2::operator BitStream2() const
{
    return BitStream2 { .buf = buf.data(), .pos = 0, .end = pos };
}

uint32_t consume_bits(BitStream2& stream, int width)
{
    const uint32_t posInArray = stream.pos / 32;
    const uint32_t posInUint32 = stream.pos % 32;

    const uint32_t part1 = stream.buf[posInArray];
    const uint32_t mask1 = (1 << width) - 1;
    uint32_t value = (part1 >> posInUint32) & mask1;
    if (posInUint32 + width > 32) {
        const uint32_t bitsProcessed = 32 - posInUint32;
        const uint32_t bitsToGo = width - bitsProcessed;

        const uint32_t part2 = stream.buf[posInArray + 1];
        const uint32_t mask2 = (1 << bitsToGo) - 1;
        value |= (part2 & mask2) << bitsProcessed;
    }
    assert(stream.pos <= std::numeric_limits<uint32_t>::max() - width);
    if (stream.pos > std::numeric_limits<uint32_t>::max() - width)
        throw std::exception();
    stream.pos += width;
    return value;
}

void append_bit_stream(MutBitStream2& stream, BitStream2 other)
{
    while (other.pos < other.end) {
        const int numBits = (int)std::min(other.end - other.pos, 31u);
        const uint32_t bits = consume_bits(other, numBits);
        append_bits(stream, bits, numBits);
    }
}

uint32_t append_bit_stream_align_u32(MutBitStream2& stream, BitStream2 other)
{
    // Round stream.pos up to next u32
    const auto streamPosU32 = stream.pos == 0 ? 0 : ((stream.pos - 1) / 32 + 1);
    // Compute how many blocks of u32 we need to copy over from other stream.
    assert(other.end > 0);
    const auto u32ToCopyOver = ((other.end - 1) / 32) + 1;
    // Use very efficient memcpy
    stream.buf.resize(streamPosU32 + u32ToCopyOver);
    std::memcpy(&stream.buf[streamPosU32], other.buf, u32ToCopyOver * sizeof(uint32_t));
    // Set new stream pos
    stream.pos = streamPosU32 * 32 + other.end;

    return (uint32_t)streamPosU32;
}

uint32_t append_bit_stream_align_u32(tbb::concurrent_vector<uint32_t>& stream, BitStream2 other)
{
    // Compute how many blocks of u32 we need to copy over from other stream.
    assert(other.end > 0);
    const auto u32ToCopyOver = ((other.end - 1) / 32) + 1;

    // Reserve and copy over the data.
    const auto startIter = stream.grow_by(u32ToCopyOver);
    std::copy(other.buf, other.buf + u32ToCopyOver, startIter);
    return (uint32_t)std::distance(std::begin(stream), startIter);
}

}
