#include <bit>
#include <catch2/catch_all.hpp>
#include <span>
#include <tbb/concurrent_vector.h>
#include <voxcom/utility/bitstream.h>

using namespace voxcom;
using namespace voxcom;

static size_t vectorSizeInBytes(const auto& vec)
{
    return vec.size() * sizeof(typename std::remove_reference_t<decltype(vec)>::value_type);
}

static uint32_t consume_bits(MutBitStream& mutStream, int width)
{
    BitStream stream { .buf = mutStream.buf.data(), .pos = mutStream.pos };
    const uint32_t value = consume_bits(stream, width);
    mutStream.pos = stream.pos;
    return value;
}

TEST_CASE("Write then read data from BitStream  at fixed rate", "[BitStream]")
{
    SECTION("Fixed 16-bit packing")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(128, random<uint32_t>(0, 65535))));

        MutBitStream stream;
        for (const auto& value : values) {
            append_bits(stream, value, 16);
        }

        stream.pos = 0;
        for (const auto& value : values) {
            const uint32_t readValue = consume_bits(stream, 16);
            REQUIRE(readValue == value);
        }
    }

    SECTION("Fixed 8-bit packing")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(128, random<uint32_t>(0, 255))));

        MutBitStream stream;
        for (const auto& value : values) {
            append_bits(stream, value, 8);
        }

        stream.pos = 0;
        for (const auto& value : values) {
            const uint32_t readValue = consume_bits(stream, 8);
            REQUIRE(readValue == value);
        }
    }

    SECTION("Fixed 4-bit packing")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(128, random<uint32_t>(0, 15))));

        MutBitStream stream;
        for (const auto& value : values) {
            append_bits(stream, value, 4);
        }

        stream.pos = 0;
        for (const auto& value : values) {
            const uint32_t readValue = consume_bits(stream, 4);
            REQUIRE(readValue == value);
        }
    }

    SECTION("Fixed 3-bit packing")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(128, random<uint32_t>(0, 7))));

        MutBitStream stream;
        for (const auto& value : values) {
            append_bits(stream, value, 3);
        }

        stream.pos = 0;
        for (const auto& value : values) {
            const uint32_t readValue = consume_bits(stream, 3);
            REQUIRE(readValue == value);
        }
    }
}

TEST_CASE("Write then read data from BitStream at variable rate", "[BitStream]")
{
    SECTION("Variable packing up to 16-bit")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(128, random<uint32_t>(1, 65535))));

        MutBitStream stream;
        for (const auto& value : values) {
            append_bits(stream, value, std::bit_width(value));
        }

        stream.pos = 0;
        for (const auto& value : values) {
            const uint32_t readValue = consume_bits(stream, std::bit_width(value));
            REQUIRE(readValue == value);
        }
    }

    SECTION("Variable packing up to 8-bit")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(128, random<uint32_t>(1, 255))));

        MutBitStream stream;
        for (const auto& value : values) {
            append_bits(stream, value, std::bit_width(value));
        }

        stream.pos = 0;
        for (const auto& value : values) {
            const uint32_t readValue = consume_bits(stream, std::bit_width(value));
            REQUIRE(readValue == value);
        }
    }

    SECTION("Variable packing up to 11-bit with 3-bit padding")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(128, random<uint32_t>(1, 255))));

        MutBitStream stream;
        for (const auto& value : values) {
            append_bits(stream, value, std::bit_width(value) + 3);
        }

        stream.pos = 0;
        for (const auto& value : values) {
            const uint32_t readValue = consume_bits(stream, std::bit_width(value) + 3);
            REQUIRE(readValue == value);
        }
    }
}

TEST_CASE("Append BitStreams to each other", "[BitStream]")
{
    // Value for each pixel.
    const auto values = GENERATE(take(3, chunk(128, random<uint32_t>(1, 65535))));
    const size_t half = values.size() / 2;

    MutBitStream stream1, stream2;
    for (const auto& value : std::span(values).subspan(0, half)) {
        append_bits(stream1, value, std::bit_width(value));
    }
    for (const auto& value : std::span(values).subspan(half)) {
        append_bits(stream2, value, std::bit_width(value));
    }
    append_bit_stream(stream1, stream2);

    stream1.pos = 0;
    for (const auto& value : values) {
        const uint32_t readValue = consume_bits(stream1, std::bit_width(value));
        REQUIRE(readValue == value);
    }
}

TEST_CASE("Write then read data from BitStream2 at fixed rate", "[BitStream][BitStream2]")
{
    SECTION("Fixed 16-bit packing")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(128, random<uint32_t>(0, 65535))));

        MutBitStream2 mutStream;
        for (const auto& value : values) {
            append_bits(mutStream, value, 16);
        }

        BitStream2 stream = mutStream;
        for (const auto& value : values) {
            const uint32_t readValue = consume_bits(stream, 16);
            REQUIRE(readValue == value);
        }
    }

    SECTION("Fixed 8-bit packing")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(128, random<uint32_t>(0, 255))));

        MutBitStream2 mutStream;
        for (const auto& value : values) {
            append_bits(mutStream, value, 8);
        }

        BitStream2 stream = mutStream;
        for (const auto& value : values) {
            const uint32_t readValue = consume_bits(stream, 8);
            REQUIRE(readValue == value);
        }
    }

    SECTION("Fixed 4-bit packing")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(128, random<uint32_t>(0, 15))));

        MutBitStream2 mutStream;
        for (const auto& value : values) {
            append_bits(mutStream, value, 4);
        }

        BitStream2 stream = mutStream;
        for (const auto& value : values) {
            const uint32_t readValue = consume_bits(stream, 4);
            REQUIRE(readValue == value);
        }
    }

    SECTION("Fixed 3-bit packing")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(11, random<uint32_t>(0, 7))));

        MutBitStream2 mutStream;
        for (const auto& value : values) {
            append_bits(mutStream, value, 3);
        }

        BitStream2 stream = mutStream;
        for (const auto& value : values) {
            const uint32_t readValue = consume_bits(stream, 3);
            REQUIRE(readValue == value);
        }
    }
}

TEST_CASE("Append BitStream2 to each other", "[BitStream2]")
{
    SECTION("Bit packing append")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(1024, random<uint32_t>(1, 65535))));
        const size_t half = values.size() / 2;

        MutBitStream2 stream1, stream2;
        for (const auto& value : std::span(values).subspan(0, half)) {
            append_bits(stream1, value, std::bit_width(value));
        }
        for (const auto& value : std::span(values).subspan(half)) {
            append_bits(stream2, value, std::bit_width(value));
        }
        append_bit_stream(stream1, stream2);

        BitStream2 consumeStream = stream1;
        for (const auto& value : values) {
            const uint32_t readValue = consume_bits(consumeStream, std::bit_width(value));
            REQUIRE(readValue == value);
        }
    }

    SECTION("u32 aligned append")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(1024, random<uint32_t>(1, 65535))));
        const size_t half = values.size() / 2;

        MutBitStream2 stream1, stream2;
        for (const auto& value : std::span(values).subspan(0, half)) {
            append_bits(stream1, value, std::bit_width(value));
        }
        for (const auto& value : std::span(values).subspan(half)) {
            append_bits(stream2, value, std::bit_width(value));
        }
        const auto startSecondHalfU32 = append_bit_stream_align_u32(stream1, stream2);

        BitStream2 consumeStream = stream1;
        for (const auto& value : std::span(values).subspan(0, half)) {
            const uint32_t readValue = consume_bits(consumeStream, std::bit_width(value));
            REQUIRE(readValue == value);
        }
        consumeStream.pos = startSecondHalfU32 * 32;
        for (const auto& value : std::span(values).subspan(half)) {
            const uint32_t readValue = consume_bits(consumeStream, std::bit_width(value));
            REQUIRE(readValue == value);
        }
    }

    SECTION("u32 aligned append to tbb::concurrent_vector")
    {
        // Value for each pixel.
        const auto values = GENERATE(take(3, chunk(1024, random<uint32_t>(1, 65535))));
        const size_t half = values.size() / 2;

        MutBitStream2 stream1, stream2;
        for (const auto& value : std::span(values).subspan(0, half)) {
            append_bits(stream1, value, std::bit_width(value));
        }
        for (const auto& value : std::span(values).subspan(half)) {
            append_bits(stream2, value, std::bit_width(value));
        }
        tbb::concurrent_vector<uint32_t> tbbVec;
        const auto startFirstHalfU32 = append_bit_stream_align_u32(tbbVec, stream1);
        const auto startSecondHalfU32 = append_bit_stream_align_u32(tbbVec, stream2);

        MutBitStream2 outStream;
        outStream.buf.resize(tbbVec.size());
        std::copy(std::begin(tbbVec), std::end(tbbVec), std::begin(outStream.buf));
        outStream.pos = (uint32_t)(tbbVec.size() * sizeof(uint32_t) * 8);

        BitStream2 consumeStream = outStream;
        consumeStream.pos = startFirstHalfU32 * 32;
        for (const auto& value : std::span(values).subspan(0, half)) {
            const uint32_t readValue = consume_bits(consumeStream, std::bit_width(value));
            REQUIRE(readValue == value);
        }
        consumeStream.pos = startSecondHalfU32 * 32;
        for (const auto& value : std::span(values).subspan(half)) {
            const uint32_t readValue = consume_bits(consumeStream, std::bit_width(value));
            REQUIRE(readValue == value);
        }
    }
}
