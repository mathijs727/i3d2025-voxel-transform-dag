#include <catch2/catch_all.hpp>
#include <ostream>
#include <voxcom/attributes/color.h>
#include <voxcom/core/image.h>

using namespace voxcom;

TEST_CASE("Image2D base constructor", "[Image2D]")
{
    Image2D<RGB> image { glm::ivec2(2, 3) };
    REQUIRE(image.resolution.x == 2);
    REQUIRE(image.resolution.y == 3);
    REQUIRE(image.pixels.size() == 6);
}

TEST_CASE("Image2D file constructor", "[Image2D]")
{
    Image2D<RGB> image { "data/test_image_2x3.png" };
    REQUIRE(image.resolution.x == 2);
    REQUIRE(image.resolution.y == 3);
    REQUIRE(image.pixels.size() == 6);
    REQUIRE(image.get({ 0, 0 }) == RGB { 255, 0, 0 });
    REQUIRE(image.get({ 1, 0 }) == RGB { 0, 255, 0 });
    REQUIRE(image.get({ 0, 1 }) == RGB { 0, 0, 255 });
    REQUIRE(image.get({ 1, 1 }) == RGB { 255, 255, 0 });
    REQUIRE(image.get({ 0, 2 }) == RGB { 0, 255, 255 });
    REQUIRE(image.get({ 1, 2 }) == RGB { 255, 0, 255 });
}

TEST_CASE("Image2D setters and getters", "[Image2D]")
{
    SECTION("Basic")
    {
        Image2D<RGB> image { glm::ivec2(2, 3) };
        image.set({ 0, 0 }, RGB { 0, 0, 0 });
        image.set({ 1, 0 }, RGB { 0, 0, 0 });
        image.set({ 0, 1 }, RGB { 0, 0, 1 });
        image.set({ 1, 1 }, RGB { 0, 0, 1 });
        image.set({ 0, 2 }, RGB { 0, 1, 0 });
        image.set({ 1, 2 }, RGB { 0, 1, 0 });
        REQUIRE(image.get({ 0, 0 }) == RGB { 0, 0, 0 });
        REQUIRE(image.get({ 1, 0 }) == RGB { 0, 0, 0 });
        REQUIRE(image.get({ 0, 1 }) == RGB { 0, 0, 1 });
        REQUIRE(image.get({ 1, 1 }) == RGB { 0, 0, 1 });
        REQUIRE(image.get({ 0, 2 }) == RGB { 0, 1, 0 });
        REQUIRE(image.get({ 1, 2 }) == RGB { 0, 1, 0 });
    }

    SECTION("Random")
    {
        const auto numbers = GENERATE(take(100, chunk(20, random<uint32_t>(0, 255))));

        Image2D<RGB> image { glm::ivec2(5, 4) };
        size_t i = 0;
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 5; x++) {
                const uint32_t n = numbers[i++];
                const RGB pixel { (uint8_t)n, (uint8_t)(n >> 8), (uint8_t)(n >> 16) };
                image.set({ x, y }, pixel);
            }
        }

        i = 0;
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 5; x++) {
                const uint32_t n = numbers[i++];
                const RGB pixel { (uint8_t)n, (uint8_t)(n >> 8), (uint8_t)(n >> 16) };
                REQUIRE(image.get({ x, y }) == pixel);
            }
        }
    }
}

TEST_CASE("Image2D wrapping getter", "[Image2D]")
{
    Image2D<RGB> image { glm::ivec2(2, 3) };
    image.set({ 0, 0 }, RGB { 0, 0, 0 });
    image.set({ 1, 0 }, RGB { 0, 0, 0 });
    image.set({ 0, 1 }, RGB { 0, 0, 1 });
    image.set({ 1, 1 }, RGB { 0, 0, 1 });
    image.set({ 0, 2 }, RGB { 0, 1, 0 });
    image.set({ 1, 2 }, RGB { 0, 1, 0 });

    for (int x = -20; x < 20; x += 2) {
        for (int y = -30; y < 30; y += 3) {
            const glm::ivec2 offset { x, y };
            REQUIRE(image.getWrapped(offset + glm::ivec2(0, 0)) == RGB { 0, 0, 0 });
            REQUIRE(image.getWrapped(offset + glm::ivec2(1, 0)) == RGB { 0, 0, 0 });
            REQUIRE(image.getWrapped(offset + glm::ivec2(0, 1)) == RGB { 0, 0, 1 });
            REQUIRE(image.getWrapped(offset + glm::ivec2(1, 1)) == RGB { 0, 0, 1 });
            REQUIRE(image.getWrapped(offset + glm::ivec2(0, 2)) == RGB { 0, 1, 0 });
            REQUIRE(image.getWrapped(offset + glm::ivec2(1, 2)) == RGB { 0, 1, 0 });
        }
    }
}

TEST_CASE("Image2D bilinear interpolation", "[Image2D]")
{
    Image2D<RGB> image { glm::ivec2(2, 2) };
    image.set({ 0, 0 }, RGB { 255, 255, 255 });
    image.set({ 1, 0 }, RGB { 255, 000, 000 });
    image.set({ 0, 1 }, RGB { 000, 255, 000 });
    image.set({ 1, 1 }, RGB { 000, 000, 255 });

    SECTION("Sampling pixel centers")
    {
        REQUIRE(image.sampleBilinear(glm::vec2(0.25f, 0.25f)) == RGB { 255, 255, 255 });
        REQUIRE(image.sampleBilinear(glm::vec2(0.75f, 0.25f)) == RGB { 255, 0, 0 });
        REQUIRE(image.sampleBilinear(glm::vec2(0.25f, 0.75f)) == RGB { 0, 255, 0 });
        REQUIRE(image.sampleBilinear(glm::vec2(0.75f, 0.75f)) == RGB { 0, 0, 255 });
    }

    SECTION("Sampling in the middle between pixels")
    {
        // Between [0, 1]
        REQUIRE(image.sampleBilinear(glm::vec2(0.50f, 0.25f)) == RGB { 255, 127, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(0.50f, 0.75f)) == RGB { 000, 127, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(0.25f, 0.50f)) == RGB { 127, 255, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(0.75f, 0.50f)) == RGB { 127, 000, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(0.50f, 0.50f)) == RGB { 127, 127, 127 });

        // Between [1, 2]
        REQUIRE(image.sampleBilinear(glm::vec2(1.50f, 1.25f)) == RGB { 255, 127, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(1.50f, 1.75f)) == RGB { 000, 127, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(1.25f, 1.50f)) == RGB { 127, 255, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(1.75f, 1.50f)) == RGB { 127, 000, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(1.50f, 1.50f)) == RGB { 127, 127, 127 });

        // Between [-1, 0]
        REQUIRE(image.sampleBilinear(glm::vec2(-0.50f, -0.75f)) == RGB { 255, 127, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(-0.50f, -0.25f)) == RGB { 000, 127, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(-0.75f, -0.50f)) == RGB { 127, 255, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(-0.25f, -0.50f)) == RGB { 127, 000, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(-0.50f, -0.50f)) == RGB { 127, 127, 127 });
    }

    SECTION("Sampling at the boundary")
    {
        REQUIRE(image.sampleBilinear(glm::vec2(0.50f, 0.00f)) == RGB { 127, 127, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(0.50f, 1.00f)) == RGB { 127, 127, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(0.00f, 0.50f)) == RGB { 127, 127, 127 });
        REQUIRE(image.sampleBilinear(glm::vec2(1.00f, 0.50f)) == RGB { 127, 127, 127 });
    }
}
