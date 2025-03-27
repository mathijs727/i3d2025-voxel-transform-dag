#pragma once
#include "voxcom/attributes/color.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <ostream>
#include <vector>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/common.hpp> // glm::lerp
#include <glm/glm.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

namespace glm {
inline std::ostream& operator<<(std::ostream& stream, const voxcom::RGB& vec)
{
    stream << "(" << (int)vec.r << ", " << (int)vec.g << ", " << (int)vec.b << ")";
    return stream;
}
}

struct aiTexture; // Forward declare assimp

namespace voxcom {

template <typename P>
struct Image2D {
public:
    using Pixel = P;

    glm::ivec2 resolution;
    std::vector<Pixel> pixels;

public:
    constexpr Image2D(const glm::ivec2& resolution) noexcept;
    Image2D(const std::filesystem::path&);
    Image2D(const aiTexture*);

    constexpr void set(const glm::ivec2&, const Pixel& pixel) noexcept;
    [[nodiscard]] constexpr Pixel get(const glm::ivec2&) const noexcept;
    // Get a pixel. If x and/or y is larger than resolution then wrap around.
    [[nodiscard]] constexpr Pixel getWrapped(const glm::ivec2&) const noexcept;

    [[nodiscard]] constexpr Pixel sampleNN(const glm::vec2&) const noexcept;
    [[nodiscard]] constexpr Pixel sampleBilinear(const glm::vec2&) const noexcept;

private:
};

template <typename Pixel>
constexpr Image2D<Pixel>::Image2D(const glm::ivec2& resolution) noexcept
    : resolution(resolution)
{
    pixels.resize((size_t)(resolution.x * resolution.y));
}

template <typename Pixel>
constexpr inline void Image2D<Pixel>::set(const glm::ivec2& pixel, const Pixel& value) noexcept
{
    pixels[pixel.y * resolution.x + pixel.x] = value;
}

template <typename Pixel>
constexpr inline Pixel Image2D<Pixel>::get(const glm::ivec2& pixel) const noexcept
{
    return pixels[pixel.y * resolution.x + pixel.x];
}

template <typename Pixel>
constexpr inline Pixel Image2D<Pixel>::getWrapped(const glm::ivec2& pixel) const noexcept
{
    glm::ivec2 wrappedPixel = pixel % resolution;
    if (wrappedPixel.x < 0)
        wrappedPixel.x = resolution.x + wrappedPixel.x;
    if (wrappedPixel.y < 0)
        wrappedPixel.y = resolution.y + wrappedPixel.y;
    return get(wrappedPixel);
}

template <typename Pixel>
constexpr Pixel Image2D<Pixel>::sampleNN(const glm::vec2& texCoord) const noexcept
{
    glm::ivec2 texel = glm::ivec2(texCoord * glm::vec2(resolution));
    texel = texel % resolution; // Nearest-neighbor sampling with texture wrapping
    if (texel.x < 0)
        texel.x += resolution.x;
    if (texel.y < 0)
        texel.y += resolution.y;
    return this->get(texel);
}

template <typename Pixel>
constexpr Pixel Image2D<Pixel>::sampleBilinear(const glm::vec2& texCoord) const noexcept
{
    // May be negative and/or greater than resolution (both will use texture wrapping).O
    const glm::vec2 texel = texCoord * glm::vec2(resolution) - 0.5f;
    glm::vec2 integralPart;
    glm::vec2 fractionalPart { std::modf(texel.x, &integralPart.x), std::modf(texel.y, &integralPart.y) };
    if (fractionalPart.x < 0.0f)
        fractionalPart.x += 1.0f;
    if (fractionalPart.y < 0.0f)
        fractionalPart.y += 1.0f;

    const glm::ivec2 iTexel { integralPart };
    const auto NW = getWrapped(iTexel + glm::ivec2(0, 0));
    const auto NE = getWrapped(iTexel + glm::ivec2(1, 0));
    const auto SW = getWrapped(iTexel + glm::ivec2(0, 1));
    const auto SE = getWrapped(iTexel + glm::ivec2(1, 1));
    if constexpr (std::is_same_v<Pixel, RGB>) {
        static_assert(sizeof(RGB) == 3);
        const auto north = glm::mix((glm::vec3)NW, (glm::vec3)NE, fractionalPart.x);
        const auto south = glm::mix((glm::vec3)SW, (glm::vec3)SE, fractionalPart.x);
        return make_rgb(glm::mix(north, south, fractionalPart.y));
    } else {
        assert(false); // Code below is untested. Add unit test before using it.
        const auto north = std::lerp(NW, NE, fractionalPart.x);
        const auto south = std::lerp(SW, SE, fractionalPart.x);
        return std::lerp(north, south, fractionalPart.y);
    }
}
}
