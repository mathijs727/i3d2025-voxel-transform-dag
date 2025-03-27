#pragma once
#include <algorithm> // std::clamp
#include <cstdint>
#include <ostream>
#include <string>
#include <voxcom/utility/hash.h>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <fmt/format.h>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
DISABLE_WARNINGS_POP()

namespace voxcom {

struct RGB {
    uint8_t r = 0, g = 0, b = 0;

    auto operator<=>(const RGB&) const noexcept = default;
    inline operator glm::u8vec4() const noexcept
    {
        return glm::u8vec4(r, g, b, 0);
    }
    inline operator glm::vec3() const noexcept
    {
        return glm::vec3(r, g, b) / 255.0f;
    }
    inline operator glm::ivec3() const noexcept
    {
        return glm::ivec3(r, g, b);
    }
    inline uint8_t operator[](size_t i) const noexcept
    {
        if (i == 0)
            return r;
        else if (i == 1)
            return g;
        else if (i == 2)
            return b;
        else
            return 0; // Don't do this.
    }
    inline uint8_t& operator[](size_t i) noexcept
    {
        if (i == 0)
            return r;
        else if (i == 1)
            return g;
        else if (i == 2)
            return b;
        else
            return r; // Don't do this.
    }
};
// Separate function instead of constructor so we can use designated initializers on RGB
inline RGB make_rgb(const glm::vec3& c)
{
    return RGB {
        .r = (uint8_t)std::clamp(c.r * 255.0f, 0.0f, 255.0f),
        .g = (uint8_t)std::clamp(c.g * 255.0f, 0.0f, 255.0f),
        .b = (uint8_t)std::clamp(c.b * 255.0f, 0.0f, 255.0f),
    };
}
inline RGB make_rgb(const glm::ivec3& c)
{
    return RGB { .r = (uint8_t)c.r, .g = (uint8_t)c.g, .b = (uint8_t)c.b };
}
inline RGB make_rgb(const glm::u8vec4& c)
{
    return RGB { .r = (uint8_t)c.r, .g = (uint8_t)c.g, .b = (uint8_t)c.b };
}

static_assert(sizeof(RGB) == 3);

}

inline voxcom::RGB operator*(float lhs, const voxcom::RGB& rhs)
{
    return voxcom::RGB { uint8_t(lhs * rhs.r), uint8_t(lhs * rhs.g), uint8_t(lhs * rhs.b) };
}
inline voxcom::RGB operator+(const voxcom::RGB& lhs, const voxcom::RGB& rhs)
{
    return voxcom::RGB { uint8_t(lhs.r + rhs.r), uint8_t(lhs.g + rhs.g), uint8_t(lhs.b + rhs.b) };
}

namespace std {

inline std::ostream& operator<<(std::ostream& stream, const voxcom::RGB& lhs)
{
    stream << "(" << (int)lhs.r << ", " << (int)lhs.g << ", " << (int)lhs.b << ")";
    return stream;
}

template <>
struct hash<voxcom::RGB> {
    size_t operator()(const voxcom::RGB& rgb) const noexcept
    {
        uint32_t s = (uint32_t)rgb.r;
        s |= (uint32_t)rgb.g << 8;
        s |= (uint32_t)rgb.b << 16;
        return std::hash<uint32_t>()(s);
    }
};

}

template <>
struct fmt::formatter<voxcom::RGB> : formatter<std::string> {
    // parse is inherited from formatter<string_view>.
    template <typename FormatContext>
    auto format(voxcom::RGB c, FormatContext& ctx)
    {
        std::string name = "(";
        name += std::to_string(c.r) + ", ";
        name += std::to_string(c.g) + ", ";
        name += std::to_string(c.b) + ")";
        return formatter<std::string>::format(name, ctx);
    }
};
