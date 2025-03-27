#pragma once
#include "disable_all_warnings.h"
DISABLE_WARNINGS_PUSH()
#include <fmt/format.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
DISABLE_WARNINGS_POP()

// https://fmt.dev/latest/api.html#udt
template <glm::length_t L, typename T>
struct fmt::formatter<glm::vec<L, T>> {
    // Parses empty format specifications.
    constexpr auto parse(format_parse_context& ctx)
    {
        // Parse the presentation format and store it in the formatter:
        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            throw format_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    template <typename FormatContext>
    auto format(const glm::vec<L, T>& p, FormatContext& ctx)
    {
        // auto format(const point &p, FormatContext &ctx) -> decltype(ctx.out()) // c++11
        // ctx.out() is an output iterator to write to.
        if constexpr (L == 2) {
            return fmt::format_to(
                ctx.out(),
                "({}, {})",
                p.x, p.y);
        } else if constexpr (L == 3) {
            return fmt::format_to(
                ctx.out(),
                "({}, {}, {})",
                p.x, p.y, p.z);
        } else if constexpr (L == 4) {
            return fmt::format_to(
                ctx.out(),
                "({}, {}, {}, {})",
                p.x, p.y, p.z, p.w);
        }
    }
};
