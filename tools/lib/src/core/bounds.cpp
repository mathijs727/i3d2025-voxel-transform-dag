#include "voxcom/core/bounds.h"
#include <limits>

namespace voxcom {

Bounds& Bounds::operator*=(const glm::mat4& matrix)
{
    const glm::vec3 v0 = matrix * glm::vec4(lower.x, lower.y, lower.z, 1.0f);
    const glm::vec3 v1 = matrix * glm::vec4(lower.x, lower.y, upper.z, 1.0f);
    const glm::vec3 v2 = matrix * glm::vec4(lower.x, upper.y, lower.z, 1.0f);
    const glm::vec3 v3 = matrix * glm::vec4(lower.x, upper.y, upper.z, 1.0f);
    const glm::vec3 v4 = matrix * glm::vec4(upper.x, lower.y, lower.z, 1.0f);
    const glm::vec3 v5 = matrix * glm::vec4(upper.x, lower.y, upper.z, 1.0f);
    const glm::vec3 v6 = matrix * glm::vec4(upper.x, upper.y, lower.z, 1.0f);
    const glm::vec3 v7 = matrix * glm::vec4(upper.x, upper.y, upper.z, 1.0f);
    this->lower = glm::min(v0, glm::min(v1, glm::min(v2, glm::min(v3, glm::min(v4, glm::min(v5, glm::min(v6, v7)))))));
    this->upper = glm::max(v0, glm::max(v1, glm::max(v2, glm::max(v3, glm::max(v4, glm::max(v5, glm::max(v6, v7)))))));
    return *this;
}

bool Bounds::operator==(const Bounds& other) const
{
    return lower == other.lower && upper == other.upper;
}

void Bounds::reset()
{
    lower = glm::vec3(std::numeric_limits<float>::max());
    upper = glm::vec3(std::numeric_limits<float>::lowest());
}

void Bounds::grow(const glm::vec3& vec)
{
    lower = glm::min(lower, vec);
    upper = glm::max(upper, vec);
}

void Bounds::extend(const Bounds& other)
{
    lower = glm::min(lower, other.lower);
    upper = glm::max(upper, other.upper);
}

Bounds Bounds::extended(const Bounds& other) const
{
    auto lowerBounds = glm::min(lower, other.lower);
    auto upperBounds = glm::max(upper, other.upper);
    return { lowerBounds, upperBounds };
}

glm::vec3 Bounds::center() const
{
    return (lower + upper) / 2.0f;
}

glm::vec3 Bounds::extent() const
{
    return upper - lower;
}

float Bounds::surfaceArea() const
{
    glm::vec3 extent = upper - lower;
    return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
}

Bounds Bounds::intersection(const Bounds& other) const
{
    return Bounds { glm::max(lower, other.lower), glm::min(upper, other.upper) };
}

bool Bounds::overlaps(const Bounds& other) const
{
    glm::vec3 extent = intersection(other).extent();
    return (extent.x >= 0.0f && extent.y >= 0.0f && extent.z >= 0.0f);
}

bool Bounds::contains(const glm::vec3& point) const
{
    return glm::all(glm::greaterThanEqual(point, lower)) && glm::all(glm::lessThanEqual(point, upper));
}

bool Bounds::contains(const Bounds& other) const
{
    return contains(other.lower) && contains(other.upper);
}

}
