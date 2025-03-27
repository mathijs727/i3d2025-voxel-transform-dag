#pragma once
#include <limits>
#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

namespace voxcom {

struct Bounds {
public:
    glm::vec3 lower { std::numeric_limits<float>::max() };
    glm::vec3 upper { std::numeric_limits<float>::lowest() };

public:
    Bounds& operator*=(const glm::mat4& matrix);
    bool operator==(const Bounds& other) const;

    void reset();
    void grow(const glm::vec3& vec);
    void extend(const Bounds& other);
    Bounds extended(const Bounds& other) const;

    glm::vec3 center() const;
    glm::vec3 extent() const;
    float surfaceArea() const;

    Bounds intersection(const Bounds& other) const;
    bool overlaps(const Bounds& other) const;
    bool contains(const glm::vec3& point) const;
    bool contains(const Bounds& other) const;
};
}