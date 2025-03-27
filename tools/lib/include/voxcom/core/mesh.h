#pragma once
#include "voxcom/core/bounds.h"
#include "voxcom/core/image.h"
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

namespace voxcom {

struct Mesh {
public:
    std::vector<glm::vec3> positions;
    std::vector<glm::vec2> texCoords;
    std::vector<glm::vec2> lightMapCoords;
    std::vector<glm::uvec3> triangles;
    std::vector<glm::uvec4> quads;

    voxcom::RGB diffuseBaseColor;
    std::shared_ptr<const Image2D<voxcom::RGB>> pDiffuseTexture;
    std::shared_ptr<const Image2D<voxcom::RGB>> pLightMapTexture;

public:
    Bounds computeBounds() const;
};

std::vector<Mesh> loadMeshes(const std::filesystem::path&);
void saveMesh(const Mesh&, const std::filesystem::path&);
void saveMesh(std::span<const Mesh>, const std::filesystem::path&);

}