#pragma once
#include "voxcom/attributes/color.h"
#include "voxcom/voxel/structure.h"
#include <span>
#include <vector>
#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

namespace voxcom {
struct Mesh;
struct Bounds;
}

namespace voxcom {

std::vector<Mesh> prepareMeshForVoxelization(std::span<const Mesh> meshes, unsigned resolution, Bounds& inOutSceneBounds);

template <typename Attribute, bool conservative>
EditStructure<Attribute, uint32_t> voxelizeHierarchical(std::span<const Mesh> meshes, unsigned resolution);

// Loop over all voxels in 3D bounding box of triangle.
template <bool conservative, typename Target>
void voxelizeMeshNaive(Target& target, const Mesh& mesh, const glm::vec3& translation = glm::vec3(0), float scale = 1.0f);
// Project triangles onto plane based on dominant axis and find min/max depth along each pixel column.
template <bool conservative, typename Target>
void voxelizeMeshOptimized(Target& target, const Mesh& mesh, const glm::vec3& translation = glm::vec3(0), float scale = 1.0f);

EditStructure<void, uint32_t> voxelizeSparseSolid(std::span<const Mesh> meshes, unsigned resolution);

}