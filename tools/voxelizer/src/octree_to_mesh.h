#pragma once
#include <vector>
#include <voxcom/core/mesh.h>
#include <voxcom/voxel/structure.h>
#include <voxcom/voxel/voxel_grid.h>

template <typename Attribute>
std::vector<voxcom::Mesh> octreeToMesh(const voxcom::EditStructure<Attribute, uint32_t>& tree);

template <typename Attribute>
std::vector<voxcom::Mesh> octreeToMesh(const voxcom::VoxelGrid<Attribute>& grid);
