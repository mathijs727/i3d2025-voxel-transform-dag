#pragma once
#include "voxcom/attributes/color.h"
#include "voxcom/core/bounds.h"
#include "voxcom/voxel/structure.h"
#include <filesystem>

namespace voxcom {

// Store octree to a file that is compatible with the HashDAG program:
// https://github.com/Phyronnaz/HashDAG
template <template <typename, typename> typename Structure, typename Attribute>
void exportHashDAG(const Structure<Attribute, uint32_t>& structure, const Bounds& bounds, const std::filesystem::path& filePath);

}