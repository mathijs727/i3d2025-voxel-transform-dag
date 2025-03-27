#pragma once
#include "voxcom/attributes/color.h"
#include "voxcom/core/bounds.h"
#include "voxcom/voxel/ssvdag.h"
#include "voxcom/voxel/structure.h"
#include <filesystem>

namespace voxcom {

// Store octree to a file that is compatible with the SSVDAG code-base:
// https://github.com/RvanderLaan/SVDAG-Compression
template <template <typename, typename> typename Structure, typename Pointer>
void exportUSSVDAG(const Structure<void, Pointer>& structure, const Bounds& bounds, const std::filesystem::path& filePath);
EditStructure<void, uint32_t> importSVDAG(const std::filesystem::path& filePath, Bounds& outBounds);

}