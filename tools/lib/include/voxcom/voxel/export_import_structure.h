#pragma once
#include "structure.h"
#include <filesystem>

namespace voxcom {

template <template <typename, typename> typename Structure, typename Attribute>
void exportStructure(const Structure<Attribute, uint32_t>&, const std::filesystem::path&);

template <typename T>
EditStructure<T, uint32_t> importEditStructure(const std::filesystem::path&);
template <typename T>
EditStructureOOC<T, uint32_t> importEditStructureOOC(const std::filesystem::path&);
template <typename T>
StaticStructure<T, uint32_t> importStaticStructure(const std::filesystem::path&);

}
