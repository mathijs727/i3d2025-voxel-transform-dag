#include "voxcom/voxel/export_import_structure.h"
#include "voxcom/utility/binary_reader.h"
#include "voxcom/utility/binary_writer.h"
#include "voxcom/utility/error_handling.h"
#include "voxcom/voxel/structure.h"
#include <cstdio>
#include <fstream>
#include <voxcom/utility/error_handling.h>

namespace voxcom {

static constexpr uint32_t magicNumber = 72198382;
static constexpr uint32_t octreeFormatVersion = 2;

template <template <typename, typename> typename Structure, typename Attribute>
void exportStructure(const Structure<Attribute, uint32_t>& structure, const std::filesystem::path& filePath)
{
    std::ofstream file { filePath, std::ios::binary };
    assert(file.is_open());
    BinaryWriter writer { file };
    writer.write(magicNumber);
    writer.write(octreeFormatVersion);
    structure.writeTo(writer);
}

template <typename T>
EditStructure<T, uint32_t> importEditStructure(const std::filesystem::path& filePath)
{
    std::ifstream file { filePath, std::ios::binary };
    BinaryReader reader { file };
    assert_always(reader.read<uint32_t>() == magicNumber);
    assert_always(reader.read<uint32_t>() == octreeFormatVersion);
    EditStructure<T, uint32_t> out;
    out.readFrom(reader);
    return out;
}
template <typename T>
EditStructureOOC<T, uint32_t> importEditStructureOOC(const std::filesystem::path& filePath)
{
    std::ifstream file { filePath, std::ios::binary };
    BinaryReader reader { file };
    assert_always(reader.read<uint32_t>() == magicNumber);
    assert_always(reader.read<uint32_t>() == octreeFormatVersion);
    return EditStructureOOC<T, uint32_t>(reader, filePath);
}
template <typename T>
StaticStructure<T, uint32_t> importStaticStructure(const std::filesystem::path& filePath)
{
    std::ifstream file { filePath, std::ios::binary };
    BinaryReader reader { file };
    assert_always(reader.read<uint32_t>() == magicNumber);
    assert_always(reader.read<uint32_t>() == octreeFormatVersion);
    StaticStructure<T> out;
    out.readFrom(reader);
    return out;
}

template void exportStructure(const EditStructure<void, uint32_t>&, const std::filesystem::path&);
template void exportStructure(const EditStructureOOC<void, uint32_t>&, const std::filesystem::path&);
template void exportStructure(const StaticStructure<void, uint32_t>&, const std::filesystem::path&);
template EditStructure<void, uint32_t> importEditStructure(const std::filesystem::path&);
template EditStructureOOC<void, uint32_t> importEditStructureOOC(const std::filesystem::path&);

template void exportStructure(const EditStructure<voxcom::RGB, uint32_t>&, const std::filesystem::path&);
template void exportStructure(const EditStructureOOC<voxcom::RGB, uint32_t>&, const std::filesystem::path&);
template void exportStructure(const StaticStructure<voxcom::RGB, uint32_t>&, const std::filesystem::path&);
template EditStructure<voxcom::RGB, uint32_t> importEditStructure(const std::filesystem::path&);
template EditStructureOOC<voxcom::RGB, uint32_t> importEditStructureOOC(const std::filesystem::path&);

}
