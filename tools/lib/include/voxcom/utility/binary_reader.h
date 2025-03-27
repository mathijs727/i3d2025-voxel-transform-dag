#pragma once
#include "is_of_type.h"
#include "voxcom/utility/error_handling.h"
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <istream>
#include <optional>
#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <mio/mmap.hpp>
DISABLE_WARNINGS_POP()

namespace voxcom {

class BinaryReader;
template <typename T>
concept has_read_from = requires(T& item, BinaryReader& reader) {
    {
        item.readFrom(reader)
    }
    -> std::same_as<void>;
};

template <typename T>
struct read_mmap_type { };

template <typename S>
struct read_mmap_type<std::vector<S>> {
    using type = std::vector<typename read_mmap_type<S>::type>;
};
template <typename S>
struct read_mmap_type<std::span<const S>> {
    using type = mio::mmap_source;
};
template <typename T>
using read_mmap_type_t = typename read_mmap_type<T>::type;

class BinaryReader {
public:
    BinaryReader(std::istream& inputStream);

    template <typename T>
    void readRange(T* items, size_t sizeInItems);
    template <typename T>
    void read(T& dst);
    template <typename T>
    void read(std::optional<T>& dst);

    template <typename T>
    T read();

    template <typename T>
    void readMmap(std::vector<T>& dst, read_mmap_type_t<std::vector<T>>& dstMemoryMap, const std::filesystem::path& currentFilePath);
    template <typename T>
    void readMmap(std::span<const T>& dst, mio::mmap_source& dstMemoryMap, const std::filesystem::path& currentFilePath);

private:
    template <typename T>
    void readVariant(T& dst);
    template <typename T, size_t curIdx = 0>
    void readVariantIdx(T& dst, size_t dstIdx);

private:
    std::istream& m_inputStream;
};

template <typename T>
void BinaryReader::readMmap(std::vector<T>& dst, read_mmap_type_t<std::vector<T>>& dstMemoryMap, const std::filesystem::path& currentFilePath)
{
    const size_t vectorLength = read<size_t>();
    dst.resize(vectorLength);
    dstMemoryMap.resize(vectorLength);
    for (size_t i = 0; i < vectorLength; ++i) {
        readMmap(dst[i], dstMemoryMap[i], currentFilePath);
    }
}

template <typename T>
void BinaryReader::readMmap(std::span<const T>& dst, mio::mmap_source& dstMemoryMap, const std::filesystem::path& currentFilePath)
{
    static_assert(std::is_trivially_copyable_v<T>);
    const size_t vectorLength = read<size_t>();
    if (vectorLength == 0)
        return;

    std::error_code err;
    const size_t offsetInFile = m_inputStream.tellg();
    dstMemoryMap = mio::make_mmap_source(currentFilePath.string(), offsetInFile, vectorLength * sizeof(T), err);
    assert_always(!err);
    dst = std::span((const T*)dstMemoryMap.data(), vectorLength);
    m_inputStream.seekg(vectorLength * sizeof(T), std::ios::cur);
}

inline BinaryReader::BinaryReader(std::istream& inputStream)
    : m_inputStream(inputStream)
{
    assert_always((bool)m_inputStream);
}

template <typename T>
void BinaryReader::readRange(T* items, size_t sizeInItems)
{
    m_inputStream.read(reinterpret_cast<char*>(items), sizeInItems * sizeof(T));
}

template <typename T>
inline void BinaryReader::read(T& dst)
{
    dst = read<T>();
}

template <typename T>
inline void BinaryReader::read(std::optional<T>& dst)
{
    const bool containsValue = read<bool>();
    if (containsValue)
        dst.emplace(read<T>());
}

template <typename T>
inline T BinaryReader::read()
{
    if constexpr (has_read_from<T>) {
        T dst {};
        dst.readFrom(*this);
        return dst;
    } else if constexpr (is_std_vector<T>::value) {
        using ItemT = typename T::value_type;
        const size_t vectorLength = read<size_t>();

        T dst;
        if constexpr (std::is_trivially_copyable_v<ItemT>) {
            dst.resize(vectorLength);
            m_inputStream.read(reinterpret_cast<char*>(dst.data()), vectorLength * sizeof(ItemT));
        } else {
            for (size_t i = 0; i < vectorLength; i++)
                dst.push_back(read<ItemT>());
        }
        return dst;
    } else if constexpr (is_std_variant<T>::value) {
        T dst;
        readVariant(dst);
        return dst;
    } else if constexpr (is_std_optional_v<T>) {
        if (read<bool>())
            return read<std_optional_type_t<T>>();
        else
            return {};
    } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
        return std::filesystem::path(read<std::string>());
    } else if constexpr (std::is_same_v<T, std::string>) {
        const size_t stringLength = read<size_t>();

        std::string dst;
        dst.resize(stringLength);
        m_inputStream.read(dst.data(), dst.size());
        return dst;
    } else if constexpr (std::is_trivially_copyable_v<T>) {
        T dst {};
        m_inputStream.read(reinterpret_cast<char*>(&dst), sizeof(T));
        return dst;
    } else {
        assert_always(false);
        return {};
    }
}

template <typename T>
void BinaryReader::readVariant(T& dst)
{
    size_t index;
    read(index);

    readVariantIdx<T, 0>(dst, index);
}

template <typename T, size_t curIdx>
void BinaryReader::readVariantIdx(T& dst, size_t dstIdx)
{
    if (curIdx == dstIdx)
        read(std::get<curIdx>(dst));

    if constexpr (curIdx + 1 < std::variant_size_v<T>)
        readVariantIdx<T, curIdx + 1>(dst, dstIdx);
}

}
