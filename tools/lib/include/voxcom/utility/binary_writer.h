#pragma once
#include "is_of_type.h"
#include "voxcom/utility/error_handling.h"
#include <cassert>
#include <concepts>
#include <filesystem>
#include <ostream>
#include <optional>

namespace voxcom {

class BinaryWriter;
template <typename T>
concept has_write_to = requires(const T& item, BinaryWriter& writer) {
    {
        item.writeTo(writer)
    }
    -> std::same_as<void>;
};

// Write objects to a binary file.
// Can serialize trivially copyable types, std::string, std::filesystem, std::vector and std::variant.
class BinaryWriter {
public:
    std::ostream& m_outputStream;

public:
    BinaryWriter(std::ostream& outputStream);

    template <typename T>
    void writeRange(const T* items, size_t sizeInItems);
    template <typename T>
    void write(const T&);
    template <typename T>
    void write(const std::optional<T>&);

private:
    template <typename T, size_t idx = 0>
    void writeVariant(const T& variant);
};

inline BinaryWriter::BinaryWriter(std::ostream& outputStream)
    : m_outputStream(outputStream)
{
    assert(m_outputStream);
}

template <typename T>
inline void BinaryWriter::write(const T& src)
{
    if constexpr (has_write_to<T>) {
        src.writeTo(*this);
    } else if constexpr (is_std_vector<T>::value) {
        using ItemT = typename T::value_type;
        write(src.size());
        if constexpr (std::is_trivially_copyable_v<ItemT> && !has_write_to<T>) {
            m_outputStream.write(reinterpret_cast<const char*>(src.data()), src.size() * sizeof(ItemT));
        } else {
            for (const auto& item : src)
                write(item);
        }
    } else if constexpr (is_std_variant<T>::value) {
        writeVariant(src);
    } else if constexpr (is_std_optional_v<T>) {
        write(src.has_value());
        if (src.has_value())
            write(src.value());
    } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
        write(src.string());
    } else if constexpr (std::is_same_v<T, std::string>) {
        write(src.size());
        m_outputStream.write(src.data(), src.size());
    } else if constexpr (std::is_trivially_copyable_v<T>) {
        m_outputStream.write(reinterpret_cast<const char*>(&src), sizeof(T));
    } else {
        voxcom::assert_always(false);
    }
}
template <typename T>
inline void BinaryWriter::writeRange(const T* items, size_t sizeInItems)
{
    m_outputStream.write(reinterpret_cast<const char*>(items), sizeInItems * sizeof(T));
}

template <typename T>
void BinaryWriter::write(const std::optional<T>& optSrc)
{
    const bool hasValue = optSrc.has_value();
    write(hasValue);
    if (hasValue)
        write(optSrc.value());
}

template <typename T, size_t idx>
void BinaryWriter::writeVariant(const T& variant)
{
    if (variant.index() == idx)
        write(std::get<idx>(variant));

    if constexpr (idx + 1 < std::variant_size_v<T>)
        writeVariant<T, idx + 1>(variant);
}

}
