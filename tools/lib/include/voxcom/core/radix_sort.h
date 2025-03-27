#pragma once
#include <algorithm>
#include <array>
#include <numeric>
#include <span>
#include <tuple>
#include <type_traits>

namespace voxcom {

template <typename T, size_t BitsPerIteration = 8>
void radix_sort(std::span<T> values)
{
    constexpr size_t NumBits = sizeof(T) * 8;
    constexpr size_t HistogramSize = 1u << BitsPerIteration;
    constexpr T BitMask = ((T)1 << BitsPerIteration) - (T)1;

    // Even number of passes so the output is stored in "values" and not "values2".
    static_assert(NumBits % BitsPerIteration == 0);
    static_assert((NumBits / BitsPerIteration) % 2 == 0);

    std::vector<T> tmp(values.size());
    std::span<T> values2 { tmp };
    for (size_t startingBit = 0; startingBit < NumBits; startingBit += BitsPerIteration) {
        // Create histogram.
        std::array<size_t, HistogramSize> histogram;
        std::fill(std::begin(histogram), std::end(histogram), 0);
        for (T value : values) {
            const T bits = (value >> startingBit) & BitMask;
            ++histogram[bits];
        }

        // Partial sort.
        std::exclusive_scan(std::begin(histogram), std::end(histogram), std::begin(histogram), (size_t)0);
        for (T value : values) {
            const T bits = (value >> startingBit) & BitMask;
            values2[histogram[bits]++] = value;
        }

        // Swap the (pointers to) the two buffers.
        std::swap(values, values2);
    }
}

template <typename Key, typename Value, size_t BitsPerIteration = 8>
void radix_sort_key_values(std::span<std::pair<Key, Value>> items, std::span<std::pair<Key, Value>> items2)
{
    constexpr size_t NumBits = sizeof(Key) * 8;
    constexpr size_t HistogramSize = 1u << BitsPerIteration;
    constexpr Key BitMask = ((Key)1 << BitsPerIteration) - (Key)1;

    // Even number of passes so the output is stored in "values" and not "values2".
    static_assert(NumBits % BitsPerIteration == 0);
    static_assert((NumBits / BitsPerIteration) % 2 == 0);

    for (size_t startingBit = 0; startingBit < NumBits; startingBit += BitsPerIteration) {
        // Create histogram.
        size_t histogram[HistogramSize];
        std::fill(histogram, histogram + HistogramSize, 0);
        for (const auto& [key, value] : items) {
            const Key bits = (key >> startingBit) & BitMask;
            ++histogram[bits];
        }

        // Partial sort.
        std::exclusive_scan(std::begin(histogram), std::end(histogram), std::begin(histogram), (size_t)0);
        for (const auto& item : items) {
            const Key bits = (std::get<0>(item) >> startingBit) & BitMask;
            items2[histogram[bits]++] = item;
        }

        // Swap the (pointers to) the two buffers.
        std::swap(items, items2);
    }
}

template <typename Key, typename Value, size_t BitsPerIteration = 8>
void radix_sort_key_values(std::span<std::pair<Key, Value>> items)
{
    std::vector<std::pair<Key, Value>> tmp(items.size());
    return radix_sort_key_values<Key, Value, BitsPerIteration>(items, tmp);
}

}