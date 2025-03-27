#pragma once
#include <algorithm>
#include <execution>
#include <span>
#include <vector>

namespace voxcom {

template <typename IndexT, typename ItemT>
std::vector<IndexT> argSort(std::span<const ItemT> items)
{
    std::vector<IndexT> indices(items.size());
    std::iota(std::begin(indices), std::end(indices), 0);
    std::sort(std::execution::par, std::begin(indices), std::end(indices), [&](size_t lhs, size_t rhs) { return items[lhs] < items[rhs]; });
    return indices;
}

template <typename IndexT, typename ItemT>
std::vector<IndexT> indexSort(std::span<ItemT> items)
{
    const auto indices = argSort<IndexT, ItemT>(items);
    std::vector<IndexT> out(items.size());
    std::transform(
        std::execution::par,
        std::begin(indices), std::end(indices), std::begin(out),
        [&](size_t index) { return items[index]; });
    std::copy(std::execution::par, std::begin(out), std::end(out), std::begin(items));
    return indices;
}

template <typename IndexT, typename ItemT>
std::vector<IndexT> inPlaceIndexSort(std::span<ItemT> items)
{
    const auto indices = argSort<IndexT, ItemT>(items);
    std::sort(std::execution::par, std::begin(items), std::end(items));
    return indices;
}

template <typename T>
std::vector<T> invertIndexDirection(std::span<const T> indices) {
    std::vector<T> out(indices.size());
    for (T i = 0; i < (T)indices.size(); ++i) {
        out[indices[i]] = i;
    }
    return out;
}

}