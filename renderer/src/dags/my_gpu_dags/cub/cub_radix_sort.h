#pragma once
#include <span>

template <typename T>
void cubDeviceRadixSortKeys(std::span<const T> inKeys, std::span<T> outKeys, cudaStream_t stream);