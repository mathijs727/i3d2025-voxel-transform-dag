#pragma once
#include <span>

template <typename T1, typename T2>
void cubDeviceMergeSortPairs(std::span<T1> keys, std::span<T2> items, cudaStream_t stream);
