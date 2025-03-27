#pragma once
#include <span>

template <typename T>
void cubInclusiveSumMinusOne(std::span<const T> items, std::span<T> result, cudaStream_t stream);

template <typename T>
void cubExclusiveSum(std::span<const T> items, std::span<T> result, cudaStream_t stream);
