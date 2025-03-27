#pragma once
#include <span>

template <typename T>
void cubDeviceSelectUnique(std::span<const T> inKeys, std::span<T> outKeys, uint32_t* pNumOutKeys, cudaStream_t stream);
