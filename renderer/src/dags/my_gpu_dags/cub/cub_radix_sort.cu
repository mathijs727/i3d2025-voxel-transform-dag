#include <cuda.h>
#pragma warning(disable : 4324)
#include <cub/device/device_radix_sort.cuh>
#pragma warning(default : 4324)
#include "cub_radix_sort.h"

template <typename T>
void cubDeviceRadixSortKeys(std::span<const T> inKeys, std::span<T> outKeys, cudaStream_t stream)
{
    size_t tmpStorageSize = 0;
    void* pTmpStorage = nullptr;
    cub::DeviceRadixSort::SortKeys(pTmpStorage, tmpStorageSize, inKeys.data(), outKeys.data(), inKeys.size(), 0, sizeof(T) * 8, stream);
    cudaMallocAsync(&pTmpStorage, tmpStorageSize, stream);
    cub::DeviceRadixSort::SortKeys(pTmpStorage, tmpStorageSize, inKeys.data(), outKeys.data(), inKeys.size(), 0, sizeof(T) * 8, stream);
    cudaFreeAsync(pTmpStorage, stream);
}

template void cubDeviceRadixSortKeys(std::span<const uint32_t>, std::span<uint32_t>, cudaStream_t);
