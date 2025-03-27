#pragma once
#include <cstdint>
#include <cstdio>

#ifdef __CUDACC__
#define CONSTEXPR_GLM
#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__

[[maybe_unused]] static void __cudaCheckError(const char* file, unsigned line)
{
    {
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
            std::fprintf(stderr, "ERROR %s:%d: cuda error \"%s\"\n", file, line, cudaGetErrorString(err));
            std::abort();
        }
    }
}
#define CUDA_CHECKED_CALL ::detail::CudaErrorChecker(__LINE__, __FILE__) =
#define CUDA_CHECK_ERROR() __cudaCheckError(__FILE__, __LINE__)

#else
#define CONSTEXPR_GLM constexpr
#define HOST
#define DEVICE
#define HOST_DEVICE
#define CUDA_CHECKED_CALL
#define CUDA_CHECK_ERROR()
#endif

[[maybe_unused]] inline uint32_t computeNumWorkGroups(size_t domain, uint32_t workGroupSize)
{
    return (uint32_t)((domain - 1) / workGroupSize + 1llu);
}