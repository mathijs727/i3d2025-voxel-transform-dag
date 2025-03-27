#pragma once
#include <source_location>
#include <stdexcept>
#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <spdlog/spdlog.h>
DISABLE_WARNINGS_POP()

namespace voxcom {

#if 0
[[maybe_unused]] inline void assert_always(bool)
{
}
#else

#ifdef __CUDACC__
__host__ __device__ void assert_always_impl(bool condition, const char* file, int line)
{
    if (!condition) {
        printf("%s:%i\n", file, line);
#ifdef __CUDA_ARCH__
        __threadfence();
        asm("trap;");
#elif defined(WIN32)
        __debugbreak();
#else
        throw std::runtime_error("exception failed");
#endif
    }
}
#define assert_always(cond) assert_always_impl(cond, __FILE__, __LINE__)
#else
[[maybe_unused]] static void assert_always(bool condition, const std::source_location location = std::source_location::current())
{
    if (!condition) {
        spdlog::error("{}:{}", location.file_name(), location.line());
#if WIN32
        __debugbreak();
#else
        throw std::runtime_error("exception failed");
#endif
    }
}
[[maybe_unused]] static void assert_equal_float(float lhs, float rhs, float margin = 0.00001f, const std::source_location location = std::source_location::current())
{
    if (std::abs(lhs - rhs) > margin) {
        spdlog::error("{}:{}", location.file_name(), location.line());
#if WIN32
        __debugbreak();
#else
        throw std::runtime_error("exception failed");
#endif
    }
}
#endif

#endif

}
