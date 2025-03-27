#pragma once
#include "typedefs.h"
#include <bit> // bitcast
#include <cmath>
#include <ostream>

#ifndef __CUDACC__
template <typename T>
inline T max(T a, T b) { return (((a) > (b)) ? (a) : (b)); }
template <typename T>
inline T min(T a, T b) { return (((a) < (b)) ? (a) : (b)); }
#endif
template <typename T>
HOST_DEVICE T constexpr constexpr_max(T a, T b)
{
    return (((a) > (b)) ? (a) : (b));
}
template <typename T>
HOST_DEVICE T constexpr constexpr_min(T a, T b) { return (((a) < (b)) ? (a) : (b)); }

template <typename T>
constexpr HOST_DEVICE T make_vector2(const decltype(T::x) x, const decltype(T::x) y)
{
    T t {};
    t.x = x;
    t.y = y;
    return t;
}

template <typename T>
constexpr HOST_DEVICE T make_vector3(const decltype(T::x) x, const decltype(T::x) y, const decltype(T::x) z)
{
    T t {};
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}

template <typename T, typename U>
HOST_DEVICE constexpr auto lerp(T a, T b, U f) { return a * (1 - f) + b * f; }
template <typename T>
HOST_DEVICE constexpr T squared(T f) { return f * f; }
template <typename T>
HOST_DEVICE constexpr T clamp(T f, T a, T b) { return f < a ? a : f > b ? b
                                                                        : f; }

#define BIN_VECTOR2_OP(T, OP)                                                                                               \
    HOST_DEVICE constexpr T operator OP(const decltype(T::x) a, const T& b) { return make_vector2<T>(a OP b.x, a OP b.y); } \
    HOST_DEVICE constexpr T operator OP(const T& a, const decltype(T::x) b) { return make_vector2<T>(a.x OP b, a.y OP b); } \
    HOST_DEVICE constexpr T operator OP(const T& a, const T& b) { return make_vector2<T>(a.x OP b.x, a.y OP b.y); }

#define VECTOR2_OP(T)                                                                                                                                \
    BIN_VECTOR2_OP(T, +)                                                                                                                             \
    BIN_VECTOR2_OP(T, -)                                                                                                                             \
    BIN_VECTOR2_OP(T, *)                                                                                                                             \
    BIN_VECTOR2_OP(T, /)                                                                                                                             \
    HOST_DEVICE constexpr bool operator<(const T& a, const T& b) { return a.x < b.x && a.y < b.y; }                                                  \
    HOST_DEVICE constexpr bool operator>(const T& a, const T& b) { return a.x > b.x && a.y > b.y; }                                                  \
    HOST_DEVICE constexpr bool operator<=(const T& a, const T& b) { return a.x <= b.x && a.y <= b.y; }                                               \
    HOST_DEVICE constexpr bool operator>=(const T& a, const T& b) { return a.x >= b.x && a.y >= b.y; }                                               \
    HOST_DEVICE constexpr decltype(T::x) dot(const T& a, const T& b) { return a.x * b.x + a.y * b.y; }                                               \
    HOST_DEVICE constexpr decltype(T::x) length_squared(const T& a) { return dot(a, a); }                                                            \
    HOST_DEVICE constexpr decltype(T::x) max(const T& v) { return constexpr_max(v.x, v.y); }                                                         \
    HOST_DEVICE constexpr decltype(T::x) min(const T& v) { return constexpr_min(v.x, v.y); }                                                         \
    HOST_DEVICE constexpr T max(const T& a, const T& b) { return make_vector2<T>(constexpr_max(a.x, b.x), constexpr_max(a.y, b.y)); }                \
    HOST_DEVICE constexpr T min(const T& a, const T& b) { return make_vector2<T>(constexpr_min(a.x, b.x), constexpr_min(a.y, b.y)); }                \
    HOST_DEVICE constexpr T clamp_vector(const T& x, const T& a, const T& b) { return make_vector2<T>(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y)); } \
    HOST_DEVICE constexpr T clamp_vector(const T& x, decltype(T::x) a, decltype(T::x) b) { return make_vector2<T>(clamp(x.x, a, b), clamp(x.y, a, b)); }

#define VECTOR2_FLOAT_OP(T)                                                                            \
    HOST_DEVICE constexpr T operator-(T a) { return make_vector2<T>(-a.x, -a.y); }                     \
    HOST_DEVICE auto length(T a) { return (decltype(T::x))sqrt(dot(a, a)); }                           \
    HOST_DEVICE T normalize(T a) { return (1 / length(a)) * a; }                                       \
    HOST_DEVICE constexpr T abs(const T& v) { return make_vector2<T>(std::abs(v.x), std::abs(v.y)); }; \
    HOST_DEVICE T ceil(const T& v) { return make_vector2<T>(std::ceil(v.x), std::ceil(v.y)); };        \
    HOST_DEVICE T round(const T& v) { return make_vector2<T>(std::round(v.x), std::round(v.y)); };     \
    HOST_DEVICE uint2 round_to_uint(const T& v) { return make_uint2(uint32(std::round(v.x)), uint32(std::round(v.y))); };

VECTOR2_OP(float2)
VECTOR2_OP(double2)
VECTOR2_OP(uint2)
VECTOR2_OP(int2)

VECTOR2_FLOAT_OP(float2)
VECTOR2_FLOAT_OP(double2)

#define BIN_VECTOR3_OP(T, OP)                                                                                                         \
    HOST_DEVICE constexpr T operator OP(const decltype(T::x) a, const T& b) { return make_vector3<T>(a OP b.x, a OP b.y, a OP b.z); } \
    HOST_DEVICE constexpr T operator OP(const T& a, const decltype(T::x) b) { return make_vector3<T>(a.x OP b, a.y OP b, a.z OP b); } \
    HOST_DEVICE constexpr T operator OP(const T& a, const T& b) { return make_vector3<T>(a.x OP b.x, a.y OP b.y, a.z OP b.z); }

#define VECTOR3_OP(T)                                                                                                                                                      \
    BIN_VECTOR3_OP(T, +)                                                                                                                                                   \
    BIN_VECTOR3_OP(T, -)                                                                                                                                                   \
    BIN_VECTOR3_OP(T, *)                                                                                                                                                   \
    BIN_VECTOR3_OP(T, /)                                                                                                                                                   \
    inline std::ostream& operator<<(std::ostream& out, const T& a)                                                                                                         \
    {                                                                                                                                                                      \
        out << "(" << a.x << ", " << a.y << ", " << a.z << ")";                                                                                                            \
        return out;                                                                                                                                                        \
    }                                                                                                                                                                      \
    HOST_DEVICE constexpr bool operator<(const T& a, const T& b) { return a.x < b.x && a.y < b.y && a.z < b.z; }                                                           \
    HOST_DEVICE constexpr bool operator>(const T& a, const T& b) { return a.x > b.x && a.y > b.y && a.z > b.z; }                                                           \
    HOST_DEVICE constexpr bool operator<=(const T& a, const T& b) { return a.x <= b.x && a.y <= b.y && a.z <= b.z; }                                                       \
    HOST_DEVICE constexpr bool operator>=(const T& a, const T& b) { return a.x >= b.x && a.y >= b.y && a.z >= b.z; }                                                       \
    HOST_DEVICE constexpr decltype(T::x) dot(const T& a, const T& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }                                                         \
    HOST_DEVICE constexpr decltype(T::x) length_squared(const T& a) { return dot(a, a); }                                                                                  \
    HOST_DEVICE constexpr decltype(T::x) max(const T& v) { return constexpr_max(v.x, constexpr_max(v.y, v.z)); }                                                           \
    HOST_DEVICE constexpr decltype(T::x) min(const T& v) { return constexpr_min(v.x, constexpr_min(v.y, v.z)); }                                                           \
    HOST_DEVICE constexpr T max(const T& a, const T& b) { return make_vector3<T>(constexpr_max(a.x, b.x), constexpr_max(a.y, b.y), constexpr_max(a.z, b.z)); }             \
    HOST_DEVICE constexpr T min(const T& a, const T& b) { return make_vector3<T>(constexpr_min(a.x, b.x), constexpr_min(a.y, b.y), constexpr_min(a.z, b.z)); }             \
    HOST_DEVICE constexpr T clamp_vector(const T& x, const T& a, const T& b) { return make_vector3<T>(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y), clamp(x.z, a.z, b.z)); } \
    HOST_DEVICE constexpr T clamp_vector(const T& x, decltype(T::x) a, decltype(T::x) b) { return make_vector3<T>(clamp(x.x, a, b), clamp(x.y, a, b), clamp(x.z, a, b)); }

#define VECTOR3_FLOAT_OP(T)                                                                                           \
    HOST_DEVICE constexpr T operator-(T a) { return make_vector3<T>(-a.x, -a.y, -a.z); }                              \
    HOST_DEVICE auto length(T a) { return (decltype(T::x))sqrt(dot(a, a)); }                                          \
    HOST_DEVICE T normalize(T a) { return (1 / length(a)) * a; }                                                      \
    HOST_DEVICE constexpr T abs(const T& v) { return make_vector3<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z)); }; \
    HOST_DEVICE T ceil(const T& v) { return make_vector3<T>(std::ceil(v.x), std::ceil(v.y), std::ceil(v.z)); };       \
    HOST_DEVICE T round(const T& v) { return make_vector3<T>(std::round(v.x), std::round(v.y), std::round(v.z)); };   \
    HOST_DEVICE uint3 round_to_uint(const T& v) { return make_uint3(uint32(std::round(v.x)), uint32(std::round(v.y)), uint32(std::round(v.z))); };

#define VECTOR3_INT_OP(T) \
    BIN_VECTOR3_OP(T, ^)  \
    BIN_VECTOR3_OP(T, &)  \
    BIN_VECTOR3_OP(T, |)  \
    BIN_VECTOR3_OP(T, <<) \
    BIN_VECTOR3_OP(T, >>)

VECTOR3_OP(float3)
VECTOR3_OP(double3)
VECTOR3_OP(uint3)
VECTOR3_OP(int3)

VECTOR3_FLOAT_OP(float3)
VECTOR3_FLOAT_OP(double3)

VECTOR3_INT_OP(uint3)
VECTOR3_INT_OP(int3)

HOST_DEVICE constexpr float2 make_float2(float a) { return make_vector2<float2>(a, a); }
HOST_DEVICE constexpr float2 make_float2(uint2 a) { return make_vector2<float2>(float(a.x), float(a.y)); }

HOST_DEVICE constexpr float3 make_float3(float a) { return make_vector3<float3>(a, a, a); }
HOST_DEVICE constexpr float3 make_float3(double3 d) { return make_vector3<float3>(float(d.x), float(d.y), float(d.z)); }
HOST_DEVICE constexpr float3 make_float3(const uint3& a) { return make_vector3<float3>(float(a.x), float(a.y), float(a.z)); };
HOST_DEVICE constexpr float3 make_float3(const int3& a) { return make_vector3<float3>(float(a.x), float(a.y), float(a.z)); };
HOST_DEVICE constexpr float3 make_float3(const float4& a) { return make_vector3<float3>(a.x, a.y, a.z); };

HOST_DEVICE constexpr double3 make_double3(double a) { return make_vector3<double3>(a, a, a); }
HOST_DEVICE constexpr double3 make_double3(const uint3& a) { return make_vector3<double3>(double(a.x), double(a.y), double(a.z)); }
HOST_DEVICE constexpr double3 make_double3(const float3& f) { return make_vector3<double3>(double(f.x), double(f.y), double(f.z)); }

HOST double length(uint3 a) { return std::sqrt(dot(a, a)); }
// HOST_DEVICE constexpr uint3 operator<<(const uint3& v, const uint32 shift) { return make_vector3<uint3>(v.x << shift, v.y << shift, v.z << shift); }
// HOST_DEVICE constexpr uint3 operator>>(const uint3& v, const uint32 shift) { return make_vector3<uint3>(v.x >> shift, v.y >> shift, v.z >> shift); }
HOST_DEVICE constexpr bool operator==(const uint3& a, const uint3& b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z); }
HOST_DEVICE constexpr bool operator!=(const uint3& a, const uint3& b) { return !(a == b); }
HOST_DEVICE constexpr uint3 make_uint3(const uint32& v) { return make_vector3<uint3>(v, v, v); }
HOST_DEVICE constexpr uint3 make_uint3(const float3& v) { return make_vector3<uint3>((unsigned)v.x, (unsigned)v.y, (unsigned)v.z); }
HOST_DEVICE constexpr uint3 make_uint3(const int3& v) { return make_vector3<uint3>(v.x, v.y, v.z); }
HOST_DEVICE constexpr uint3 make_uint3(const uint4& v) { return make_vector3<uint3>(v.x, v.y, v.z); }

HOST_DEVICE constexpr int3 make_int3(const uint3& v) { return make_vector3<int3>(v.x, v.y, v.z); }
HOST_DEVICE constexpr int3 make_int3(const float3& v) { return make_vector3<int3>((int)v.x, (int)v.y, (int)v.z); }

HOST_DEVICE constexpr uint3 truncate(const float3& f) { return make_vector3<uint3>(uint32(f.x), uint32(f.y), uint32(f.z)); }
HOST_DEVICE constexpr uint2 truncate(const float2& f) { return make_vector2<uint2>(uint32(f.x), uint32(f.y)); }
HOST_DEVICE constexpr int3 truncateSigned(const float3& f) { return make_vector3<int3>(int32(f.x), int32(f.y), int32(f.z)); }
HOST_DEVICE constexpr int2 truncateSigned(const float2& f) { return make_vector2<int2>(int32(f.x), int32(f.y)); }

inline __host__ __device__ float3 cross(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

HOST_DEVICE float3 uint3_as_float3(const uint3& v)
{
#ifdef __CUDA_ARCH__
    return make_float3(
        __uint_as_float(v.x),
        __uint_as_float(v.y),
        __uint_as_float(v.z));
#else
    return make_float3(
        std::bit_cast<float>(v.x),
        std::bit_cast<float>(v.y),
        std::bit_cast<float>(v.z));
#endif
}
HOST_DEVICE float3 int3_as_float3(const int3& v)
{
#ifdef __CUDA_ARCH__
    return make_float3(
        __int_as_float(v.x),
        __int_as_float(v.y),
        __int_as_float(v.z));
#else
    return make_float3(
        std::bit_cast<float>(v.x),
        std::bit_cast<float>(v.y),
        std::bit_cast<float>(v.z));
#endif
}
HOST_DEVICE uint3 float3_as_uint3(const float3& v)
{
#ifdef __CUDA_ARCH__
    return make_uint3(
        __float_as_uint(v.x),
        __float_as_uint(v.y),
        __float_as_uint(v.z));
#else
    return make_uint3(
        std::bit_cast<uint32_t>(v.x),
        std::bit_cast<uint32_t>(v.y),
        std::bit_cast<uint32_t>(v.z));
#endif
}
HOST_DEVICE int3 float3_as_int3(const float3& v)
{
#ifdef __CUDA_ARCH__
    return make_int3(
        __float_as_int(v.x),
        __float_as_int(v.y),
        __float_as_int(v.z));
#else
    return make_int3(
        std::bit_cast<int32_t>(v.x),
        std::bit_cast<int32_t>(v.y),
        std::bit_cast<int32_t>(v.z));
#endif
}

namespace std {
template <>
struct hash<uint3> {
    inline constexpr std::size_t operator()(const uint3& k) const noexcept
    {
        return size_t(k.x) + 81799 * size_t(k.y) + 38351 * size_t(k.z);
    }
};
}