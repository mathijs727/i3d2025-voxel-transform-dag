#pragma once
#include "voxcom/utility/my_cuda.h"
#include <cstdint>
#include <functional>

namespace voxcom {

// Hash combine function was copied from:
// https://stackoverflow.com/questions/28367913/how-to-stdhash-an-unordered-stdpair
template <typename T>
inline void hash_combine(std::size_t& seed, T key) noexcept
{
    std::hash<T> hasher {};
    seed ^= hasher(key) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
template <typename T, typename Hasher>
HOST_DEVICE inline void hash_combine(auto& seed, T key) noexcept
{
    seed ^= Hasher()(key) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename HashA, typename HashB>
struct PairHasher {
    inline size_t operator()(const auto& pair) const
    {
        size_t seed = 0;
        voxcom::hash_combine(seed, HashA()(std::get<0>(pair)));
        voxcom::hash_combine(seed, HashB()(std::get<1>(pair)));
        return seed;
    }
};

}

namespace std {
template <typename A, typename B>
struct hash<std::pair<A, B>> {
    inline size_t operator()(const std::pair<A, B>& pair) const
    {
        size_t seed = 0;
        voxcom::hash_combine<A>(seed, std::get<0>(pair));
        voxcom::hash_combine<B>(seed, std::get<1>(pair));
        return seed;
    }
};
}
