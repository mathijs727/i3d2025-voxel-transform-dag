#pragma once
#include <vector>

template <typename T>
struct attribute_vector {
    using type = std::vector<T>;
};
template <>
struct attribute_vector<void> {
    using type = int; // Random type that doesn't occupy a lot of memory.
};
template <typename T>
using attribute_vector_t = typename attribute_vector<T>::type;
