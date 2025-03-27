#pragma once
#include <vector>

namespace voxcom {

template <typename T>
size_t sizeOfVector(const std::vector<T>& v)
{
    return v.size() * sizeof(T);
}

}