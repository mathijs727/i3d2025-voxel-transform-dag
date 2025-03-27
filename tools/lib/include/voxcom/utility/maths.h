#pragma once

namespace voxcom {

template <typename T>
inline T divideRoundUp(T a, T b)
{
    return (a + b - 1) / b;
}

}