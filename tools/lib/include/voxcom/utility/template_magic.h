#pragma once
#include <type_traits>

template <int Min, int Max, typename F>
static void templateForLoop(F f, int runTimeEnd)
{
    if (runTimeEnd == Min)
        f(std::integral_constant<int, Min>());
    if constexpr (Min != Max)
        templateForLoop<Min + 1, Max, F>(f, runTimeEnd);
}
template <int Min, int Max, typename F>
static void templateForLoop(F f)
{
    f(std::integral_constant<int, Min>());
    if constexpr (Min != Max)
        templateForLoop<Min + 1, Max, F>(f);
}