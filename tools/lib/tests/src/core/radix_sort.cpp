#include <algorithm>
#include <catch2/catch_all.hpp>
#include <voxcom/core/radix_sort.h>

using namespace voxcom;

TEMPLATE_TEST_CASE("radix_sort<T>", "[RadixSort]", uint32_t, uint64_t)
{
    auto keys = GENERATE(take(10, chunk(1000, random<TestType>(1, std::numeric_limits<TestType>::max()))));
    auto originalKeys = keys;
    voxcom::radix_sort(std::span(keys));

    for (const auto key : originalKeys) {
        REQUIRE(std::find(std::begin(keys), std::end(keys), key) != std::end(keys));
    }
    REQUIRE(std::is_sorted(std::begin(keys), std::end(keys)));
}

TEMPLATE_TEST_CASE("radix_sort_key_values<T, T>", "[RadixSort]", uint32_t, uint64_t)
{
    const auto keys = GENERATE(take(10, chunk(1000, random<TestType>(1, std::numeric_limits<TestType>::max()))));
    const auto values = GENERATE(take(1, chunk(1000, random<TestType>(1, std::numeric_limits<TestType>::max()))));

    std::vector<std::pair<TestType, TestType>> items;
    for (size_t i = 0; i < keys.size(); ++i) {
        items.push_back({ keys[i], values[i] });
    }
    auto originalItems = items;
    voxcom::radix_sort_key_values(std::span(items));

    for (const auto& item : originalItems) {
        REQUIRE(std::find(std::begin(items), std::end(items), item) != std::end(items));
    }
    REQUIRE(std::is_sorted(std::begin(items), std::end(items)));
}
