#include <catch2/catch_all.hpp>
#include <limits>
#include <voxcom/core/cuckoo_hash_table.h>

using namespace voxcom;

TEMPLATE_TEST_CASE("CuckooHashTable<T, T>", "[CuckooHashTable]", uint32_t, uint64_t)
{
    const auto keys = GENERATE(take(10, chunk(40, random<TestType>(1, std::numeric_limits<TestType>::max()))));
    const auto values = GENERATE(take(10, chunk(20, random<TestType>(1, std::numeric_limits<TestType>::max()))));

    CuckooHashTable<TestType, TestType, 0> cuckooHashTable { std::span(keys).subspan(0, values.size()), values };

    for (size_t i = 0; i < values.size(); ++i) {
        TestType foundValue;
        const bool found = cuckooHashTable.find(keys[i], foundValue);
        REQUIRE(found);
        REQUIRE(foundValue == values[i]);
    }
    for (size_t i = values.size(); i < keys.size(); ++i) {
        TestType foundValue;
        const bool found = cuckooHashTable.find(keys[i], foundValue);
        REQUIRE(!found);
    }
}
