#include "voxcom/attributes/color.h"
#include <catch2/catch_all.hpp>
#include <vector>
#include <voxcom/attributes/dado2016_optimized.h>
#include <voxcom/attributes/dado2016_original.h>
#include <voxcom/attributes/perfect_spatial_hashing.h>

using namespace voxcom;

TEST_CASE("Compress attributes with Dado2016", "[Dado2016]")
{
    constexpr unsigned numAttributes = 8192;
    const auto rawAttributes = GENERATE(take(3, chunk(numAttributes * 3, random<uint32_t>(0, 255))));

    std::vector<RGB> attributes;
    for (unsigned i = 0; i < numAttributes; i++)
        attributes.push_back({ (uint8_t)rawAttributes[i * 3 + 0], (uint8_t)rawAttributes[i * 3 + 1], (uint8_t)rawAttributes[i * 3 + 2] });

    // Make sure some neighboring attributes are the same (to trigger large block compression).
    for (int i = 0; i < 100; i++)
        attributes[100 + i] = attributes[100 + i - 1];

    // Some more of the same but with some space in-between.
    attributes[501] = attributes[500];
    attributes[502] = attributes[501];
    attributes[503] = attributes[502];
    //attributes[504] = attributes[503];
    //attributes[505] = attributes[504];
    //attributes[506] = attributes[505];
    attributes[506] = attributes[503];
    attributes[507] = attributes[506];

    AttributesDado2016_Original attributesDado2016 { attributes };
    for (unsigned i = 0; i < numAttributes; i++) {
        const auto encodedAttribute = attributesDado2016.getAttribute(i);
        CHECK(encodedAttribute.r == attributes[i].r);
        CHECK(encodedAttribute.g == attributes[i].g);
        CHECK(encodedAttribute.b == attributes[i].b);
    }
    (void)attributesDado2016.getSize();
}

TEST_CASE("Compress attributes with Dado2016_Optimized", "[Dado2016_Optimized]")
{
    constexpr unsigned numAttributes = 8192;
    const auto rawAttributes = GENERATE(take(3, chunk(numAttributes * 3, random<uint32_t>(0, 255))));

    std::vector<RGB> attributes;
    for (unsigned i = 0; i < numAttributes; i++)
        attributes.push_back({ (uint8_t)rawAttributes[i * 3 + 0], (uint8_t)rawAttributes[i * 3 + 1], (uint8_t)rawAttributes[i * 3 + 2] });

    // Make sure some neighboring attributes are the same (to trigger large block compression).
    for (int i = 0; i < 100; i++)
        attributes[100 + i] = attributes[100 + i - 1];

    // Some more of the same but with some space in-between.
    attributes[501] = attributes[500];
    attributes[502] = attributes[501];
    attributes[503] = attributes[502];
    //attributes[504] = attributes[503];
    //attributes[505] = attributes[504];
    //attributes[506] = attributes[505];
    attributes[506] = attributes[503];
    attributes[507] = attributes[506];

    AttributesDado2016_Optimized attributesDado2016 { attributes };
    for (unsigned i = 0; i < numAttributes; i++) {
        const auto encodedAttribute = attributesDado2016.getAttribute(i);
        CHECK(encodedAttribute.r == attributes[i].r);
        CHECK(encodedAttribute.g == attributes[i].g);
        CHECK(encodedAttribute.b == attributes[i].b);
    }
    (void)attributesDado2016.getSize();
}

TEST_CASE("Compress attributes with PerfectSpatialHashing", "[PerfectSpatialHashing]")
{
    constexpr unsigned numAttributes = 8192;
    const auto rawAttributes = GENERATE(take(3, chunk(numAttributes * 3, random<uint32_t>(0, 255))));
    const auto voxelFilled = GENERATE(take(3, chunk(numAttributes, random<uint32_t>(0, 3))));

    std::vector<RGB> attributes;
    std::vector<uint32_t> attributePositions;
    for (unsigned i = 0; i < numAttributes; i++) {
        // Skip over some voxels. The hash is supposed to be sparse!
        if (voxelFilled[i] != 0)
            continue;
        attributes.push_back({ (uint8_t)rawAttributes[i * 3 + 0], (uint8_t)rawAttributes[i * 3 + 1], (uint8_t)rawAttributes[i * 3 + 2] });
        attributePositions.push_back(i);
    }

    AttributesPerfectSpatialHashing attributesPerfectSpatialHashing { attributes, attributePositions };
    for (unsigned i = 0; i < attributes.size(); i++) {
        const auto encodedAttribute = attributesPerfectSpatialHashing.getAttribute(attributePositions[i]);
        const auto& attribute = attributes[i];
        CHECK(encodedAttribute.r == attribute.r);
        CHECK(encodedAttribute.g == attribute.g);
        CHECK(encodedAttribute.b == attribute.b);
    }
}
