#include <catch2/catch_all.hpp>
#include <voxcom/voxel/dag.h>
#include <voxcom/voxel/octree.h>

using namespace voxcom;

TEST_CASE("Convert Octree<void> to a DAG", "[Octree][DAG]")
{
    SECTION("16x16x16 Random")
    {
        constexpr unsigned resolution = 16;
        constexpr unsigned resolution3 = resolution * resolution * resolution;

        // Value for each pixel.
        const auto voxelValues = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));

        Octree<void> octree { resolution };
        REQUIRE(octree.nodesPerLevel.size() == 5);
        REQUIRE(octree.resolution == resolution);
        REQUIRE(octree.treeType == TreeType::Tree);

        for (unsigned i = 0; i < resolution3; i++) {
            if (!voxelValues[i])
                continue;

            const glm::uvec3 voxel = morton_decode32<3>(i);
            octree.set(voxel);
        }

        const auto dag = octreeToDAG(std::move(octree));

        // Lowest levels should be empty because we store 4x4x4 regions as a single leaf.
        REQUIRE(dag.treeType == TreeType::DAG);
        REQUIRE(dag.nodesPerLevel[0].empty());
        REQUIRE(dag.nodesPerLevel[1].empty());
        REQUIRE(dag.nodesPerLevel[2].empty());

        for (unsigned i = 0; i < resolution3; i++) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            CAPTURE(voxel.x, voxel.y, voxel.z, voxelValues[i]);
            REQUIRE(dag.get(voxel) == (bool)voxelValues[i]);
        }
    }

    SECTION("16x16x16 Homogeneous")
    {
        constexpr unsigned resolution = 16;
        constexpr unsigned resolution3 = resolution * resolution * resolution;

        // Value for each pixel.
        Octree<void> octree { resolution };
        REQUIRE(octree.nodesPerLevel.size() == 5);
        REQUIRE(octree.resolution == resolution);
        REQUIRE(octree.treeType == TreeType::Tree);

        for (unsigned i = 0; i < resolution3; i++) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            octree.set(voxel);
        }

        const auto octreeNodeCount = octree.computeNodeCount();
        const auto dag = octreeToDAG(octree);

        // Lowest levels should be empty because we store 4x4x4 regions as a single leaf.
        REQUIRE(dag.treeType == TreeType::DAG);
        REQUIRE(dag.computeNodeCount() < octreeNodeCount);
        REQUIRE(dag.nodesPerLevel[0].empty());
        REQUIRE(dag.nodesPerLevel[1].empty());
        REQUIRE(dag.nodesPerLevel[2].empty());

        for (unsigned i = 0; i < resolution3; i++) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            REQUIRE(dag.get(voxel) == true);
        }
    }
}

TEST_CASE("Convert Octree<void>'s to a MultiDAG", "[Octree, DAG]")
{
    SECTION("16x16x16 Random")
    {
        constexpr unsigned resolution = 16;
        constexpr unsigned resolution3 = resolution * resolution * resolution;

        // Value for each pixel.
        const auto voxelValues = GENERATE(take(3, chunk(resolution3, random<uint32_t>(0, 1))));

        Octree<void> octree { resolution };
        REQUIRE(octree.nodesPerLevel.size() == 5);
        REQUIRE(octree.resolution == resolution);
        REQUIRE(octree.treeType == TreeType::Tree);

        for (unsigned i = 0; i < resolution3; i++) {
            if (!voxelValues[i])
                continue;

            const glm::uvec3 voxel = morton_decode32<3>(i);
            octree.set(voxel);
        }

        std::vector<Octree<void>> octrees;
        octrees.push_back(octree);
        octrees.push_back(octree);
        const auto multiDAG = octreeToDAG<void>(octrees);

        // Lowest levels should be empty because we store 4x4x4 regions as a single leaf.
        REQUIRE(multiDAG.treeType == TreeType::MultiDAG);
        REQUIRE(multiDAG.nodesPerLevel[0].empty());
        REQUIRE(multiDAG.nodesPerLevel[1].empty());
        REQUIRE(multiDAG.nodesPerLevel[2].empty());
        REQUIRE(multiDAG.nodesPerLevel[3].size() < 2 * octree.nodesPerLevel[3].size());
        REQUIRE(multiDAG.nodesPerLevel[4].size() == 2);

        for (unsigned i = 0; i < resolution3; i++) {
            const glm::uvec3 voxel = morton_decode32<3>(i);
            CAPTURE(voxel.x, voxel.y, voxel.z, voxelValues[i]);
            REQUIRE(multiDAG.get(voxel, 0) == (bool)voxelValues[i]);
            REQUIRE(multiDAG.get(voxel, 1) == (bool)voxelValues[i]);
        }
    }
}
