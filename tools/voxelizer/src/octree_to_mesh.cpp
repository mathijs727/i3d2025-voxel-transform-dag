#include "octree_to_mesh.h"
#include <array>
#include <cassert>
#include <glm/vec3.hpp>
#include <stack>
#include <voxcom/core/mesh.h>
#include <voxcom/voxel/morton.h>
#include <voxcom/voxel/structure.h>

glm::ivec3 childIdxOffset(int idx)
{
    return {
        (idx & 0b001) ? 1 : 0,
        (idx & 0b010) ? 1 : 0,
        (idx & 0b100) ? 1 : 0,
    };
}

[[maybe_unused]] static void addCube(voxcom::Mesh& mesh, const glm::ivec3& xyz, int size)
{
    constexpr static std::array cubePositions {
        glm::vec3 { +0, +0, +0 }, glm::vec3 { +0, +0, +1 }, glm::vec3 { +0, +1, +1 }, glm::vec3 { +0, +1, +0 },
        glm::vec3 { +1, +0, +1 }, glm::vec3 { +1, +0, +0 }, glm::vec3 { +1, +1, +0 }, glm::vec3 { +1, +1, +1 },
        glm::vec3 { +0, +0, +0 }, glm::vec3 { +1, +0, +0 }, glm::vec3 { +1, +0, +1 }, glm::vec3 { +0, +0, +1 },
        glm::vec3 { +1, +1, +0 }, glm::vec3 { +0, +1, +0 }, glm::vec3 { +0, +1, +1 }, glm::vec3 { +1, +1, +1 },
        glm::vec3 { +0, +0, +0 }, glm::vec3 { +0, +1, +0 }, glm::vec3 { +1, +1, +0 }, glm::vec3 { +1, +0, +0 },
        glm::vec3 { +0, +1, +1 }, glm::vec3 { +0, +0, +1 }, glm::vec3 { +1, +0, +1 }, glm::vec3 { +1, +1, +1 }
    };
    constexpr static std::array quads {
        glm::ivec4 { 0, 1, 2, 3 },
        glm::ivec4 { 4, 5, 6, 7 },
        glm::ivec4 { 8, 9, 10, 11 },
        glm::ivec4 { 12, 13, 14, 15 },
        glm::ivec4 { 16, 17, 18, 19 },
        glm::ivec4 { 20, 21, 22, 23 }
    };

    const unsigned offset = (unsigned)mesh.positions.size();
    for (auto& q : quads) {
        //mesh.triangles.push_back(glm::uvec3 { q.x, q.y, q.z } + offset);
        //mesh.triangles.push_back(glm::uvec3 { q.x, q.z, q.w } + offset);
        mesh.quads.push_back(glm::uvec4 { q.x, q.y, q.z, q.w } + offset);
    }

    for (int i = 0; i < 24; ++i) {
        mesh.positions.push_back(glm::vec3(xyz) + cubePositions[i] * (float)size);
    }
}

[[maybe_unused]] static void addCubeFace(voxcom::Mesh& mesh, const glm::ivec3& xyz, int size, int axis, int positive)
{
    constexpr static std::array cubePositions {
        glm::vec3 { +0, +0, +0 }, glm::vec3 { +0, +0, +1 }, glm::vec3 { +0, +1, +1 }, glm::vec3 { +0, +1, +0 },
        glm::vec3 { +1, +0, +1 }, glm::vec3 { +1, +0, +0 }, glm::vec3 { +1, +1, +0 }, glm::vec3 { +1, +1, +1 },
        glm::vec3 { +0, +0, +0 }, glm::vec3 { +1, +0, +0 }, glm::vec3 { +1, +0, +1 }, glm::vec3 { +0, +0, +1 },
        glm::vec3 { +1, +1, +0 }, glm::vec3 { +0, +1, +0 }, glm::vec3 { +0, +1, +1 }, glm::vec3 { +1, +1, +1 },
        glm::vec3 { +0, +0, +0 }, glm::vec3 { +0, +1, +0 }, glm::vec3 { +1, +1, +0 }, glm::vec3 { +1, +0, +0 },
        glm::vec3 { +0, +1, +1 }, glm::vec3 { +0, +0, +1 }, glm::vec3 { +1, +0, +1 }, glm::vec3 { +1, +1, +1 }
    };

    const unsigned offset = (unsigned)mesh.positions.size();
    //mesh.triangles.push_back(glm::uvec3 { 0, 1, 2 } + offset);
    //mesh.triangles.push_back(glm::uvec3 { 0, 2, 3 } + offset);
    mesh.quads.push_back(glm::uvec4 { 0, 1, 2, 3 } + offset);

    const int start = (axis * 2 + positive) * 4;
    for (int i = start; i < start + 4; ++i) {
        mesh.positions.push_back(glm::vec3(xyz) + cubePositions[i] * (float)size);
    }
}

template <typename Attribute>
std::vector<voxcom::Mesh> octreeToMesh(const voxcom::EditStructure<Attribute, uint32_t>& tree)
{
    std::vector<voxcom::Mesh> meshes;
    voxcom::Mesh singleMesh;

    struct NodeRef {
        uint32_t level;
        uint32_t idx;
        glm::ivec3 xyz { 0 };
    };
    std::stack<NodeRef> stack;
    stack.push({ .level = tree.rootLevel, .idx = 0 });
    while (!stack.empty()) {
        const NodeRef nodeRef = stack.top();
        stack.pop();

        // Subgrid level.
        if (nodeRef.level == tree.subGridLevel) {
            if (!tree.subGrids.empty()) {
                const auto subGrid = tree.subGrids[nodeRef.idx];
                for (uint64_t voxelIdx = 0, voxelBit = 1; voxelIdx < 64; ++voxelIdx, voxelBit <<= 1) {
                    if (!(subGrid.bitmask & voxelBit))
                        continue;

                    if constexpr (std::is_same_v<Attribute, voxcom::RGB>) {
                        voxcom::Mesh cubeMesh;
                        cubeMesh.diffuseBaseColor = subGrid.attributes[voxelIdx];
                        addCube(cubeMesh, nodeRef.xyz + glm::ivec3(voxcom::morton_decode64<3>(voxelIdx)), 1);
                        meshes.push_back(cubeMesh);
                    } else {
                        if constexpr (true) {
                            addCube(singleMesh, nodeRef.xyz + glm::ivec3(voxcom::morton_decode64<3>(voxelIdx)), 1);
                        } else {
                            // Don't generate faces inside octree.
                            const auto voxelPos = nodeRef.xyz + glm::ivec3(voxcom::morton_decode64<3>(voxelIdx));
                            if (!tree.get(voxelPos + glm::ivec3(-1, 0, 0)))
                                addCubeFace(singleMesh, voxelPos, 1, 0, 0);
                            if (!tree.get(voxelPos + glm::ivec3(+1, 0, 0)))
                                addCubeFace(singleMesh, voxelPos, 1, 0, 1);

                            if (!tree.get(voxelPos + glm::ivec3(0, -1, 0)))
                                addCubeFace(singleMesh, voxelPos, 1, 1, 0);
                            if (!tree.get(voxelPos + glm::ivec3(0, +1, 0)))
                                addCubeFace(singleMesh, voxelPos, 1, 1, 1);

                            if (!tree.get(voxelPos + glm::ivec3(0, 0, -1)))
                                addCubeFace(singleMesh, voxelPos, 1, 2, 0);
                            if (!tree.get(voxelPos + glm::ivec3(0, 0, +1)))
                                addCubeFace(singleMesh, voxelPos, 1, 2, 1);
                        }
                    }
                }
            } else {
                addCube(singleMesh, nodeRef.xyz, 4);
            }
        } else {
            // Homogeneous inner node.
            const auto nodeWidth = 1 << nodeRef.level;
            const auto& node = tree.nodesPerLevel[nodeRef.level][nodeRef.idx];
            // if (node.isHomogeneous && node.isFilled)
            //     addCube(singleMesh, nodeRef.xyz, nodeWidth);

            // Regular inner node.
            assert(nodeRef.level > 0);
            const auto halfNodeWidth = nodeWidth >> 1;
            for (int childIdx = 0; childIdx < 8; childIdx++) {
                const auto child = node.children[childIdx];
                if (child != voxcom::EditNode<uint32_t>::EmptyChild) {
                    const auto childXyz = nodeRef.xyz + childIdxOffset(childIdx) * halfNodeWidth;
                    stack.push({ .level = nodeRef.level - 1, .idx = child, .xyz = childXyz });
                }
            }
        }
    }

    if (!singleMesh.positions.empty()) {
        singleMesh.diffuseBaseColor = voxcom::RGB { 200, 200, 200 };
        meshes.push_back(singleMesh);
    }

    return meshes;
}

template <typename Attribute>
std::vector<voxcom::Mesh> octreeToMesh(const voxcom::VoxelGrid<Attribute>& grid)
{
    std::vector<voxcom::Mesh> meshes;
    voxcom::Mesh singleMesh;

    const auto addCubeToMesh = [&](const glm::ivec3& voxel) {
        if constexpr (false) {
            addCube(singleMesh, voxel, 1);
        } else {
            // Don't generate faces inside octree.
            if (!grid.get(voxel + glm::ivec3(-1, 0, 0)))
                addCubeFace(singleMesh, voxel, 1, 0, 0);
            if (!grid.get(voxel + glm::ivec3(+1, 0, 0)))
                addCubeFace(singleMesh, voxel, 1, 0, 1);

            if (!grid.get(voxel + glm::ivec3(0, -1, 0)))
                addCubeFace(singleMesh, voxel, 1, 1, 0);
            if (!grid.get(voxel + glm::ivec3(0, +1, 0)))
                addCubeFace(singleMesh, voxel, 1, 1, 1);

            if (!grid.get(voxel + glm::ivec3(0, 0, -1)))
                addCubeFace(singleMesh, voxel, 1, 2, 0);
            if (!grid.get(voxel + glm::ivec3(0, 0, +1)))
                addCubeFace(singleMesh, voxel, 1, 2, 1);
        }
    };

    for (int z = 0; z < (int)grid.resolution; ++z) {
        for (int y = 0; y < (int)grid.resolution; ++y) {
            for (int x = 0; x < (int)grid.resolution; ++x) {
                const glm::ivec3 voxel { x, y, z };
                if (grid.get(voxel))
                    addCube(singleMesh, voxel, 1);
            }
        }
    }
    if (!singleMesh.positions.empty()) {
        singleMesh.diffuseBaseColor = voxcom::RGB { 200, 200, 200 };
        meshes.push_back(singleMesh);
    }

    return meshes;
}

template std::vector<voxcom::Mesh> octreeToMesh(const voxcom::EditStructure<void, uint32_t>& tree);
template std::vector<voxcom::Mesh> octreeToMesh(const voxcom::EditStructure<voxcom::RGB, uint32_t>& tree);

template std::vector<voxcom::Mesh> octreeToMesh(const voxcom::VoxelGrid<void>& grid);
// template std::vector<voxcom::Mesh> octreeToMesh(const voxcom::VoxelGrid<voxcom::RGB>& grid);
