#include "voxcom/voxel/voxelize.h"
#include "voxcom/core/bounds.h"
#include "voxcom/core/mesh.h"
#include "voxcom/utility/error_handling.h"
#include "voxcom/utility/hash.h"
#include "voxcom/voxel/morton.h"
#include "voxcom/voxel/structure.h"
#include "voxcom/voxel/voxel_grid.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <execution>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#pragma warning(disable : 4459) // declaration of 'relation' hides global declaration (tbb_profiling.h:253)
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#pragma warning(default : 4459)
#define GLM_ENABLE_EXPERIMENTAL 1
#include <glm/gtx/component_wise.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <spdlog/spdlog.h>
DISABLE_WARNINGS_POP()

using namespace voxcom;

namespace voxcom {

constexpr float epsilon = 0.000001f;

// Two pass algorithm that first voxelizes the scene at a very low resolution to create a coarse (top level) tree, and then
// refines each leaf by voxelizing again at a higher resolution. The latter step is performed in parallel. To prevent running
// out of memory for very large scenes, the tree is stored to disk every once in a while.
template <typename Attribute, bool conservative>
EditStructure<Attribute, uint32_t> voxelizeHierarchical(std::span<const Mesh> meshes, unsigned resolution)
{
    const unsigned ParallelDepth = std::max((int)std::bit_width(resolution - 1) - 14, 1);
    constexpr unsigned BatchSize = 256;

    EditStructure<Attribute, uint32_t> out { resolution };

    // Voxelize into a low resolution octree.
    spdlog::info("Voxelizing at a low resolution (single threaded)");
    VoxelGrid<void> topLevelGrid { 1u << ParallelDepth };
    // Make voxelizer think grid is bigger such that it scales the mesh accordingly.
    const float scale = (float)topLevelGrid.resolution / (float)resolution;
    for (const auto& mesh : meshes) {
        voxelizeMeshNaive<true>(topLevelGrid, mesh, glm::vec3(0), scale);
    }
    // Gather leaf voxels of the low resolution octree.
    std::vector<glm::uvec3> workTiles;
    // Scale by the size of the size that each tile represents in the output octree.
    const auto tileSize = 1u << (out.rootLevel - ParallelDepth);
    for (uint32_t z = 0; z < topLevelGrid.resolution; ++z) {
        for (uint32_t y = 0; y < topLevelGrid.resolution; ++y) {
            for (uint32_t x = 0; x < topLevelGrid.resolution; ++x) {
                if (topLevelGrid.get(glm::ivec3(x, y, z)))
                    workTiles.push_back(glm::uvec3(x, y, z) * tileSize);
            }
        }
    }
    spdlog::info("Num tiles: {}", workTiles.size());

    // Compute bounding boxes of the meshes.
    std::vector<voxcom::Bounds> meshesBounds(meshes.size());
    std::transform(std::begin(meshes), std::end(meshes), std::begin(meshesBounds), [](const Mesh& mesh) { return mesh.computeBounds(); });

    spdlog::info("Voxelizing at a high resolution (multithreaded)");
    // Compute partial structures in batches to reduce memory usage.
    for (size_t batchStartIdx = 0; batchStartIdx < workTiles.size(); batchStartIdx += BatchSize) {
        const size_t batchEndIdx = std::min(batchStartIdx + BatchSize, workTiles.size());

        spdlog::info("Voxelizing tiles [{}, {}]", batchStartIdx, batchEndIdx);
        std::vector<EditStructure<Attribute, uint32_t>> partialStructures(batchEndIdx - batchStartIdx);
        // for (size_t tileIdx = batchStartIdx; tileIdx < batchEndIdx; ++tileIdx) {
        tbb::parallel_for(tbb::blocked_range<size_t>(batchStartIdx, batchEndIdx),
            [&](tbb::blocked_range<size_t> localRange) {
                for (size_t tileIdx = localRange.begin(); tileIdx < localRange.end(); ++tileIdx) {
                    // Compute the bounds of the tile.
                    const glm::vec3 tileStart = workTiles[tileIdx];
                    const glm::vec3 tileEnd = tileStart + (float)tileSize;

                    // Loop over all meshes and voxelize them if they intersect the tile.
                    EditStructure<Attribute, uint32_t> partialTree(resolution >> ParallelDepth);
                    for (size_t meshIdx = 0; meshIdx < meshes.size(); meshIdx++) {
                        const auto& mesh = meshes[meshIdx];
                        const auto meshBounds = meshesBounds[meshIdx];

                        // Skip meshes whose bounds do not overlap the tile.
                        if (glm::any(glm::lessThan(meshBounds.upper, tileStart)) || glm::any(glm::greaterThan(meshBounds.lower, tileEnd)))
                            continue;

                        // TODO(Mathijs): the optimized voxelization code seems to be broken for triangles that are (partially?) outside the voxelization region.
                        voxelizeMeshNaive<conservative>(partialTree, mesh, -tileStart);
                    }

                    partialTree.toDAG();

                    // Add subtree to the final structure.
                    partialStructures[tileIdx - batchStartIdx] = std::move(partialTree);
                }
            });

        // Merge into the output structure.
        spdlog::info("Merging tiles [{}, {}]", batchStartIdx, batchEndIdx);
        for (size_t tileIdx = batchStartIdx; tileIdx < batchEndIdx; ++tileIdx) {
            out.set(workTiles[tileIdx], partialStructures[tileIdx - batchStartIdx]);
        }
    }

    spdlog::info("Convert to DAG");
    out.toDAG();

    return out;
}

template <int PrimaryAxis>
struct OtherAxis;
template <>
struct OtherAxis<0> {
    static constexpr int a1 = 1;
    static constexpr int a2 = 2;
};
template <>
struct OtherAxis<1> {
    static constexpr int a1 = 2;
    static constexpr int a2 = 0;
};
template <>
struct OtherAxis<2> {
    static constexpr int a1 = 0;
    static constexpr int a2 = 1;
};

struct TriangleProps {
    glm::uvec3 indices;
    glm::vec3 v[3]; // vertices
    glm::vec3 e[3]; // edges
    glm::vec3 n; // normal
    glm::vec3 abs_n; // abs(normal)
    glm::vec3 aabbMin, aabbMax; // AABB of the triangle.
};

template <int PrimaryAxis, bool Conservative>
bool testTrianglePlane(const TriangleProps& tri, const glm::vec3& p, const glm::vec3& delta_p)
{
    // Two axis spanning the plane.
    constexpr int A1 = OtherAxis<PrimaryAxis>::a1;
    constexpr int A2 = OtherAxis<PrimaryAxis>::a2;

    // Test that the AABBs of the triangle and voxels overlap.
    const glm::vec2 delta_p_xy { delta_p[A1], delta_p[A2] };
    const glm::vec2 p_xy_i { p[A1], p[A2] };
    const glm::vec2 tri_aabbMin_xy { tri.aabbMin[A1], tri.aabbMin[A2] };
    const glm::vec2 tri_aabbMax_xy { tri.aabbMax[A1], tri.aabbMax[A2] };
    if (glm::any(glm::lessThan(p_xy_i + delta_p_xy, tri_aabbMin_xy) | glm::greaterThan(p_xy_i, tri_aabbMax_xy)))
        return false;

    // Test that the critical points of the AABB lie on the inside of the triangle edges.
    if (tri.abs_n[PrimaryAxis] > 0) {
        for (int i = 0; i < 3; i++) {
            const glm::vec2 n_xy_ei = glm::vec2(-tri.e[i][A2], tri.e[i][A1]) * (tri.n[PrimaryAxis] >= 0 ? +1.0f : -1.0f);
            const glm::vec2 v_xy_i { tri.v[i][A1], tri.v[i][A2] };
            float d_xy_ei;
            if constexpr (Conservative) {
                d_xy_ei = -glm::dot(n_xy_ei, v_xy_i) + std::max(0.0f, delta_p_xy[0] * n_xy_ei[0]) + std::max(0.0f, delta_p_xy[1] * n_xy_ei[1]);
            } else {
                const glm::vec2 abs_n_xy_ei = glm::abs(n_xy_ei);
                const int max_comp_n_xy_ei = abs_n_xy_ei[0] > abs_n_xy_ei[1] ? 0 : 1;
                d_xy_ei = glm::dot(n_xy_ei, 0.5f * delta_p_xy - v_xy_i) + 0.5f * delta_p_xy[max_comp_n_xy_ei] * abs_n_xy_ei[max_comp_n_xy_ei];
            }
            const float distFromEdge = glm::dot(p_xy_i, n_xy_ei) + d_xy_ei;
            if (distFromEdge < 0)
                return false;
        }
    }
    return true;
}

static glm::vec3 computeClosestPointBarycentricCoordinates(const TriangleProps& tri, const glm::vec3& p)
{
    // Compute the barycentric coordinates of the point projected onto the triangle plane.
    // Copied from here:
    // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    const glm::vec3 v0 = tri.v[1] - tri.v[0], v1 = tri.v[2] - tri.v[0], v2 = p - tri.v[0];
    const float d00 = glm::dot(v0, v0);
    const float d01 = glm::dot(v0, v1);
    const float d11 = glm::dot(v1, v1);
    const float d20 = glm::dot(v2, v0);
    const float d21 = glm::dot(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    const float u = 1.0f - v - w;

    // The projected point may lie outside of the triangle so we need to clamp the barycentric coordinates.
    // Copied from answer by DJohn & klappvisor on StackOverflow:
    // https://stackoverflow.com/questions/14467296/barycentric-coordinate-clamping-on-3d-triangle
    if (u < 0) {
        float t = glm::dot(p - tri.v[1], tri.v[2] - tri.v[1]) / glm::dot(tri.v[2] - tri.v[1], tri.v[2] - tri.v[1]);
        t = glm::clamp(t, 0.0f, 1.0f);
        return glm::vec3(0.0f, 1.0f - t, t);
    } else if (v < 0) {
        float t = glm::dot(p - tri.v[2], tri.v[0] - tri.v[2]) / glm::dot(tri.v[0] - tri.v[2], tri.v[0] - tri.v[2]);
        t = glm::clamp(t, 0.0f, 1.0f);
        return glm::vec3(t, 0.0f, 1.0f - t);
    } else if (w < 0) {
        float t = glm::dot(p - tri.v[0], tri.v[1] - tri.v[0]) / glm::dot(tri.v[1] - tri.v[0], tri.v[1] - tri.v[0]);
        t = glm::clamp(t, 0.0f, 1.0f);
        return glm::vec3(1.0f - t, t, 0.0f);
    } else {
        return glm::vec3(u, v, w);
    }
}

template <typename Target>
static void fillVoxel(Target& target, const Mesh& mesh, const TriangleProps& tri, const glm::ivec3& voxel)
{
    if constexpr (std::is_void_v<typename Target::Attribute>) {
        target.set(voxel);
    } else {
        const glm::vec3 barycentricCoordinates = computeClosestPointBarycentricCoordinates(tri, glm::vec3(voxel) + 0.5f);
        glm::vec3 diffuseColor { 0.0f };
        if (mesh.texCoords.empty() || !mesh.pDiffuseTexture) {
            diffuseColor = mesh.diffuseBaseColor;
        } else {
            // clang-format off
            const glm::vec2 texCoord = \
                barycentricCoordinates.x * mesh.texCoords[tri.indices[0]] +
                barycentricCoordinates.y * mesh.texCoords[tri.indices[1]] +
                barycentricCoordinates.z * mesh.texCoords[tri.indices[2]];
            // clang-format on
            diffuseColor = mesh.pDiffuseTexture->sampleBilinear(texCoord);
        }

        glm::vec3 lightMapIrradiance { 1.0f };
        if (!mesh.lightMapCoords.empty() && mesh.pLightMapTexture) {
            // clang-format off
            const glm::vec2 lightMapCoord = \
                barycentricCoordinates.x * mesh.lightMapCoords[tri.indices[0]] +
                barycentricCoordinates.y * mesh.lightMapCoords[tri.indices[1]] +
                barycentricCoordinates.z * mesh.lightMapCoords[tri.indices[2]];
            // clang-format on
            lightMapIrradiance = mesh.pDiffuseTexture->sampleBilinear(lightMapCoord);
        }

        target.set(voxel, make_rgb(diffuseColor * lightMapIrradiance));
    }
}

// Copy of implementation from my Pandora renderer:
//  https://github.com/mathijs727/pandora/blob/master/projects/pandora/src/svo/mesh_to_voxel.cpp
//
// Naive mesh voxelization
// Based on: http://research.michael-schwarz.com/publ/files/vox-siga10.pdf
// Outline:
// For each triangle:
//   For each voxel in triangles AABB:
//     Test intersection between voxel and triangle
//
// Conservative will use conservative voxelization (any voxel that a triangle touches is filled).
// With conservative off this function will perform 6-separating voxelization which is not conservative but is watertight.
template <bool conservative, typename Target>
void voxelizeMeshNaive(Target& target, const Mesh& mesh, const glm::vec3& translation, float scale)
{
    const glm::ivec3 gridResolution { (int)target.resolution };
    const glm::ivec3 gridResolutionMinusOne { (int)(target.resolution - 1) };

    // World space extent of a voxel
    const glm::vec3 delta_p { 1.0f };

    for (size_t t = 0; t < mesh.triangles.size(); t++) {
        TriangleProps tri {};
        tri.indices = mesh.triangles[t];
        tri.v[0] = mesh.positions[tri.indices[0]] * scale + translation;
        tri.v[1] = mesh.positions[tri.indices[1]] * scale + translation;
        tri.v[2] = mesh.positions[tri.indices[2]] * scale + translation;
        tri.e[0] = tri.v[1] - tri.v[0];
        tri.e[1] = tri.v[2] - tri.v[1];
        tri.e[2] = tri.v[0] - tri.v[2];
        tri.n = glm::cross(tri.e[0], tri.e[1]);
        // Skip degenerate triangles whose edges lie in a line.
        if (tri.n.x == 0.0f && tri.n.y == 0.0f && tri.n.z == 0.0f)
            continue;
        tri.abs_n = glm::abs(tri.n);
        tri.aabbMin = glm::min(tri.v[0], glm::min(tri.v[1], tri.v[2]));
        tri.aabbMax = glm::max(tri.v[0], glm::max(tri.v[1], tri.v[2]));

        // Triangle bounds
        const glm::vec3 tBoundsMin = glm::min(tri.v[0], glm::min(tri.v[1], tri.v[2]));
        const glm::vec3 tBoundsMax = glm::max(tri.v[0], glm::max(tri.v[1], tri.v[2]));
        const glm::ivec3 tBoundsMinVoxel = glm::clamp(glm::ivec3(glm::floor(tBoundsMin - epsilon)), glm::ivec3(0), gridResolutionMinusOne); // Round down
        const glm::ivec3 tBoundsMaxVoxel = glm::clamp(glm::ivec3(glm::ceil(tBoundsMax + epsilon)), glm::ivec3(0), gridResolution); // Round up

        float d1, d2;
        if constexpr (conservative) {
            // Critical point on a corner of voxel (26-separating).
            const glm::vec3 c {
                tri.n.x > 0 ? delta_p.x : 0,
                tri.n.y > 0 ? delta_p.y : 0,
                tri.n.z > 0 ? delta_p.z : 0
            };
            d1 = glm::dot(tri.n, c - tri.v[0]);
            d2 = glm::dot(tri.n, (delta_p - c) - tri.v[0]);
        } else {
            // 6-separating surface voxelization.
            // argmax(abs(n_x), abs(n_y), abs(n_z))
            const int maxCompN = tri.abs_n.x > tri.abs_n.y ? (tri.abs_n.x > tri.abs_n.z ? 0 : 2) : (tri.abs_n.y > tri.abs_n.z ? 1 : 2);
            d1 = glm::dot(tri.n, 0.5f * delta_p - tri.v[0]) + 0.5f * delta_p[maxCompN] * tri.abs_n[maxCompN];
            d2 = glm::dot(tri.n, 0.5f * delta_p - tri.v[0]) - 0.5f * delta_p[maxCompN] * tri.abs_n[maxCompN];
        }

        // For each voxel in the triangles AABB
        for (int z = tBoundsMinVoxel.z; z < tBoundsMaxVoxel.z; z++) {
            for (int y = tBoundsMinVoxel.y; y < tBoundsMaxVoxel.y; y++) {
                for (int x = tBoundsMinVoxel.x; x < tBoundsMaxVoxel.x; x++) {
                    // Intersection test
                    const glm::ivec3 ip { x, y, z };
                    const glm::vec3 p { ip };

                    const bool planeIntersect = ((glm::dot(tri.n, p) + d1) * (glm::dot(tri.n, p) + d2)) <= 0;
                    if (!planeIntersect)
                        continue;

                    bool triangleIntersect2D = true;
                    triangleIntersect2D &= testTrianglePlane<0, conservative>(tri, p, delta_p);
                    triangleIntersect2D &= testTrianglePlane<1, conservative>(tri, p, delta_p);
                    triangleIntersect2D &= testTrianglePlane<2, conservative>(tri, p, delta_p);

                    if (triangleIntersect2D) {
                        assert(x >= tBoundsMinVoxel.x);
                        assert(y >= tBoundsMinVoxel.y);
                        assert(z >= tBoundsMinVoxel.z);
                        fillVoxel(target, mesh, tri, ip);
                    }
                }
            }
        }
    }
}

template <bool conservative, typename Target>
void voxelizeMeshOptimized(Target& target, const Mesh& mesh, const glm::vec3& translation, float scale)
{
    const glm::vec3 fGridResolution { (float)target.resolution };
    const glm::ivec3 gridResolution { (int)target.resolution };
    const glm::ivec3 gridResolutionMinusOne { (int)(target.resolution - 1) };

    // World space extent of a voxel
    const glm::vec3 delta_p { 1.0f };
    const glm::vec2 delta_p_xy { delta_p.x, delta_p.y };
    const glm::vec2 delta_p_xz { delta_p.x, delta_p.z };
    const glm::vec2 delta_p_yz { delta_p.y, delta_p.z };

    for (size_t t = 0; t < mesh.triangles.size(); t++) {
        TriangleProps tri {};
        tri.indices = mesh.triangles[t];
        if (tri.indices[0] >= mesh.positions.size() || tri.indices[1] >= mesh.positions.size() || tri.indices[2] >= mesh.positions.size())
            continue;
        tri.v[0] = mesh.positions[tri.indices[0]] * scale + translation;
        tri.v[1] = mesh.positions[tri.indices[1]] * scale + translation;
        tri.v[2] = mesh.positions[tri.indices[2]] * scale + translation;
        tri.e[0] = tri.v[1] - tri.v[0];
        tri.e[1] = tri.v[2] - tri.v[1];
        tri.e[2] = tri.v[0] - tri.v[2];
        tri.n = glm::cross(tri.e[0], tri.e[1]);
        // Skip degenerate triangles whose edges lie in a line.
        if (tri.n.x == 0.0f && tri.n.y == 0.0f && tri.n.z == 0.0f)
            continue;
        tri.abs_n = glm::abs(tri.n);
        tri.aabbMin = glm::min(tri.v[0], glm::min(tri.v[1], tri.v[2]));
        tri.aabbMax = glm::max(tri.v[0], glm::max(tri.v[1], tri.v[2]));

        // Skip triangles that lie outside the target grid.
        if (glm::any(glm::lessThan(tri.aabbMax, glm::vec3(0.0f)) | glm::greaterThan(tri.aabbMin, fGridResolution)))
            continue;

        float d1, d2;
        if constexpr (conservative) {
            // Critical point on a corner of voxel (26-separating).
            const glm::vec3 c {
                tri.n.x > 0 ? delta_p.x : 0,
                tri.n.y > 0 ? delta_p.y : 0,
                tri.n.z > 0 ? delta_p.z : 0
            };
            d1 = glm::dot(tri.n, c - tri.v[0]);
            d2 = glm::dot(tri.n, (delta_p - c) - tri.v[0]);
        } else {
            // 6-separating surface voxelization.
            // argmax(abs(n_x), abs(n_y), abs(n_z))
            const int maxCompN = tri.abs_n.x > tri.abs_n.y ? (tri.abs_n.x > tri.abs_n.z ? 0 : 2) : (tri.abs_n.y > tri.abs_n.z ? 1 : 2);
            d1 = glm::dot(tri.n, 0.5f * delta_p - tri.v[0]) + 0.5f * delta_p[maxCompN] * tri.abs_n[maxCompN];
            d2 = glm::dot(tri.n, 0.5f * delta_p - tri.v[0]) - 0.5f * delta_p[maxCompN] * tri.abs_n[maxCompN];
        }

        const int max_comp_abs_n = tri.abs_n.x > tri.abs_n.y ? (tri.abs_n.x > tri.abs_n.z ? 0 : 2) : (tri.abs_n.y > tri.abs_n.z ? 1 : 2);
        const float plane_d = glm::dot(tri.n, tri.v[0]); // Distance of triangle plane to origin.

        // Triangle bounds
        const glm::vec3 tBoundsMin = glm::min(tri.v[0], glm::min(tri.v[1], tri.v[2]));
        const glm::vec3 tBoundsMax = glm::max(tri.v[0], glm::max(tri.v[1], tri.v[2]));
        const glm::ivec3 tBoundsMinVoxel = glm::clamp(glm::ivec3(glm::floor(tBoundsMin - epsilon)), glm::ivec3(0), gridResolutionMinusOne); // Round down
        const glm::ivec3 tBoundsMaxVoxel = glm::clamp(glm::ivec3(glm::ceil(tBoundsMax + epsilon)), glm::ivec3(0), gridResolution); // Round up

        if (max_comp_abs_n == 0) { // x is the dominant axis.
            for (int z = tBoundsMinVoxel.z; z < tBoundsMaxVoxel.z; z++) {
                for (int y = tBoundsMinVoxel.y; y < tBoundsMaxVoxel.y; y++) {
                    const glm::vec3 p0 { 0, y, z };
                    if (!testTrianglePlane<0, true>(tri, p0, delta_p))
                        continue;

                    // Corners of the column with min/max depth of triangle plane.
                    const glm::vec2 c = glm::max(glm::vec2(0.0f), glm::sign(glm::vec2(tri.n.y, tri.n.z)));
                    const glm::vec2 p_yz { p0.y, p0.z };
                    const glm::vec2 min_p_yz = p_yz + delta_p_yz * c;
                    const glm::vec2 max_p_yz = p_yz + delta_p_yz * (1.0f - c);
                    float min_p_x = (plane_d - tri.n.y * min_p_yz[0] - tri.n.z * min_p_yz[1]) / tri.n.x;
                    float max_p_x = (plane_d - tri.n.y * max_p_yz[0] - tri.n.z * max_p_yz[1]) / tri.n.x;
                    if (min_p_x > max_p_x)
                        std::swap(min_p_x, max_p_x);
                    const int minX = std::clamp((int)(min_p_x - epsilon), 0, gridResolution.x - 1);
                    const int maxX = std::clamp((int)(max_p_x + 1 + epsilon), 0, gridResolution.x);
                    for (int x = minX; x < maxX; x++) {
                        const glm::vec3 p { x, y, z };

                        const bool planeIntersect = ((glm::dot(tri.n, p) + d1) * (glm::dot(tri.n, p) + d2)) <= 0;
                        if (planeIntersect && testTrianglePlane<1, false>(tri, p, delta_p) && testTrianglePlane<2, false>(tri, p, delta_p)) {
                            fillVoxel(target, mesh, tri, { x, y, z });
                        }
                    }
                }
            }
        } else if (max_comp_abs_n == 1) { // y is the dominant axis.
            for (int z = tBoundsMinVoxel.z; z < tBoundsMaxVoxel.z; z++) {
                for (int x = tBoundsMinVoxel.x; x < tBoundsMaxVoxel.x; x++) {
                    // Test overlap between the projection of the triangle and voxel column on the XZ-plane
                    const glm::vec3 p0 { x, 0, z };
                    if (!testTrianglePlane<1, true>(tri, p0, delta_p))
                        continue;

                    // Corners of the column with min/max depth of triangle plane.
                    const glm::vec2 c = glm::max(glm::vec2(0.0f), glm::sign(glm::vec2(tri.n.x, tri.n.z)));
                    const glm::vec2 p_xz { p0.x, p0.z };
                    const glm::vec2 min_p_xz = p_xz + delta_p_xz * c;
                    const glm::vec2 max_p_xz = p_xz + delta_p_xz * (1.0f - c);
                    float min_p_y = (plane_d - tri.n.x * min_p_xz[0] - tri.n.z * min_p_xz[1]) / tri.n.y;
                    float max_p_y = (plane_d - tri.n.x * max_p_xz[0] - tri.n.z * max_p_xz[1]) / tri.n.y;
                    if (min_p_y > max_p_y)
                        std::swap(min_p_y, max_p_y);
                    const int minY = std::clamp((int)(min_p_y - epsilon), 0, gridResolution.y - 1);
                    const int maxY = std::clamp((int)(max_p_y + 1 + epsilon), 0, gridResolution.y);
                    for (int y = minY; y < maxY; y++) {
                        const glm::vec3 p { x, y, z };

                        const bool planeIntersect = ((glm::dot(tri.n, p) + d1) * (glm::dot(tri.n, p) + d2)) <= 0;
                        if (planeIntersect && testTrianglePlane<0, false>(tri, p, delta_p) && testTrianglePlane<2, false>(tri, p, delta_p)) {
                            fillVoxel(target, mesh, tri, { x, y, z });
                        }
                    }
                }
            }
        } else if (max_comp_abs_n == 2) { // z is the dominant axis.
            for (int y = tBoundsMinVoxel.y; y < tBoundsMaxVoxel.y; y++) {
                for (int x = tBoundsMinVoxel.x; x < tBoundsMaxVoxel.x; x++) {
                    // Test overlap between the projection of the triangle and voxel column on the XY-plane
                    const glm::vec3 p0 { x, y, 0 };
                    if (!testTrianglePlane<2, true>(tri, p0, delta_p))
                        continue;

                    // Corners of the column with min/max depth of triangle plane.
                    const glm::vec2 c = glm::max(glm::vec2(0.0f), glm::sign(glm::vec2(tri.n.x, tri.n.y)));
                    const glm::vec2 p_xy { p0.x, p0.y };
                    const glm::vec2 min_p_xy = p_xy + delta_p_xy * c;
                    const glm::vec2 max_p_xy = p_xy + delta_p_xy * (1.0f - c);
                    float min_p_z = (plane_d - tri.n.x * min_p_xy[0] - tri.n.y * min_p_xy[1]) / tri.n.z;
                    float max_p_z = (plane_d - tri.n.x * max_p_xy[0] - tri.n.y * max_p_xy[1]) / tri.n.z;
                    if (min_p_z > max_p_z)
                        std::swap(min_p_z, max_p_z);
                    const int minZ = std::clamp((int)(min_p_z - epsilon), 0, gridResolution.z - 1);
                    const int maxZ = std::clamp((int)(max_p_z + 1 + epsilon), 0, gridResolution.z);
                    for (int z = minZ; z < maxZ; z++) {
                        const glm::vec3 p { x, y, z };

                        const bool planeIntersect = ((glm::dot(tri.n, p) + d1) * (glm::dot(tri.n, p) + d2)) <= 0;
                        if (planeIntersect && testTrianglePlane<0, false>(tri, p, delta_p) && testTrianglePlane<1, false>(tri, p, delta_p)) {
                            fillVoxel(target, mesh, tri, { x, y, z });
                        }
                    }
                }
            }
        }
    }
}

struct SolidNode {
    static constexpr uint32_t EmptyChild = std::numeric_limits<uint32_t>::max();
    uint32_t firstChildIdx = EmptyChild;
    bool filled { false };
};
struct SolidEditStructure {
public:
    std::vector<std::vector<SolidNode>> nodesPerLevel;
    std::vector<uint64_t> subGrids;

    static constexpr uint32_t subGridLevel = 2; // lvl0 = 1x1x1, lvl1 = 2x2x2, lvl2 = 4x4x4 subgrids
    unsigned resolution = 0;
    uint32_t rootLevel;

public:
    SolidEditStructure(unsigned resolution)
        : resolution(resolution)
    {
        if (std::popcount(resolution) != 1)
            spdlog::warn("SolidEditStructure has resolution that is not a power of 2");

        // Total number of levels including subgrid levels.
        const uint32_t numLevels = std::bit_width(resolution - 1) + 1;
        nodesPerLevel.resize(numLevels);
        // Construct empty root node.
        nodesPerLevel.back().emplace_back();
        rootLevel = numLevels - 1;
    }

    uint64_t& createSubGrid(const glm::ivec3& voxel)
    {
        uint32_t nodeIdx = 0;
        for (uint32_t level = rootLevel; level > subGridLevel; --level) {
            SolidNode& node = nodesPerLevel[level][nodeIdx];

            // Bit pattern: y|x
            const int childLevel = level - 1;
            const glm::uvec3 childOffset = (voxel >> childLevel) & 0b1;
            const uint32_t childIdx = morton_encode32(childOffset);

            if (node.firstChildIdx == SolidNode::EmptyChild) {
                if (childLevel == subGridLevel) {
                    // Create sub grid if it does not exist yet.
                    node.firstChildIdx = (uint32_t)subGrids.size();
                    subGrids.resize(subGrids.size() + 8);
                } else {
                    // Create child node if it does not exist yet.
                    auto& childLevelNodes = nodesPerLevel[childLevel];
                    node.firstChildIdx = (uint32_t)childLevelNodes.size();
                    childLevelNodes.resize(childLevelNodes.size() + 8);
                }
            }
            nodeIdx = node.firstChildIdx + childIdx;
        }

        return subGrids[nodeIdx];
    }
};

struct SolidNodeReference {
    // Order is important:
    // Sort by y/z first, only then by x!
    uint32_t y, z, x;
    uint32_t nodeIdx;
    bool flipped { false };

    inline glm::uvec3 getPosition() const { return glm::uvec3(x, y, z); }
    inline void setPosition(const glm::uvec3& pos)
    {
        x = pos.x;
        y = pos.y;
        z = pos.z;
    }

    constexpr auto operator<=>(const SolidNodeReference&) const = default;
};
static void collectAllNodesRecurse(const SolidEditStructure& target, const glm::uvec3& currentPath, uint32_t level, uint32_t nodeIdx, std::vector<std::vector<SolidNodeReference>>& outNodeReferences)
{
    outNodeReferences[level].push_back({ .y = currentPath.y, .z = currentPath.z, .x = currentPath.x, .nodeIdx = nodeIdx });

    if (level == target.subGridLevel)
        return;

    const uint32_t childLevel = level - 1;
    const auto& node = target.nodesPerLevel[level][nodeIdx];
    if (node.firstChildIdx != SolidNode::EmptyChild) {
        for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
            const auto childPathOffset = morton_decode32<3>(childIdx);
            const auto childPath = currentPath + (childPathOffset << childLevel);
            const auto child = node.firstChildIdx + childIdx;
            collectAllNodesRecurse(target, childPath, childLevel, child, outNodeReferences);
        }
    }
}

static void sparseFill(SolidEditStructure& target)
{
    // Traverse the tree and collect pointers to the neighbours in the +x direction.
    spdlog::info("Collect all inner nodes");
    std::vector<std::vector<SolidNodeReference>> nodesReferences;
    nodesReferences.resize(target.nodesPerLevel.size());
    collectAllNodesRecurse(target, glm::uvec3(0), target.rootLevel, 0, nodesReferences);

    spdlog::info("Sort nodes by y/z/x");
    for (auto& levelNodeReferences : nodesReferences) {
        std::sort(std::execution::par_unseq, std::begin(levelNodeReferences), std::end(levelNodeReferences));
    }

    spdlog::info("Propagate subgrids along x-links");
    std::vector<SolidNodeReference> parentFlips;
    {
        const auto nodeSize = 1 << target.subGridLevel;
        const auto parentNodeSize = 2 * nodeSize;

        // Cache so we don't need to call morton_encode/morton_decode many times.
        std::array<uint64_t, 16> masksLastX, masks;
        for (uint32_t i = 0; i < 16; ++i) {
            const auto yz = morton_decode32<2>(i);
            const auto bit0Idx = morton_encode32(glm::uvec3(0, yz[0], yz[1]));
            const auto bit1Idx = morton_encode32(glm::uvec3(1, yz[0], yz[1]));
            const auto bit2Idx = morton_encode32(glm::uvec3(2, yz[0], yz[1]));
            const auto bit3Idx = morton_encode32(glm::uvec3(3, yz[0], yz[1]));
            masksLastX[i] = ((uint64_t)1) << bit3Idx;
            masks[i] = 0;
            masks[i] |= ((uint64_t)1) << bit0Idx;
            masks[i] |= ((uint64_t)1) << bit1Idx;
            masks[i] |= ((uint64_t)1) << bit2Idx;
            masks[i] |= ((uint64_t)1) << bit3Idx;
        }

        // Parent nodes that need to be flipped because the child nodes last x bit was set.
        // Loop over all the string in +x direction and propagate fill information.
        std::array<uint64_t, 16> flipMask;
        SolidNodeReference previous = nodesReferences[target.subGridLevel][0];
        for (const auto& nodeRef : nodesReferences[target.subGridLevel]) {
            auto& subGrid = target.subGrids[nodeRef.nodeIdx];

            if (nodeRef.x != previous.x + nodeSize || nodeRef.y != previous.y || nodeRef.z != previous.z) {
                // Start a new string.
                std::fill(std::begin(flipMask), std::end(flipMask), 0);

                // At the end of a link all subgrids with the same parent should have all x=3 bits set to the same value.
                // We use one of these subgrids to propagate this information up towards the parent node.
                const auto previousPosition = previous.getPosition();
                const auto previousParentPosition = previousPosition & (~(parentNodeSize - 1u));
                if (previousPosition == previousParentPosition + glm::uvec3(nodeSize, 0, 0) && (target.subGrids[previous.nodeIdx] & masksLastX[0])) {
                    auto previousParentRef = previous;
                    previousParentRef.setPosition(previousParentPosition);
                    parentFlips.push_back(previousParentRef);
                }
            }

            // Apply bit flip.
            for (uint32_t i = 0; i < 16; ++i)
                subGrid ^= flipMask[i];

            // Flip next nodes along string if last x bit is set.
            for (uint32_t i = 0; i < 16; ++i)
                flipMask[i] = (subGrid & masksLastX[i]) ? masks[i] : 0x0;

            // Continue to the next node.
            previous = nodeRef;
        }
    }

    for (uint32_t level = target.subGridLevel + 1; level < target.rootLevel; ++level) {
        auto& nodeRefs = nodesReferences[level];
        auto& outLevelNodes = target.nodesPerLevel[level];
        assert_always(nodeRefs.size() == outLevelNodes.size());
        assert_always(parentFlips.size() <= outLevelNodes.size());

        spdlog::info("Propagate to next coarser level");
        // Assume both nodeRefs and parentFlips are sorted.
        auto iter = std::begin(parentFlips);
        for (auto& nodeRef : nodeRefs) {
            iter = std::lower_bound(iter, std::end(parentFlips), nodeRef);
            if (iter != std::end(parentFlips) && iter->getPosition() == nodeRef.getPosition())
                nodeRef.flipped = true;
        }
        parentFlips.clear();

        spdlog::info("Propagate along x-links");
        // Loop over all the string in +x direction and propagate fill information.
        const auto nodeSize = 1u << level;
        const auto parentNodeSize = 2 * nodeSize;
        bool flip = false;
        SolidNodeReference previous = nodeRefs[0];
        for (auto& nodeRef : nodeRefs) {
            auto& outNode = outLevelNodes[nodeRef.nodeIdx];

            if (nodeRef.x != previous.x + nodeSize || nodeRef.y != previous.y || nodeRef.z != previous.z) {
                // Start a new node string.
                flip = nodeRef.flipped;

                // At the end of a link all nodes with the same parent should have all x=3 bits set to the same value.
                // We use one of these nodes to propagate this information up towards the parent node.
                const auto previousPosition = previous.getPosition();
                const auto previousParentPosition = previousPosition & (~(parentNodeSize - 1u));
                if (previous.flipped && previousPosition == previousParentPosition + glm::uvec3(nodeSize, 0, 0)) {
                    auto previousParentRef = previous;
                    previousParentRef.setPosition(previousParentPosition);
                    parentFlips.push_back(previousParentRef);
                }
            } else {
                // Set fully filled bit. We store this information even if the node is not homogeneous (aka it has children).
                // This is required for the rest of the algorithm (see figure 7 of the paper).
                outNode.filled ^= flip;

                // Continuation of existing node string.
                flip = (nodeRef.flipped ^= flip);
            }

            // Continue to the next node.
            previous = nodeRef;
        }
    }

    spdlog::info("Propagate to next finer level");
    for (uint32_t level = target.rootLevel; level > target.subGridLevel; --level) {
        const uint32_t childLevel = level - 1;
        const auto& levelNodes = target.nodesPerLevel[level];

        if (childLevel == target.subGridLevel) {
            for (const auto& node : levelNodes) {
                if (node.firstChildIdx != node.EmptyChild) {
                    for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
                        auto& subGrid = target.subGrids[node.firstChildIdx + childIdx];
                        subGrid ^= node.filled * std::numeric_limits<uint64_t>::max();
                    }
                }
            }
        } else {
            auto& childLevelNodes = target.nodesPerLevel[childLevel];
            for (size_t i = 0; i < levelNodes.size(); ++i) {
                const auto& node = levelNodes[i];
                if (node.firstChildIdx != node.EmptyChild) {
                    for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
                        auto& childNode = childLevelNodes[node.firstChildIdx + childIdx];
                        childNode.filled ^= node.filled;
                    }
                }
            }
        }
    }
}

// Sparse Solid Voxelization based on:
// http://research.michael-schwarz.com/publ/files/vox-siga10.pdf
//
// Outline:
// For each triangle:
//   For each voxel in triangles AABB:
//     Test intersection between voxel and triangle
//
// Conservative will use conservative voxelization (any voxel that a triangle touches is filled).
// With conservative off this function will perform 6-separating voxelization which is not conservative but is watertight.
EditStructure<void, uint32_t> voxelizeSparseSolid(std::span<const Mesh> meshes, unsigned resolution)
{
    // TODO: I should really write proper unit tests for this...
    // testSparseFill();

    const glm::ivec3 gridResolution { (int)resolution };

    // World space extent of a voxel
    const glm::vec3 delta_p { 1.0f };
    const glm::vec3 tile_delta_p { 5.0f, 4.0f, 4.0f };
    const glm::vec2 delta_p_yz { delta_p.y, delta_p.z };

    spdlog::info("Voxelizing meshes");
    SolidEditStructure intermediateStructure { resolution };
#if 0
    for (const auto& mesh : meshes)
        voxelizeMeshOptimized<false>(target, mesh);
#else
    for (const auto& mesh : meshes) {
        for (size_t t = 0; t < mesh.triangles.size(); t++) {
            TriangleProps tri {};
            tri.indices = mesh.triangles[t];
            tri.v[0] = mesh.positions[tri.indices[0]];
            tri.v[1] = mesh.positions[tri.indices[1]];
            tri.v[2] = mesh.positions[tri.indices[2]];
            // Ensure a consistent winding order.
            if ((tri.v[1].x - tri.v[0].x) * (tri.v[2].x - tri.v[0].x) < 0.0f)
                std::swap(tri.v[0], tri.v[1]);
            tri.e[0] = tri.v[1] - tri.v[0];
            tri.e[1] = tri.v[2] - tri.v[1];
            tri.e[2] = tri.v[0] - tri.v[2];
            tri.n = glm::cross(tri.e[0], tri.e[1]);
            // Skip degenerate triangles whose edges lie in a line.
            if (tri.n.x == 0.0f && tri.n.y == 0.0f && tri.n.z == 0.0f)
                continue;
            tri.abs_n = glm::abs(tri.n);

            // Enlarge triangle bounding box by one subgrid voxel in the -x direction (see section 5.1 of the paper).
            const glm::vec3 tBoundsMin = glm::min(tri.v[0], glm::min(tri.v[1], tri.v[2])) - glm::vec3(delta_p.x, 0, 0);
            const glm::vec3 tBoundsMax = glm::max(tri.v[0], glm::max(tri.v[1], tri.v[2]));
            const glm::ivec3 tBoundsMinVoxel = glm::clamp(glm::ivec3(tBoundsMin), glm::ivec3(0), gridResolution - 1);
            const glm::ivec3 tBoundsMaxVoxel = glm::clamp(glm::ivec3(tBoundsMax), glm::ivec3(0), gridResolution - 1);

            // Critical point on a corner of voxel (26-separating).
            // Shift all critical points that are on the voxel's max x face in the +x direction by one voxel.
            // See section 5.1 of the paper:
            // http://research.michael-schwarz.com/publ/files/vox-siga10.pdf
            const glm::vec3 tileC { tri.n.x > 0 ? tile_delta_p.x : 0, tri.n.y > 0 ? tile_delta_p.y : 0, tri.n.z > 0 ? tile_delta_p.z : 0 };
            const float tileD1 = glm::dot(tri.n, tileC - tri.v[0]);
            const float tileD2 = glm::dot(tri.n, (tile_delta_p - tileC) - tri.v[0]);

            // For each voxel in the triangles AABB
            const auto tBoundsMinTile = tBoundsMinVoxel & (~0b11);
            const auto tBoundsMaxTile = tBoundsMaxVoxel & (~0b11);
            for (int tz = tBoundsMinTile.z; tz <= tBoundsMaxTile.z; tz += 4) {
                for (int ty = tBoundsMinTile.y; ty <= tBoundsMaxTile.y; ty += 4) {
                    for (int tx = tBoundsMinTile.x; tx <= tBoundsMaxTile.x; tx += 4) {
                        // Intersection test
                        const glm::ivec3 tileIp { tx, ty, tz };
                        const glm::vec3 tileP { tileIp };

                        bool triangleIntersect2D = ((glm::dot(tri.n, tileP) + tileD1) * (glm::dot(tri.n, tileP) + tileD2)) <= 0;
                        triangleIntersect2D &= testTrianglePlane<0, true>(tri, tileP, tile_delta_p);
                        triangleIntersect2D &= testTrianglePlane<1, true>(tri, tileP, tile_delta_p);
                        triangleIntersect2D &= testTrianglePlane<2, true>(tri, tileP, tile_delta_p);
                        if (!triangleIntersect2D)
                            continue;

                        //    For each voxel in the triangles AABB
                        uint64_t bitmask = 0;
                        for (int dz = 0; dz < 4; dz++) {
                            for (int dy = 0; dy < 4; dy++) {
                                // bitmask |= ((uint64_t)1) << morton_encode64(glm::uvec3(0, dy, dz));
                                // continue;

                                const glm::ivec2 ip_yz { ty + dy, tz + dz };
                                const glm::vec2 p_center_yz { glm::vec2(ip_yz) + 0.5f * delta_p_yz };

                                // Test if triangle overlaps with voxel column.
                                bool triangleOverlapYZ = true;
                                for (int i = 0; i < 3; i++) {
                                    const glm::vec2 n_yz_ei = glm::vec2(-tri.e[i].z, tri.e[i].y) * (tri.n.x >= 0 ? +1.0f : -1.0f);
                                    const glm::vec2 v_yz_i { tri.v[i].y, tri.v[i].z };
                                    const float distFromEdge = glm::dot(p_center_yz - v_yz_i, n_yz_ei);
                                    // Add a tiny margin such that edges that exactly overlap the pixel center only count if they are left or top edges.
                                    const float f_yz = (n_yz_ei[0] > 0.0f || (n_yz_ei[0] == 0.0f && n_yz_ei[1] < 0.0f)) ? std::numeric_limits<float>::epsilon() : 0.0f;
                                    triangleOverlapYZ &= (distFromEdge + f_yz) > 0;
                                }
                                if (!triangleOverlapYZ)
                                    continue;

                                // dot(c - v, n) = 0
                                // (c_x-v_x) * n_x + (c_y-v_y) * n_y + (c_z-v_z) * n_z = 0
                                // (c_x-v_x) * n_x = -(c_y-v_y) * n_y - (c_z-v_z) * n_z
                                // c_x - v_x = (-(c_y-v_y) * n_y - (c_z-v_z) * n_z) / n_x
                                // c_x = (-(c_y-v_y)*n_y - (c_z-v_z)*n_z) / n_x + v_x
                                const float c_y = p_center_yz.x;
                                const float c_z = p_center_yz.y;
                                [[maybe_unused]] const float centerX = (-(c_y - tri.v[0].y) * tri.n.y - (c_z - tri.v[0].z) * tri.n.z) / tri.n.x + tri.v[0].x;
                                // NOTE: we shifted the triangle by 0.5 in the +x direction (see initialization of TriangleProps).
                                //       for the original triangle to have x > tx+0.5, which triangles to x +0.5 > (tx+0.5) +0.5 with the shifted triangle.
                                float relativeX = centerX - tx;
                                if (relativeX < 0.0f || relativeX >= 4.0f)
                                    continue;
                                // int q = int(relativeX);

                                // TODO: create lookup table (or some smart bit stuff).
                                uint64_t pixelColumn = 0;
                                uint64_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
                                if (relativeX < 1.0f)
                                    p0 = ((uint64_t)1) << morton_encode64(glm::uvec3(0, dy, dz));
                                if (relativeX < 2.0f)
                                    p1 = ((uint64_t)1) << morton_encode64(glm::uvec3(1, dy, dz));
                                if (relativeX < 3.0f)
                                    p2 = ((uint64_t)1) << morton_encode64(glm::uvec3(2, dy, dz));
                                if (relativeX < 4.0f)
                                    p3 = ((uint64_t)1) << morton_encode64(glm::uvec3(3, dy, dz));
                                pixelColumn = (p0 | p1 | p2 | p3);
                                bitmask |= pixelColumn;
                            }
                        }

                        uint64_t& subGrid = intermediateStructure.createSubGrid(tileP);
                        subGrid ^= bitmask;
                        // target.updateSubGrid(tileP, [&](uint64_t oldBitmask) { return oldBitmask ^ bitmask; });
                    } // Tile x
                } // Tile y
            } // Tile Z
        } // Triangles
    } // Meshes
#endif

    spdlog::info("Sparse flood filling.");
    sparseFill(intermediateStructure);

    spdlog::info("Converting to octree.");
    EditStructure<void, uint32_t> out { resolution };
    // Copy the non-empty leaf grids.
    std::vector<uint32_t> prevLevelMapping(intermediateStructure.subGrids.size());
    for (size_t leafIdx = 0; leafIdx < intermediateStructure.subGrids.size(); ++leafIdx) {
        const auto bitmask = intermediateStructure.subGrids[leafIdx];
        if (bitmask) {
            prevLevelMapping[leafIdx] = (uint32_t)out.subGrids.size();
            out.subGrids.push_back(EditSubGrid<void> { .bitmask = bitmask });
        } else {
            prevLevelMapping[leafIdx] = EditNode<uint32_t>::EmptyChild;
        }
    }

    uint32_t highestFullyFilledLevel = 0; // Highest level at which a fully filled node occurs.
    for (uint32_t level = out.subGridLevel + 1; level <= out.rootLevel; ++level) {
        // Copy nodes from the old format to the new format.
        const auto& inLevelNodes = intermediateStructure.nodesPerLevel[level];
        auto& outLevelNodes = out.nodesPerLevel[level];
        outLevelNodes.clear(); // Octree has a default root node; remove it.

        // Fully filled node will be added at the end of the array of children nodes.
        const auto childLevel = level - 1;
        const uint32_t fullyFilledChild = uint32_t(childLevel == out.subGridLevel ? out.subGrids.size() : out.nodesPerLevel[childLevel].size());
        std::vector<uint32_t> curLevelMapping(inLevelNodes.size());
        for (size_t nodeIdx = 0; nodeIdx < inLevelNodes.size(); ++nodeIdx) {
            const auto& inNode = inLevelNodes[nodeIdx];

            EditNode<uint32_t> outNode;
            if (inNode.firstChildIdx == inNode.EmptyChild && inNode.filled) {
                // Node does not have children and should be completely filled. Add a new node with the (to-be added) fully filled children.
                std::fill(std::begin(outNode.children), std::end(outNode.children), fullyFilledChild);
                highestFullyFilledLevel = level;
            } else if (inNode.firstChildIdx == inNode.EmptyChild && !inNode.filled) {
                std::fill(std::begin(outNode.children), std::end(outNode.children), EditNode<uint32_t>::EmptyChild);
            } else {
                // Node with children
                for (uint32_t childIdx = 0; childIdx < 8; ++childIdx)
                    outNode.children[childIdx] = prevLevelMapping[inNode.firstChildIdx + childIdx];
            }

            const bool fullyEmpty = ((size_t)std::count(std::begin(outNode.children), std::end(outNode.children), EditNode<uint32_t>::EmptyChild) == outNode.children.size());
            if (fullyEmpty) {
                curLevelMapping[nodeIdx] = EditNode<uint32_t>::EmptyChild;
            } else {
                curLevelMapping[nodeIdx] = (uint32_t)outLevelNodes.size();
                outLevelNodes.push_back(outNode);
            }
        }

        prevLevelMapping = std::move(curLevelMapping);
    }

    // Add fully filled nodes at the end of the nodes array.
    if (highestFullyFilledLevel >= out.subGridLevel)
        out.subGrids.push_back(EditSubGrid<void> { .bitmask = std::numeric_limits<uint64_t>::max() });
    for (uint32_t level = out.subGridLevel + 1; level <= highestFullyFilledLevel; ++level) {
        // Fully filled node will be added at the end of the array of children nodes.
        const auto childLevel = level - 1;
        const uint32_t fullyFilledChild = uint32_t(childLevel == out.subGridLevel ? out.subGrids.size() - 1 : out.nodesPerLevel[childLevel].size() - 1);

        EditNode<uint32_t> fullyFilledNode;
        std::fill(std::begin(fullyFilledNode.children), std::end(fullyFilledNode.children), fullyFilledChild);
        out.nodesPerLevel[level].push_back(fullyFilledNode);
    }
    return out;
}

std::vector<Mesh> prepareMeshForVoxelization(std::span<const Mesh> meshes, unsigned resolution, Bounds& inOutSceneBounds)
{
    const float maxExtent = glm::compMax(inOutSceneBounds.extent());
    const glm::vec3 scale { (float)resolution / maxExtent };
    const glm::vec3 offset = -inOutSceneBounds.lower;

    inOutSceneBounds.reset();
    std::vector<Mesh> out;
    for (Mesh mesh : meshes) {
        for (auto& vertex : mesh.positions) {
            vertex = (vertex + offset) * scale;
            inOutSceneBounds.grow(vertex);
        }
        out.emplace_back(std::move(mesh));
    }
    return out;
}

template EditStructure<voxcom::RGB, uint32_t> voxelizeHierarchical<voxcom::RGB, false>(std::span<const Mesh> meshes, unsigned resolution);
template EditStructure<void, uint32_t> voxelizeHierarchical<void, false>(std::span<const Mesh> meshes, unsigned resolution);

template void voxelizeMeshNaive<false>(EditStructure<voxcom::RGB, uint32_t>&, const Mesh&, const glm::vec3&, float);
template void voxelizeMeshOptimized<false>(EditStructure<voxcom::RGB, uint32_t>&, const Mesh&, const glm::vec3&, float);

template void voxelizeMeshNaive<false>(EditStructure<void, uint32_t>&, const Mesh&, const glm::vec3&, float);
template void voxelizeMeshNaive<true>(EditStructure<void, uint32_t>&, const Mesh&, const glm::vec3&, float);
template void voxelizeMeshNaive<false>(VoxelGrid<void>&, const Mesh&, const glm::vec3&, float);
template void voxelizeMeshNaive<true>(VoxelGrid<void>&, const Mesh&, const glm::vec3&, float);

template void voxelizeMeshOptimized<false>(EditStructure<void, uint32_t>&, const Mesh&, const glm::vec3&, float);
template void voxelizeMeshOptimized<true>(EditStructure<void, uint32_t>&, const Mesh&, const glm::vec3&, float);
template void voxelizeMeshOptimized<false>(VoxelGrid<void>&, const Mesh&, const glm::vec3&, float);
template void voxelizeMeshOptimized<true>(VoxelGrid<void>&, const Mesh&, const glm::vec3&, float);

// template void voxelizeSparseSolid(EditStructure<void>& target, std::span<const Mesh> meshes);
// template void voxelizeSparseSolid(EditStructure<RGB>& target, std::span<const Mesh> meshes);
}