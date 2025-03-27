#pragma once
#include "configuration/transform_dag_definitions.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/dag_utils.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/hash_dag/hash_dag_colors.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "dags/symmetry_aware_dag/symmetry_aware_dag.h"
#include "dags/transform_dag/transform_dag.h"
#include "memory.h"
#include "tracer.h"
#include "utils.h"
#include <cmath>
#include <cstddef>
#include <glm/gtc/constants.hpp>
#include <numbers>
#include <spdlog/spdlog.h>
#include <stack>
#include <type_traits>
#include <vector>

#ifdef __CUDA_ARCH__
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_math.h>
#else
#include <bit>
#endif

HOST_DEVICE float3 make_float3(small_int3 i3) { return make_float3((float)i3.xyz[0], (float)i3.xyz[1], (float)i3.xyz[2]); }
HOST_DEVICE int3 make_int3(small_int3 i3) { return make_int3((int)i3.xyz[0], (int)i3.xyz[1], (int)i3.xyz[2]); }
HOST_DEVICE bool any(small_int3 i3) { return i3.xyz[0] || i3.xyz[1] || i3.xyz[2]; }

template <typename T>
struct dag_has_materials : std::false_type {
};
#if EDITS_ENABLE_MATERIALS
template <>
struct dag_has_materials<MyGPUHashDAG<EMemoryType::GPU_Malloc>> : std::true_type {
};
// template <>
// struct dag_has_materials<MyGpuSortedDag> : std::true_type {
// };
#endif
template <typename T>
static constexpr bool dag_has_materials_v = dag_has_materials<T>::value;

namespace Tracer {

// order: (shouldFlipX, shouldFlipY, shouldFlipZ)
DEVICE uint8 next_child(uint8 order, uint8 mask)
{
    for (uint8 child = 0; child < 8; ++child) {
        uint8 childInOrder = child ^ order;
        if (mask & (1u << childInOrder))
            return childInOrder;
    }
    check(false);
    return 0;
}

template <bool isRoot, typename TDAG>
DEVICE uint8 compute_intersection_mask(uint32 level, const Path& path, const TDAG& dag, const Ray ray, const float3 invRayDirection)
{
    // Find node center = .5 * (boundsMin + boundsMax) + .5f
    const uint32 shift = dag.levels - level;

    const float radius = float(1u << (shift - 1));
    const float3 center = make_float3(radius) + path.as_position(shift);

    const float3 centerRelativeToRay = center - ray.origin;

    // Ray intersection with axis-aligned planes centered on the node
    // => rayOrg + tmid * rayDir = center
    const float3 tmid = centerRelativeToRay * invRayDirection;

    // t-values for where the ray intersects the slabs centered on the node
    // and extending to the side of the node
    float tmin, tmax;
    {
        const float3 slabRadius = radius * abs(invRayDirection);
        const float3 pmin = tmid - slabRadius;
        tmin = max(max(pmin), 0.0f);

        const float3 pmax = tmid + slabRadius;
        tmax = min(pmax);
    }
    tmin = max(tmin, ray.tmin);
    tmax = min(tmax, ray.tmax);

    // Check if we actually hit the root node
    // This test may not be entirely safe due to float precision issues.
    // especially on lower levels. For the root node this seems OK, though.
    if (isRoot && (tmin >= tmax)) {
        return 0;
    }

    // Identify first child that is intersected
    // NOTE: We assume that we WILL hit one child, since we assume that the
    //       parents bounding box is hit.
    // NOTE: To safely get the correct node, we cannot use o+ray_tmin*d as the
    //       intersection point, since this point might lie too close to an
    //       axis plane. Instead, we use the midpoint between max and min which
    //       will lie in the correct node IF the ray only intersects one node.
    //       Otherwise, it will still lie in an intersected node, so there are
    //       no false positives from this.
    uint8 intersectionMask = 0;
    {
        const float3 pointOnRay = (0.5f * (tmin + tmax)) * ray.direction;

        uint8 const firstChild = ((pointOnRay.x >= centerRelativeToRay.x) ? 4 : 0) + ((pointOnRay.y >= centerRelativeToRay.y) ? 2 : 0) + ((pointOnRay.z >= centerRelativeToRay.z) ? 1 : 0);

        intersectionMask |= (1u << firstChild);
    }

    // We now check the points where the ray intersects the X, Y and Z plane.
    // If the intersection is within (ray_tmin, ray_tmax) then the intersection
    // point implies that two voxels will be touched by the ray. We find out
    // which voxels to mask for an intersection point at +X, +Y by setting
    // ALL voxels at +X and ALL voxels at +Y and ANDing these two masks.
    //
    // NOTE: When the intersection point is close enough to another axis plane,
    //       we must check both sides or we will get robustness issues.
    const float epsilon = 1e-4f;

    if (tmin <= tmid.x && tmid.x <= tmax) {
        const float3 pointOnRay = tmid.x * ray.direction;

        uint8 A = 0;
        if (pointOnRay.y >= centerRelativeToRay.y - epsilon)
            A |= 0xCC;
        if (pointOnRay.y <= centerRelativeToRay.y + epsilon)
            A |= 0x33;

        uint8 B = 0;
        if (pointOnRay.z >= centerRelativeToRay.z - epsilon)
            B |= 0xAA;
        if (pointOnRay.z <= centerRelativeToRay.z + epsilon)
            B |= 0x55;

        intersectionMask |= A & B;
    }
    if (tmin <= tmid.y && tmid.y <= tmax) {
        const float3 pointOnRay = tmid.y * ray.direction;

        uint8 C = 0;
        if (pointOnRay.x >= centerRelativeToRay.x - epsilon)
            C |= 0xF0;
        if (pointOnRay.x <= centerRelativeToRay.x + epsilon)
            C |= 0x0F;

        uint8 D = 0;
        if (pointOnRay.z >= centerRelativeToRay.z - epsilon)
            D |= 0xAA;
        if (pointOnRay.z <= centerRelativeToRay.z + epsilon)
            D |= 0x55;

        intersectionMask |= C & D;
    }
    if (tmin <= tmid.z && tmid.z <= tmax) {
        const float3 pointOnRay = tmid.z * ray.direction;

        uint8 E = 0;
        if (pointOnRay.x >= centerRelativeToRay.x - epsilon)
            E |= 0xF0;
        if (pointOnRay.x <= centerRelativeToRay.x + epsilon)
            E |= 0x0F;

        uint8 F = 0;
        if (pointOnRay.y >= centerRelativeToRay.y - epsilon)
            F |= 0xCC;
        if (pointOnRay.y <= centerRelativeToRay.y + epsilon)
            F |= 0x33;

        intersectionMask |= E & F;
    }

    return intersectionMask;
}

// Copied from: https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/
// Reverses `pos` from range [1.0, 2.0) to (2.0, 1.0] if `dir > 0`.
HOST_DEVICE float3 getMirroredPos(float3 pos, const float3& dir, bool rangeCheck)
{
    float3 mirrored = uint3_as_float3(float3_as_uint3(pos) ^ 0x7FFFFF);
    // XOR-ing will only work for coords in range [1.0, 2.0),
    // fallback to subtractions if that's not the case.
    // if (rangeCheck && (pos.x < 1.0f || pos.x >= 2.0f || pos.y < 1.0f || pos.y >= 2.0f || pos.z < 1.0f || pos.z >= 2.0f))
    mirrored = 3.0f - pos;

    if (dir.x > 0.0f)
        pos.x = mirrored.x;
    if (dir.y > 0.0f)
        pos.y = mirrored.y;
    if (dir.z > 0.0f)
        pos.z = mirrored.z;
    return pos;
}
// Copied from: https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/
HOST_DEVICE int getNodeCellIndexXYZ(float3 pos, int scaleExp)
{
    const uint3 cellPos = (float3_as_uint3(pos) >> scaleExp) & 1;
    return cellPos.x * 4 + cellPos.y * 2 + cellPos.z * 1;
}
HOST_DEVICE int getNodeCellIndexZYX(float3 pos, int scaleExp)
{
    const uint3 cellPos = (float3_as_uint3(pos) >> scaleExp) & 1;
    return cellPos.x * 1 + cellPos.y * 2 + cellPos.z * 4;
}
// Copied from: https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/
// floor(pos / scale) * scale
HOST_DEVICE float3 floorScale(float3 pos, int scaleExp)
{
    const uint32_t mask = ~0u << scaleExp;
    return uint3_as_float3(float3_as_uint3(pos) & mask); // Erase bits lower than scale.
}

template <typename TDAG>
DEVICE RayHit intersect_ray2(const TDAG& dag, Ray& ray, uint32_t& materialId)
{
    constexpr uint32_t resolution = 1u << MAX_LEVELS;
    constexpr float invResolution = float(1.0 / double(resolution));

    // Map from voxel space to [1, 2] range.
    float3 origin = make_float3(1.0f) + ray.origin * invResolution;
    origin = getMirroredPos(origin, ray.direction, true);
    checkAlways(origin.x >= 0.999f && origin.x <= 2.001f);
    checkAlways(origin.y >= 0.999f && origin.y <= 2.001f);
    checkAlways(origin.z >= 0.999f && origin.z <= 2.001f);
    const float3 invRayDirection = make_float3(1.0f) / -abs(ray.direction);

    const uint32_t mirrorMask = (ray.direction.x > 0.f ? 4 : 0) + (ray.direction.y > 0.f ? 2 : 0) + (ray.direction.z > 0.f ? 1 : 0);

    // Copied from: https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/
    uint32_t stack[MAX_LEVELS];
    int scaleExp = 22; // 0.5
    uint32_t level = 0;

    uint32_t nodeIdx = dag.get_first_node_index();
    uint32_t childMask = Utils::child_mask(dag.get_node(level, nodeIdx));
    float3 pos = clamp(origin, make_float3(1.0f), make_float3(1.9999999f));

    Leaf leafCache;
    float3 sideDist;
    for (int i = 0; i < 512; ++i) {
        uint32_t childIdx = getNodeCellIndexXYZ(pos, scaleExp) ^ mirrorMask;
        // Descend.
        while (((childMask >> childIdx) & 0b1) && level != MAX_LEVELS - 1) {
            stack[level] = nodeIdx;
            if (level == dag.leaf_level()) {
                childMask = leafCache.get_second_child_mask(childIdx);
            } else if (level == dag.leaf_level() - 1) {
                nodeIdx = dag.get_child_index(level, nodeIdx, childMask, childIdx);
                leafCache = dag.get_leaf(nodeIdx);
                childMask = leafCache.get_first_child_mask();
            } else {
                nodeIdx = dag.get_child_index(level, nodeIdx, childMask, childIdx);
                childMask = Utils::child_mask(dag.get_node(level + 1, nodeIdx));
            }
            --scaleExp;
            level = 22 - scaleExp;

            childIdx = getNodeCellIndexXYZ(pos, scaleExp) ^ mirrorMask;
        }

        // Intersect individual voxel.
        if (level == MAX_LEVELS - 1 && ((childMask >> childIdx) & 0b1))
            break;

        // Compute next pos by intersecting with max cell sides.
        const float3 cellMin = floorScale(pos, scaleExp);

        sideDist = (cellMin - origin) * invRayDirection;
        const float tmax = min(min(sideDist.x, sideDist.y), sideDist.z);

        const int3 neighborMax = float3_as_int3(cellMin) + make_int3(sideDist.x == tmax ? -1 : (1 << scaleExp) - 1, sideDist.y == tmax ? -1 : (1 << scaleExp) - 1, sideDist.z == tmax ? -1 : (1 << scaleExp) - 1);
        pos = min(origin - abs(ray.direction) * tmax, int3_as_float3(neighborMax));

        // Find common ancestor based on left-most carry bit.
        const uint3 diffPos = float3_as_uint3(pos) ^ float3_as_uint3(cellMin);
        int diffExp = 31 - __clz(diffPos.x | diffPos.y | diffPos.z);

        if (diffExp > scaleExp) {
            scaleExp = diffExp;
            if (diffExp > 22) // Going out of root?
                break;

            level = 22 - scaleExp;
            nodeIdx = stack[level];
            checkAlways(level <= dag.leaf_level());
            if (level == dag.leaf_level()) {
                leafCache = dag.get_leaf(nodeIdx);
                childMask = leafCache.get_first_child_mask();
            } else {
                childMask = Utils::child_mask(dag.get_node(level, nodeIdx));
            }
        }
    }

    RayHit out = RayHit::empty();
    if (level == MAX_LEVELS - 1) {
        const float3 floatPos = (getMirroredPos(pos, ray.direction, false) - 1.0f) * resolution;
        const uint3 path = make_uint3((getMirroredPos(floorScale(pos, scaleExp), ray.direction, false) - 1.0f) * resolution);
        ray.tmax = min(min(sideDist.x, sideDist.y), sideDist.z);
        const uint32_t normalAxis = (ray.tmax == sideDist.x ? 0 : (ray.tmax == sideDist.y ? 1 : 2));
        out.init(ray, floatPos, Path(path), normalAxis);
    }
    return out;
}

struct StackEntry {
    uint32 index;
    uint8 childMask;
    uint8 visitMask;
};
template <typename TDAG>
DEVICE RayHit intersect_ray(const TDAG& dag, Ray& ray, uint32_t& materialId)
{
    const float3 invRayDirection = make_float3(1.0f) / ray.direction;
    const uint8 rayChildOrder = (ray.direction.x < 0.f ? 4 : 0) + (ray.direction.y < 0.f ? 2 : 0) + (ray.direction.z < 0.f ? 1 : 0);

    // State
    uint32_t level = 0, addr;
    Path path(0, 0, 0);

    StackEntry stack[MAX_LEVELS];
    StackEntry cache;
    Leaf cachedLeaf; // needed to iterate on the last few levels

    cache.index = dag.get_first_node_index();
    cache.childMask = Utils::child_mask(dag.get_node(0, cache.index));
    cache.visitMask = cache.childMask & compute_intersection_mask<true>(0, path, dag, ray, invRayDirection);

    // Traverse DAG
    for (;;) {
        // Ascend if there are no children left.
        {
            uint32 newLevel = level;
            while (newLevel > 0 && !cache.visitMask) {
                newLevel--;
                cache = stack[newLevel];
            }

            if (newLevel == 0 && !cache.visitMask) {
                path = Path(0, 0, 0);
                break;
            }

            path.ascend(level - newLevel);
            level = newLevel;
        }

        // Find next child in order by the current ray's direction
        const uint8 nextChild = next_child(rayChildOrder, cache.visitMask);

        // Mark it as handled
        cache.visitMask &= ~(1u << nextChild);

        // Intersect that child with the ray
        {
            path.descend(nextChild);
            stack[level] = cache;
            level++;

            // If we're at the final level, we have intersected a single voxel.
            if (level == dag.levels) {
                if constexpr (dag_has_materials_v<TDAG>) {
                    const auto voxelIdx = path.mortonU32() & 0b111111;
                    if (!dag.get_material(dag.get_leaf_ptr(addr), voxelIdx, materialId))
                        materialId = 0xFFFFFFFF;
                }
                break;
            }

            // Are we in an internal node?
            if (level < dag.leaf_level()) {
                cache.index = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
                cache.childMask = Utils::child_mask(dag.get_node(level, cache.index));
                cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, ray, invRayDirection);
            } else {
                /* The second-to-last and last levels are different: the data
                 * of these two levels (2^3 voxels) are packed densely into a
                 * single 64-bit word.
                 */
                uint8 childMask;

                if (level == dag.leaf_level()) {
                    addr = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
                    cachedLeaf = dag.get_leaf(addr);
                    childMask = cachedLeaf.get_first_child_mask();
                } else {
                    childMask = cachedLeaf.get_second_child_mask(nextChild);
                }

                // No need to set the index for bottom nodes
                cache.childMask = childMask;
                cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, ray, invRayDirection);
            }
        }
    }

    RayHit out = RayHit::empty();
    if (!path.is_null())
        out.init(path, ray, invRayDirection);
    return out;
}
template <typename TDAG>
DEVICE bool intersect_ray_node_out_of_order(const TDAG& dag, Ray ray)
{
    const float3 invRayDirection = make_float3(1.0f) / ray.direction;

    // State
    uint32 level = 0;
    Path path(0, 0, 0);

    StackEntry stack[MAX_LEVELS];
    StackEntry cache;
    Leaf cachedLeaf; // needed to iterate on the last few levels

    cache.index = dag.get_first_node_index();
    cache.childMask = Utils::child_mask(dag.get_node(0, cache.index));
    cache.visitMask = cache.childMask & compute_intersection_mask<true>(0, path, dag, ray, invRayDirection);

    // Traverse DAG
    for (;;) {
        // Ascend if there are no children left.
        {
            uint32 newLevel = level;
            while (newLevel > 0 && !cache.visitMask) {
                newLevel--;
                cache = stack[newLevel];
            }

            if (newLevel == 0 && !cache.visitMask) {
                path = Path(0, 0, 0);
                break;
            }

            path.ascend(level - newLevel);
            level = newLevel;
        }

// Find next child in order by the current ray's direction
#if __CUDA_ARCH__
        const uint8 nextChild = 31 - __clz(cache.visitMask);
#else
        const uint8 nextChild = 31 - std::countl_zero(cache.visitMask);
#endif

        // Mark it as handled
        cache.visitMask &= ~(1u << nextChild);

        // Intersect that child with the ray
        {
            path.descend(nextChild);
            stack[level] = cache;
            level++;

            // If we're at the final level, we have intersected a single voxel.
            if (level == dag.levels) {
                return true;
            }

            // Are we in an internal node?
            if (level < dag.leaf_level()) {
                cache.index = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
                cache.childMask = Utils::child_mask(dag.get_node(level, cache.index));
                cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, ray, invRayDirection);
            } else {
                /* The second-to-last and last levels are different: the data
                 * of these two levels (2^3 voxels) are packed densely into a
                 * single 64-bit word.
                 */
                uint8 childMask;

                if (level == dag.leaf_level()) {
                    const uint32 addr = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
                    cachedLeaf = dag.get_leaf(addr);
                    childMask = cachedLeaf.get_first_child_mask();
                } else {
                    childMask = cachedLeaf.get_second_child_mask(nextChild);
                }

                // No need to set the index for bottom nodes
                cache.childMask = childMask;
                cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, ray, invRayDirection);
            }
        }
    }
    return false;
}

template <bool isRoot>
HOST_DEVICE uint8 compute_intersection_mask(uint32 level, const Path& path, const Ray& ray, const float3 invRayDirection)
{
    const float radius = float(1u << (level - 1));
    const float3 center = make_float3(radius) + path.as_position(level);

    const float3 centerRelativeToRay = center - ray.origin;

    // Ray intersection with axis-aligned planes centered on the node
    // => rayOrg + tmid * rayDir = center
    const float3 tmid = centerRelativeToRay * invRayDirection;

    // t-values for where the ray enters and exists the node.
    const float3 slabRadius = radius * abs(invRayDirection);

    float tmin, tmax;
    {
        const float3 pmin = tmid - slabRadius;
        tmin = max(max(pmin), 0.0f);

        const float3 pmax = tmid + slabRadius;
        tmax = min(pmax);
    }

    // Check if we actually hit the root node
    // This test may not be entirely safe due to float precision issues.
    // especially on lower levels. For the root node this seems OK, though.
    if (isRoot && (tmin >= tmax)) {
        return 0;
    }

    // Identify first child that is intersected
    // NOTE: We assume that we WILL hit one child, since we assume that the
    //       parents bounding box is hit.
    // NOTE: To safely get the correct node, we cannot use o+ray_tmin*d as the
    //       intersection point, since this point might lie too close to an
    //       axis plane. Instead, we use the midpoint between max and min which
    //       will lie in the correct node IF the ray only intersects one node.
    //       Otherwise, it will still lie in an intersected node, so there are
    //       no false positives from this.
    uint8 intersectionMask = 0;
    {
        const float3 pointOnRay = (0.5f * (tmin + tmax)) * ray.direction;

        uint8 const firstChild = ((pointOnRay.x >= centerRelativeToRay.x) ? 1 : 0) + ((pointOnRay.y >= centerRelativeToRay.y) ? 2 : 0) + ((pointOnRay.z >= centerRelativeToRay.z) ? 4 : 0);

        intersectionMask |= (1u << firstChild);
    }

    // We now check the points where the ray intersects the X, Y and Z plane.
    // If the intersection is within (ray_tmin, ray_tmax) then the intersection
    // point implies that two voxels will be touched by the ray. We find out
    // which voxels to mask for an intersection point at +X, +Y by setting
    // ALL voxels at +X and ALL voxels at +Y and ANDing these two masks.
    //
    // NOTE: When the intersection point is close enough to another axis plane,
    //       we must check both sides or we will get robustness issues.
    const float epsilon = 1e-4f;

    if (tmin <= tmid.x && tmid.x <= tmax) {
        const float3 pointOnRay = tmid.x * ray.direction;

        uint8 A = 0;
        if (pointOnRay.y >= centerRelativeToRay.y - epsilon)
            A |= 0xCC;
        if (pointOnRay.y <= centerRelativeToRay.y + epsilon)
            A |= 0x33;

        uint8 B = 0;
        if (pointOnRay.z >= centerRelativeToRay.z - epsilon)
            B |= 0xF0;
        if (pointOnRay.z <= centerRelativeToRay.z + epsilon)
            B |= 0x0F;

        intersectionMask |= A & B;
    }
    if (tmin <= tmid.y && tmid.y <= tmax) {
        const float3 pointOnRay = tmid.y * ray.direction;

        uint8 C = 0;
        if (pointOnRay.x >= centerRelativeToRay.x - epsilon)
            C |= 0xAA;
        if (pointOnRay.x <= centerRelativeToRay.x + epsilon)
            C |= 0x55;

        uint8 D = 0;
        if (pointOnRay.z >= centerRelativeToRay.z - epsilon)
            D |= 0xF0;
        if (pointOnRay.z <= centerRelativeToRay.z + epsilon)
            D |= 0x0F;

        intersectionMask |= C & D;
    }
    if (tmin <= tmid.z && tmid.z <= tmax) {
        const float3 pointOnRay = tmid.z * ray.direction;

        uint8 E = 0;
        if (pointOnRay.x >= centerRelativeToRay.x - epsilon)
            E |= 0xAA;
        if (pointOnRay.x <= centerRelativeToRay.x + epsilon)
            E |= 0x55;

        uint8 F = 0;
        if (pointOnRay.y >= centerRelativeToRay.y - epsilon)
            F |= 0xCC;
        if (pointOnRay.y <= centerRelativeToRay.y + epsilon)
            F |= 0x33;

        intersectionMask |= E & F;
    }

    return intersectionMask;
}
template <bool isRoot>
HOST_DEVICE uint8 compute_intersection_mask(
    uint32 level, const Path& path,
    const small_int3 parentTranslation, const small_int3 currentTranslation,
    const Ray ray, const float3 invRayDirection, float& tmin, float& tmax)
{
    constexpr float selfIntersectEpsilon = 0.001f;

    const float radius = float(1u << (level - 1));
    const float3 center = make_float3(radius) + path.as_position(level) + make_float3(parentTranslation + currentTranslation);

    const float3 centerRelativeToRay = center - ray.origin;

    // Ray intersection with axis-aligned planes centered on the node
    // => rayOrg + tmid * rayDir = center
    const float3 tmid = centerRelativeToRay * invRayDirection;

    // t-values for where the ray enters and exists the node, shifted by currentTranslation.
    // This ensures that voxels that are shifted out of the current node are not visible.
    const float3 centerBeforeShiftRelativeToRay = (centerRelativeToRay - make_float3(currentTranslation) * (1.0f + selfIntersectEpsilon));
    const float3 tmidBeforeShift = centerBeforeShiftRelativeToRay * invRayDirection;

    // t-values for where the ray enters and exists the node.
    const float3 slabRadius = radius * abs(invRayDirection);
    {
        const float3 pmin = max(tmid, tmidBeforeShift) - slabRadius;
        tmin = max(tmin, max(max(pmin), 0.0f));

        const float3 pmax = min(tmid, tmidBeforeShift) + slabRadius;
        tmax = min(tmax, min(pmax));
    }

    // Check if we actually hit the root node
    // This test may not be entirely safe due to float precision issues.
    // especially on lower levels. For the root node this seems OK, though.
    if (isRoot && (tmin >= tmax)) {
        return 0;
    }

    // Identify first child that is intersected
    // NOTE: We assume that we WILL hit one child, since we assume that the
    //       parents bounding box is hit.
    // NOTE: To safely get the correct node, we cannot use o+ray_tmin*d as the
    //       intersection point, since this point might lie too close to an
    //       axis plane. Instead, we use the midpoint between max and min which
    //       will lie in the correct node IF the ray only intersects one node.
    //       Otherwise, it will still lie in an intersected node, so there are
    //       no false positives from this.
    uint8 intersectionMask = 0;
    {
        const float3 pointOnRay = (0.5f * (tmin + tmax)) * ray.direction;

        uint8 const firstChild = ((pointOnRay.x >= centerRelativeToRay.x) ? 1 : 0) + ((pointOnRay.y >= centerRelativeToRay.y) ? 2 : 0) + ((pointOnRay.z >= centerRelativeToRay.z) ? 4 : 0);

        intersectionMask |= (1u << firstChild);
    }

    // We now check the points where the ray intersects the X, Y and Z plane.
    // If the intersection is within (ray_tmin, ray_tmax) then the intersection
    // point implies that two voxels will be touched by the ray. We find out
    // which voxels to mask for an intersection point at +X, +Y by setting
    // ALL voxels at +X and ALL voxels at +Y and ANDing these two masks.
    //
    // NOTE: When the intersection point is close enough to another axis plane,
    //       we must check both sides or we will get robustness issues.
    const float epsilon = 1e-4f;

    if (tmin <= tmid.x && tmid.x <= tmax) {
        const float3 pointOnRay = tmid.x * ray.direction;

        uint8 A = 0;
        if (pointOnRay.y >= centerRelativeToRay.y - epsilon)
            A |= 0xCC;
        if (pointOnRay.y <= centerRelativeToRay.y + epsilon)
            A |= 0x33;

        uint8 B = 0;
        if (pointOnRay.z >= centerRelativeToRay.z - epsilon)
            B |= 0xF0;
        if (pointOnRay.z <= centerRelativeToRay.z + epsilon)
            B |= 0x0F;

        intersectionMask |= A & B;
    }
    if (tmin <= tmid.y && tmid.y <= tmax) {
        const float3 pointOnRay = tmid.y * ray.direction;

        uint8 C = 0;
        if (pointOnRay.x >= centerRelativeToRay.x - epsilon)
            C |= 0xAA;
        if (pointOnRay.x <= centerRelativeToRay.x + epsilon)
            C |= 0x55;

        uint8 D = 0;
        if (pointOnRay.z >= centerRelativeToRay.z - epsilon)
            D |= 0xF0;
        if (pointOnRay.z <= centerRelativeToRay.z + epsilon)
            D |= 0x0F;

        intersectionMask |= C & D;
    }
    if (tmin <= tmid.z && tmid.z <= tmax) {
        const float3 pointOnRay = tmid.z * ray.direction;

        uint8 E = 0;
        if (pointOnRay.x >= centerRelativeToRay.x - epsilon)
            E |= 0xAA;
        if (pointOnRay.x <= centerRelativeToRay.x + epsilon)
            E |= 0x55;

        uint8 F = 0;
        if (pointOnRay.y >= centerRelativeToRay.y - epsilon)
            F |= 0xCC;
        if (pointOnRay.y <= centerRelativeToRay.y + epsilon)
            F |= 0x33;

        intersectionMask |= E & F;
    }

    return intersectionMask;
}

HOST_DEVICE small_int3 applyAxisPermutation(small_int3 inShift, const TransformDAG16::Transform& transform)
{
    return small_int3 { {
        inShift.xyz[transform.axis0],
        inShift.xyz[transform.axis1],
        inShift.xyz[transform.axis2],
    } };
}

HOST_DEVICE uint8 next_child_lut(uint8 order, uint8 mask)
{
    for (uint8 child = 0; child < 8; ++child) {
        uint8 childInOrder = child ^ order;
        if (mask & (1u << childInOrder))
            return childInOrder;
    }
    check(false);
    return 0;
}

template <bool OutOfOrder, unsigned RootLevel>
DEVICE std::conditional_t<OutOfOrder, bool, RayHit> intersect_ray_impl2(
    const TransformDAG16::TraversalConstants& traversalConstants, const TransformDAG16& dag, const Path& rootPath, uint32_t rootNodeIdx, uint32_t rootTransformID, Ray& ray)
{
    constexpr uint32_t resolution = 1u << MAX_LEVELS;
    constexpr float invResolution = float(1.0 / double(resolution));

    // Map from voxel space to [1, 2] range.
    float3 origin = make_float3(1.0f) + ray.origin * invResolution;
    origin = getMirroredPos(origin, ray.direction, true);
    checkAlways(origin.x >= 0.999f && origin.x <= 2.001f);
    checkAlways(origin.y >= 0.999f && origin.y <= 2.001f);
    checkAlways(origin.z >= 0.999f && origin.z <= 2.001f);
    const float3 invRayDirection = make_float3(1.0f) / -abs(ray.direction);

    const uint32_t mirrorMask = (ray.direction.x > 0.f ? 1 : 0) + (ray.direction.y > 0.f ? 2 : 0) + (ray.direction.z > 0.f ? 4 : 0);

    // Copied from: https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/
    struct StackItem {
        uint32_t nodeIdx;
        uint16_t nodeHeader;
        uint8_t childMask;
#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
        uint8_t transformID;
#endif
    };
    static_assert(sizeof(StackItem) == 8);
    StackItem stack[RootLevel + 1];
    int scaleExp = 22 - (MAX_LEVELS - RootLevel); // 0.5
    int level = RootLevel;

    StackItem cache;
    cache.nodeIdx = dag.get_first_node_index();
    uint16_t const* pNode = &dag.nodes[dag.levelStarts[level] + cache.nodeIdx];
    cache.nodeHeader = *pNode;
    cache.childMask = dag.convert_child_mask(cache.nodeHeader);
#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
    cache.transformID = 0;
#endif

    float3 pos = clamp(origin, make_float3(1.0f), make_float3(1.9999999f));

    Leaf leafCache;
    float3 sideDist;
    for (int i = 0; i < 512; ++i) {
        uint32_t childIdx = getNodeCellIndexZYX(pos, scaleExp) ^ mirrorMask;
#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
        childIdx = traversalConstants.transformChildMappingWorldToLocal[cache.transformID][childIdx];
#endif
        // Descend.
        while (level != 1 && ((cache.childMask >> childIdx) & 0b1)) {
            stack[level] = cache;
            if (level == dag.leafLevel) {
                cache.childMask = leafCache.get_second_child_mask(childIdx);
            } else {
                const auto childPtr = dag.get_child_index(level - 1, pNode, cache.nodeHeader, childIdx);
#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
                cache.transformID = traversalConstants.transformCombineTable[cache.transformID][childPtr.transformID];
#endif
                cache.nodeIdx = (uint32_t)childPtr.index;
                if (level == dag.leafLevel + 1) {
                    leafCache = dag.get_leaf(cache.nodeIdx);
                    cache.childMask = leafCache.get_first_child_mask();
                } else {
                    pNode = &dag.nodes[dag.levelStarts[level - 1] + cache.nodeIdx];
                    cache.nodeHeader = *pNode;
                    cache.childMask = dag.convert_child_mask(cache.nodeHeader);
                }
            }
            --scaleExp;
            --level;

            childIdx = getNodeCellIndexZYX(pos, scaleExp) ^ mirrorMask;
#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
            childIdx = traversalConstants.transformChildMappingWorldToLocal[cache.transformID][childIdx];
#endif
        }

        // Intersect individual voxel.
        if (level == 1 && ((cache.childMask >> childIdx) & 0b1))
            break;

        // Compute next pos by intersecting with max cell sides.
        const float3 cellMin = floorScale(pos, scaleExp);

        sideDist = (cellMin - origin) * invRayDirection;
        const float tmax = min(min(sideDist.x, sideDist.y), sideDist.z);

        const int3 neighborMax = float3_as_int3(cellMin) + make_int3(sideDist.x == tmax ? -1 : (1 << scaleExp) - 1, sideDist.y == tmax ? -1 : (1 << scaleExp) - 1, sideDist.z == tmax ? -1 : (1 << scaleExp) - 1);
        pos = min(origin - abs(ray.direction) * tmax, int3_as_float3(neighborMax));

        // Find common ancestor based on left-most carry bit.
        const uint3 diffPos = float3_as_uint3(pos) ^ float3_as_uint3(cellMin);
        int diffExp = 31 - __clz(diffPos.x | diffPos.y | diffPos.z);

        if (diffExp > scaleExp) {
            scaleExp = diffExp;
            if (diffExp > 22) // Going out of root?
                break;

            level = scaleExp - 22 + MAX_LEVELS;
            cache = stack[level];
            if (level == dag.leafLevel) {
                leafCache = dag.get_leaf(cache.nodeIdx);
            } else {
                pNode = &dag.nodes[dag.levelStarts[level] + cache.nodeIdx];
            }
        }
    }

    if constexpr (OutOfOrder) {
        return level == 1;
    } else {
        RayHit out = RayHit::empty();
        if (level == 1) {
            const float3 floatPos = (getMirroredPos(pos, ray.direction, false) - 1.0f) * resolution;
            const uint3 path = make_uint3((getMirroredPos(floorScale(pos, scaleExp), ray.direction, false) - 1.0f) * resolution);
            ray.tmax = min(min(sideDist.x, sideDist.y), sideDist.z);
            const uint32_t normalAxis = (ray.tmax == sideDist.x ? 0 : (ray.tmax == sideDist.y ? 1 : 2));
            out.init(ray, floatPos, Path(path), normalAxis);
        }
        return out;
    }
}

template <bool OutOfOrder, unsigned RootLevel>
HOST_DEVICE std::conditional_t<OutOfOrder, bool, RayHit> intersect_ray_impl(
    const TransformDAG16::TraversalConstants& traversalConstants, const TransformDAG16& dag, const Path& rootPath, uint32_t rootNodeIdx, uint32_t rootTransformID, Ray& ray)
{
#define MIN_TRANSLATION_LEVEL 1

    const float3 invRayDirection = make_float3(1.0f) / ray.direction;
    // const uint8_t rayChildOrder = (ray.direction.x < 0.f ? 1 : 0) + (ray.direction.y < 0.f ? 2 : 0) + (ray.direction.z < 0.f ? 4 : 0);
    const uint8_t rayChildOrder = (signbit(ray.direction.x) | (signbit(ray.direction.y) << 1) | (signbit(ray.direction.z) << 2)) & 0b111;

    // State
    uint32_t level = RootLevel;
    Path path = rootPath;

    struct TransformStackEntry {
        uint32_t index;
        uint16_t childMask;
        uint8_t visitMask;
#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
        uint8_t transformID;
#endif // TRANSFORM_DAG_USE_TRANSFORMATION_ID
    };
    TransformStackEntry stack[RootLevel + 1];
    TransformStackEntry cache;
    Leaf cachedLeaf; // needed to iterate on the last few levels

#if TRANSFORM_DAG_USE_TRANSLATION
    struct TranslationStackEntry {
        float tmin, tmax;
        small_int3 translation;
    };

#if IS_OPTIX_KERNEL || !defined(__CUDA_ARCH__)
    TranslationStackEntry translationStack[TRANSFORM_DAG_MAX_TRANSLATION_LEVEL - MIN_TRANSLATION_LEVEL + 1];
#else // IS_OPTIX_KERNEL
    const auto threadBlock = cooperative_groups::this_thread_block();
    checkAlways(threadBlock.num_threads() == 64);
    constexpr uint32_t TranslationStackSize = (TRANSFORM_DAG_MAX_TRANSLATION_LEVEL - MIN_TRANSLATION_LEVEL + 1);
    __shared__ TranslationStackEntry translationStackMemory[64 * TranslationStackSize];
    TranslationStackEntry* translationStack = &translationStackMemory[threadBlock.thread_rank() * TranslationStackSize];
#endif // IS_OPTIX_KERNEL

    TranslationStackEntry translationCache;
    translationCache.tmin = ray.tmin;
    translationCache.tmax = ray.tmax;
    translationCache.translation = small_int3 { 0, 0, 0 };
#endif // TRANSFORM_DAG_USE_TRANSLATION

    cache.index = rootNodeIdx;
    uint16_t const* pNode = &dag.nodes[dag.levelStarts[level] + cache.index];
    cache.childMask = *pNode;

#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
    cache.transformID = rootTransformID;
    const uint8_t transformedChildMask = traversalConstants.transformMaskMappingLocalToWorld[cache.transformID][TransformDAG16::convert_child_mask(cache.childMask)];
#else // TRANSFORM_DAG_USE_TRANSFORMATION_ID
    check(rootTransformID == 0);
    const uint8_t transformedChildMask = TransformDAG16::convert_child_mask(cache.childMask);
#endif // TRANSFORM_DAG_USE_TRANSFORMATION_ID

#if TRANSFORM_DAG_USE_TRANSLATION
    cache.visitMask = transformedChildMask & compute_intersection_mask<true>(level, path, translationCache.translation, small_int3 { 0, 0, 0 }, ray, invRayDirection, translationCache.tmin, translationCache.tmax);
#else // TRANSFORM_DAG_USE_TRANSLATION
    cache.visitMask = transformedChildMask & compute_intersection_mask<true>(level, path, ray, invRayDirection);
#endif // TRANSFORM_DAG_USE_TRANSLATION

    // Traverse DAG
    for (;;) {
        // Ascend if there are no children left.
        {
            uint32 newLevel = level;
            while (newLevel < RootLevel && !cache.visitMask) {
                newLevel++;
                cache = stack[newLevel];
            }
#if TRANSFORM_DAG_USE_TRANSLATION
            if (newLevel != level && newLevel <= TRANSFORM_DAG_MAX_TRANSLATION_LEVEL)
                translationCache = translationStack[newLevel - MIN_TRANSLATION_LEVEL];
#endif // TRANSFORM_DAG_USE_TRANSLATION
            if (newLevel != level) {
                pNode = &dag.nodes[dag.levelStarts[newLevel] + cache.index];
            }
            if (newLevel == RootLevel && !cache.visitMask) {
                if constexpr (OutOfOrder)
                    return false;
                else
                    return RayHit::empty();
            }

            path.ascend(newLevel - level);
            level = newLevel;
        }
        // Find next child in order by the current ray's direction
        // const uint8 nextChild = next_child(rayChildOrder, cache.visitMask);
        const uint8 nextChild = traversalConstants.nextChildLUT[cache.visitMask][rayChildOrder];
        // Mark it as handled
        cache.visitMask &= ~(1u << nextChild);
        // Intersect that child with the ray
        {
            path.descendZYX(nextChild);
            stack[level] = cache;
#if TRANSFORM_DAG_USE_TRANSLATION
            if (level <= TRANSFORM_DAG_MAX_TRANSLATION_LEVEL) {
                translationStack[level - MIN_TRANSLATION_LEVEL] = translationCache;
            } else {
                translationCache.tmin = ray.tmin;
                translationCache.tmax = ray.tmax;
                translationCache.translation = small_int3 { 0, 0, 0 };
            }
#endif // TRANSFORM_DAG_USE_TRANSLATION
            level--;
            // If we're at the final level, we have intersected a single voxel.
            if (level == 0) {
#if TRANSFORM_DAG_USE_TRANSLATION
                ray.tmax = translationCache.tmin;
#endif // TRANSFORM_DAG_USE_TRANSLATION
                break;
            }

#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
            const uint8_t localNextChild = traversalConstants.transformChildMappingWorldToLocal[cache.transformID][nextChild];
#else // TRANSFORM_DAG_USE_TRANSFORMATION_ID
            const uint8_t localNextChild = nextChild;
#endif // TRANSFORM_DAG_USE_TRANSFORMATION_ID

            // Because we store 4x4x4 leaves, we can skip this step when we reach the 2x2x2 & 1x1x1 voxel levels.
#if TRANSFORM_DAG_USE_TRANSLATION
            small_int3 localTranslation { 0, 0, 0 };
#endif
            if (level >= dag.leafLevel) {
                auto transformPointer = dag.get_child_index(level, pNode, cache.childMask, localNextChild);
                cache.index = transformPointer.index;

#if TRANSFORM_DAG_USE_TRANSLATION
                if (level <= TRANSFORM_DAG_MAX_TRANSLATION_LEVEL) {
                    localTranslation = transformPointer.translation;
                    // Apply parent nodes transformations to the current translation.
                    const auto& parentTransform = traversalConstants.transforms[cache.transformID];
                    localTranslation = applyAxisPermutation(localTranslation, parentTransform);
                    localTranslation.xyz[0] *= (parentTransform.symmetry & 1) ? -1 : +1;
                    localTranslation.xyz[1] *= (parentTransform.symmetry & 2) ? -1 : +1;
                    localTranslation.xyz[2] *= (parentTransform.symmetry & 4) ? -1 : +1;
                }
#endif // TRANSFORM_DAG_USE_TRANSLATION

#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
                // cache.transformID = cache.transformID * transformPointer.transformID
                cache.transformID = traversalConstants.transformCombineTable[cache.transformID][transformPointer.transformID];
#endif // TRANSFORM_DAG_USE_TRANSFORMATION_ID
            }
            // Are we in an internal node?
            if (level > dag.leafLevel) {
                pNode = &dag.nodes[dag.levelStarts[level] + cache.index];
                cache.childMask = *pNode;
                cache.visitMask = TransformDAG16::convert_child_mask(cache.childMask);
            } else {
                /* The second-to-last and last levels are different: the data
                 * of these two levels (2^3 voxels) are packed densely into a
                 * single 64-bit word.
                 */
                if (level == dag.leafLevel) {
                    cachedLeaf = dag.get_leaf(cache.index);
                    cache.visitMask = cachedLeaf.get_first_child_mask();
                } else {
                    cache.visitMask = cachedLeaf.get_second_child_mask(localNextChild);
                }
            }

#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
            // Apply local to world transform.
            cache.visitMask = traversalConstants.transformMaskMappingLocalToWorld[cache.transformID][cache.visitMask];
#endif // TRANSFORM_DAG_USE_TRANSFORMATION_ID

// Perform traversal in world space (agnostic to transforms).
#if TRANSFORM_DAG_USE_TRANSLATION
            cache.visitMask = cache.visitMask & compute_intersection_mask<true>(level, path, translationCache.translation, localTranslation, ray, invRayDirection, translationCache.tmin, translationCache.tmax);
            translationCache.translation = translationCache.translation + localTranslation;
#else // TRANSFORM_DAG_USE_TRANSLATION
            cache.visitMask = cache.visitMask & compute_intersection_mask<true>(level, path, ray, invRayDirection);
#endif // TRANSFORM_DAG_USE_TRANSLATION
        }
    }

    if constexpr (OutOfOrder) {
        return true;
    } else {
        RayHit out;
        out.initEmpty();
        if (!path.is_null()) {
#if TRANSFORM_DAG_USE_TRANSLATION
            path.path = make_uint3(make_int3(path.path) + make_int3(translationCache.translation));
#endif // TRANSFORM_DAG_USE_TRANSLATION
            out.init(path, ray, invRayDirection);
        }
        return out;
    }
}

HOST_DEVICE SurfaceInteraction createSurfaceInteraction(const Ray ray, const Path& path, uint8_t materialId)
{
    // Find the face and UV coordinates of the voxel/ray intersection.
    const float3 invRayDirection = make_float3(1.0f) / ray.direction;
    const float3 boundsMin = path.as_position();
    const float3 boundsMax = boundsMin + make_float3(1.0f);
    const float3 t1 = (boundsMin - ray.origin) * invRayDirection;
    const float3 t2 = (boundsMax - ray.origin) * invRayDirection;

    const float3 dists = min(t1, t2);
    const int axis = dists.x > dists.y ? (dists.x > dists.z ? 0 : 2) : (dists.y >= dists.z ? 1 : 2);

    float t;
    SurfaceInteraction out;
    out.path = path.path;
    out.materialId = materialId;
    out.normal = make_float3(0.0f);
    out.dpdu = make_float3(0.0f);
    out.dpdv = make_float3(0.0f);
    if (axis == 0) {
        out.normal.x = (ray.direction.x < 0.0f ? +1.0f : -1.0f);
        out.dpdu.z = 1.0f;
        out.dpdv.y = 1.0f;
        t = dists.x;
    } else if (axis == 1) {
        out.normal.y = (ray.direction.y < 0.0f ? +1.0f : -1.0f);
        out.dpdu.x = 1.0f;
        out.dpdv.z = 1.0f;
        t = dists.y;
    } else if (axis == 2) {
        out.normal.z = (ray.direction.z < 0.0f ? +1.0f : -1.0f);
        out.dpdu.x = 1.0f;
        out.dpdv.y = 1.0f;
        t = dists.z;
    }
    out.position = ray.origin + t * ray.direction;
    return out;
}

template <typename TDAG, typename TDAGColors>
DEVICE float3 computeColorAtSurface(const ColorParams& colorParams, const TDAG dag, const TDAGColors colors, const SurfaceInteraction& si)
{
    Path path = si.path;
    if (path.is_null())
        return make_float3(0.0f);

    if (colorParams.debugColors == EDebugColors::White)
        return make_float3(0.8f);

    const auto invalidColor = [&]() {
        uint32 b = (path.path.x ^ path.path.y ^ path.path.z) & 0x1;
        return make_float3(1, b, 1.f - b);
    };

    if constexpr (dag_has_materials_v<TDAG>) { // Materials
#if EDITS_ENABLE_MATERIALS
        const auto materialId = si.materialId;
        if (materialId >= colorParams.materialTextures.size())
            return invalidColor();

        const int axis = std::abs(si.normal.x) > 0.5f ? 0 : (std::abs(si.normal.y) > 0.5f ? 1 : 2);
        const float3 offset = si.position - Path(path).as_position();
        float2 uv;
        if (axis == 0) {
            uv = make_float2(offset.z, offset.y);
        } else if (axis == 1) {
            uv = make_float2(offset.x, offset.z);
        } else { // axis == 2
            uv = make_float2(offset.x, offset.y);
        }

        const auto& material = colorParams.materialTextures[materialId];
        auto topPath = path;
        topPath.path.y += 1;
        cudaTextureObject_t texture;
        if (axis == 1 && !material.all.cuArray) {
            texture = material.top.cuTexture;
        } else if (axis == 1 && si.normal.y > 0 && material.top.cuArray && DAGUtils::get_value(dag, topPath)) {
            texture = material.top.cuTexture;
        } else if (axis != 1 && material.side.cuArray) {
            texture = material.side.cuTexture;
        } else {
            texture = material.all.cuTexture;
        }

        const auto tmp = tex2D<float4>(texture, uv.x, 1 - uv.y);
        return make_float3(tmp.x, tmp.y, tmp.z);
#else
        return invalidColor();
#endif
    } else if constexpr (std::is_same_v<TDAG, TransformDAG16> || std::is_same_v<TDAG, SymmetryAwareDAG16>) {
        return make_float3(0.8f);
    } else { // Materials vs colors
        if (!colors.is_valid())
            return invalidColor();

        uint64 nof_leaves = 0;
        uint32 debugColorsIndex = 0;

        uint32 colorNodeIndex = 0;
        typename TDAGColors::ColorLeaf colorLeaf = colors.get_default_leaf();

        uint32 level = 0;
        uint32 nodeIndex = dag.get_first_node_index();
        while (level < dag.leaf_level()) {
            level++;

            // Find the current childmask and which subnode we are in
            const uint32 node = dag.get_node(level - 1, nodeIndex);
            const uint8 childMask = Utils::child_mask(node);
            const uint8 child = path.child_index(level, dag.levels);

            // Make sure the node actually exists
            if (!(childMask & (1 << child)))
                return make_float3(1.0f, 0.0f, 1.0f);

            ASSUME(level > 0);
            if (level - 1 < colors.get_color_tree_levels()) {
                colorNodeIndex = colors.get_child_index(level - 1, colorNodeIndex, child);
                if (level == colors.get_color_tree_levels()) {
                    check(nof_leaves == 0);
                    colorLeaf = colors.get_leaf(colorNodeIndex);
                } else {
                    // TODO nicer interface
                    if (!colorNodeIndex)
                        return invalidColor();
                }
            }

            // Debug
            if (colorParams.debugColors == EDebugColors::Index || colorParams.debugColors == EDebugColors::Position || colorParams.debugColors == EDebugColors::ColorTree) {
                if (colorParams.debugColors == EDebugColors::Index && colorParams.debugColorsIndexLevel == level - 1) {
                    debugColorsIndex = nodeIndex;
                }
                if (level == dag.leaf_level()) {
                    if (colorParams.debugColorsIndexLevel == dag.leaf_level()) {
                        check(debugColorsIndex == 0);
                        const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
                        debugColorsIndex = childIndex;
                    }

                    if (colorParams.debugColors == EDebugColors::Index) {
                        return ColorUtils::rgb888_to_float3(Utils::murmurhash32(debugColorsIndex));
                    } else if (colorParams.debugColors == EDebugColors::Position) {
                        constexpr uint32 checkerSize = 0x7FF;
                        float color = ((path.path.x ^ path.path.y ^ path.path.z) & checkerSize) / float(checkerSize);
                        color = (color + 0.5) / 2;
                        return Utils::has_flag(nodeIndex) ? make_float3(color, 0, 0) : make_float3(color);
                    } else {
                        check(colorParams.debugColors == EDebugColors::ColorTree);
                        const uint32 offset = dag.levels - colors.get_color_tree_levels();
                        const float color = ((path.path.x >> offset) ^ (path.path.y >> offset) ^ (path.path.z >> offset)) & 0x1;
                        return make_float3(color);
                    }
                    return make_float3(0.0f, 0.0f, 0.0f);
                } else {
                    nodeIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
                    continue;
                }
            }

            //////////////////////////////////////////////////////////////////////////
            // Find out how many leafs are in the children preceding this
            //////////////////////////////////////////////////////////////////////////
            // If at final level, just count nof children preceding and exit
            if (level == dag.leaf_level()) {
                for (uint8 childBeforeChild = 0; childBeforeChild < child; ++childBeforeChild) {
                    if (childMask & (1u << childBeforeChild)) {
                        const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, childBeforeChild);
                        const Leaf leaf = dag.get_leaf(childIndex);
                        nof_leaves += Utils::popcll(leaf.to_64());
                    }
                }
                const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
                const Leaf leaf = dag.get_leaf(childIndex);
                const uint8 leafBitIndex = (((path.path.x & 0x1) == 0) ? 0 : 4) | (((path.path.y & 0x1) == 0) ? 0 : 2) | (((path.path.z & 0x1) == 0) ? 0 : 1) | (((path.path.x & 0x2) == 0) ? 0 : 32) | (((path.path.y & 0x2) == 0) ? 0 : 16) | (((path.path.z & 0x2) == 0) ? 0 : 8);
                nof_leaves += Utils::popcll(leaf.to_64() & ((uint64(1) << leafBitIndex) - 1));

                break;
            } else {
                ASSUME(level > 0);
                if (level > colors.get_color_tree_levels()) {
                    // Otherwise, fetch the next node (and accumulate leaves we pass by)
                    for (uint8 childBeforeChild = 0; childBeforeChild < child; ++childBeforeChild) {
                        if (childMask & (1u << childBeforeChild)) {
                            const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, childBeforeChild);
                            // if constexpr (std::is_same_v<TDAG, MyHashDag> || std::is_same_v<TDAG, MyGPUHashDAG<EMemoryType::GPU_Malloc>> || std::is_same_v<TDAG, MyGpuSortedDag>)
                            if constexpr (std::is_same_v<TDAG, MyGPUHashDAG<EMemoryType::GPU_Malloc>>)
                                nof_leaves += 0;
                            else
                                nof_leaves += colors.get_leaves_count(level, dag.get_node(level, childIndex));
                        }
                    }
                }
                nodeIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
            }
        }

        if (!colorLeaf.is_valid() || !colorLeaf.is_valid_index(nof_leaves)) {
            return invalidColor();
        }

        const auto compressedColor = colorLeaf.get_color(nof_leaves);
        if (colorParams.debugColors == EDebugColors::White) {
            return make_float3(0.8f);
        } else if (colorParams.debugColors == EDebugColors::ColorBits) {
            return ColorUtils::rgb888_to_float3(compressedColor.get_debug_hash());
        } else if (colorParams.debugColors == EDebugColors::MinColor) {
            return compressedColor.get_min_color();
        } else if (colorParams.debugColors == EDebugColors::MaxColor) {
            return compressedColor.get_max_color();
        } else if (colorParams.debugColors == EDebugColors::Weight) {
            return make_float3(compressedColor.get_weight());
        } else {
            return compressedColor.get_color();
        }
    }
}

// Directed towards the sun
HOST_DEVICE float3 sun_direction()
{
    return normalize(make_float3(0.3f, 1.f, 0.5f));
}
HOST_DEVICE float3 sun_color()
{
    return make_float3(0.95f, 0.94f, 0.47f); // White/yellow;
}
HOST_DEVICE float3 sky_color()
{
    return make_float3(187, 242, 250) / 255.f; // blue
}

// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations
struct RandomSample {
    float3 direction;
    float pdf;
};
HOST_DEVICE RandomSample uniformSampleHemisphere(const float2& u)
{
    const auto z = u.x;
    const auto r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    const auto phi = 2 * std::numbers::pi_v<float> * u.y;
    return {
        .direction = make_float3(r * std::cos(phi), r * std::sin(phi), z),
        .pdf = 0.5f * std::numbers::inv_pi_v<float>
    };
}
HOST_DEVICE RandomSample cosineSampleHemisphere(const float2& u)
{
    // https://www.rorydriscoll.com/2009/01/07/better-sampling/
    const float r = std::sqrt(u.x);
    const float theta = 2 * std::numbers::pi_v<float> * u.y;

    const float x = r * std::cos(theta);
    const float y = r * std::sin(theta);
    const float z = std::sqrt(std::max(0.0f, 1 - u.x));

    return {
        .direction = make_float3(x, y, z),
        .pdf = std::max(z * std::numbers::inv_pi_v<float>, 0.005f)
    };
}

HOST_DEVICE float3 applyFog(
    float3 rgb, // original color of the pixel
    const Ray ray,
    const SurfaceInteraction si,
    float fogDensity)
{
    fogDensity *= 0.00001f;
    const float fogAmount = 1.0f - exp(-length(ray.origin - si.position) * fogDensity);
    const float sunAmount = 1.01f * max(dot(ray.direction, sun_direction()), 0.0f);
    const float3 fogColor = lerp(
        sky_color(), // blue
        make_float3(1.0f), // white
        pow(sunAmount, 30.0f));
    return lerp(rgb, fogColor, clamp(fogAmount, 0.0f, 1.0f));
}

[[maybe_unused]] DEVICE static float getColorLuminance(float3 linear)
{
    const float3 sRGB = ColorUtils::approximationLinearToSRGB(linear);
    return 0.299f * sRGB.x + 0.587f * sRGB.y * 0.114f * sRGB.z;
}

} // namespace Tracer
