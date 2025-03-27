#include "dags/basic_dag/basic_dag.h"
#include "dags/dag_utils.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/hash_dag/hash_dag_colors.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "dags/symmetry_aware_dag/symmetry_aware_dag.h"
#include "dags/transform_dag/transform_dag.h"
#include "memory.h"
#include "tracer.h"
#include "tracer_impl.h"
#include "utils.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <cuda.h>
#include <cuda_math.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numbers>
#include <span>
#include <spdlog/spdlog.h>
#include <stack>
#include <tuple>
#include <type_traits>
#include <vector>

#if ENABLE_OPTIX
#include "tracer_optix.h"
#include <cooperative_groups.h>
#include <optix.h>
#endif

namespace Tracer {

__constant__ TransformDAG16::TraversalConstants s_transformTraversalConstants;
__constant__ SymmetryAwareDAG16::TraversalConstants s_symmetryTraversalConstants;

template <bool OutOfOrder>
DEVICE std::conditional_t<OutOfOrder, bool, RayHit> intersect_ray_impl(const TransformDAG16& dag, Ray& ray)
{
    return intersect_ray_impl<OutOfOrder, dag.levels>(s_transformTraversalConstants, dag, Path(0, 0, 0), dag.get_first_node_index(), 0, ray);
}

template <>
DEVICE RayHit intersect_ray(const TransformDAG16& dag, Ray& ray, uint32_t& materialId)
{
    materialId = 0;
    return intersect_ray_impl<false>(dag, ray);
}
template <>
DEVICE bool intersect_ray_node_out_of_order(const TransformDAG16& dag, Ray ray)
{
    return intersect_ray_impl<true>(dag, ray);
}

DEVICE uint8_t reorderChildMask(uint8_t inMask)
{
    uint8_t outMask = 0;
    for (uint32_t c = 0; c < 7; ++c) {
        if ((inMask >> c) & 0b1)
            outMask |= (uint8_t)1 << (7 - c);
    }
    return outMask;
}

template <bool OutOfOrder>
DEVICE std::conditional_t<OutOfOrder, bool, RayHit> intersect_ray_impl(const SymmetryAwareDAG16& dag, Ray& ray)
{
    const float3 invRayDirection = make_float3(1.0f) / ray.direction;
    const uint8_t rayChildOrder = (ray.direction.x < 0.f ? 1 : 0) + (ray.direction.y < 0.f ? 2 : 0) + (ray.direction.z < 0.f ? 4 : 0);
    // const uint8_t rayChildOrder = (signbit(ray.direction.x) | (signbit(ray.direction.y) << 1) | (signbit(ray.direction.z) << 2)) & 0b111;

    constexpr uint32_t RootLevel = SCENE_DEPTH;

    // State
    uint32_t level = RootLevel;
    Path path = Path(0, 0, 0);

    struct SymmetryStackEntry {
        uint32_t index;
        uint16_t childMask;
        uint8_t visitMask;
        uint8_t symmetry;
    };
    SymmetryStackEntry stack[RootLevel + 1];
    SymmetryStackEntry cache;
    Leaf cachedLeaf; // needed to iterate on the last few levels

    cache.index = dag.get_first_node_index();
    uint16_t const* pNode = &dag.nodes[dag.levelStarts[level] + cache.index];
    cache.childMask = *pNode;
    const uint8_t childMask8 = SymmetryAwareDAG16::convert_child_mask(cache.childMask);
    cache.visitMask = childMask8 & compute_intersection_mask<true>(level, path, ray, invRayDirection);
    cache.symmetry = 0;

    // Traverse DAG
    for (;;) {
        // Ascend if there are no children left.
        {
            uint32 newLevel = level;
            while (newLevel < RootLevel && !cache.visitMask) {
                newLevel++;
                cache = stack[newLevel];
            }
            if (newLevel != level)
                pNode = &dag.nodes[dag.levelStarts[newLevel] + cache.index];
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
        const uint8 nextChild = next_child(rayChildOrder, cache.visitMask);
        // Mark it as handled
        cache.visitMask &= ~(1u << nextChild);
        // Intersect that child with the ray
        {
            path.descendZYX(nextChild);
            stack[level] = cache;
            level--;
            // If we're at the final level, we have intersected a single voxel.
            if (level == 0)
                break;

            // Because we store 4x4x4 leaves, we can skip this step when we reach the 2x2x2 & 1x1x1 voxel levels.
            if (level >= dag.leafLevel) {
                auto symmetryPointer = dag.get_child_index(level, pNode, cache.childMask, nextChild ^ cache.symmetry);
                cache.index = symmetryPointer.index;
                cache.symmetry ^= symmetryPointer.symmetry;
            }
            // Are we in an internal node?
            if (level > dag.leafLevel) {
                pNode = &dag.nodes[dag.levelStarts[level] + cache.index];
                cache.childMask = *pNode;
                cache.visitMask = SymmetryAwareDAG16::convert_child_mask(cache.childMask);
            } else {
                /* The second-to-last and last levels are different: the data
                 * of these two levels (2^3 voxels) are packed densely into a
                 * single 64-bit word.
                 */
                if (level == dag.leafLevel) {
                    cachedLeaf = dag.get_leaf(cache.index);
                    cache.visitMask = cachedLeaf.get_first_child_mask();
                } else {
                    cache.visitMask = cachedLeaf.get_second_child_mask(nextChild ^ cache.symmetry);
                }
            }
            // Apply symmetry to the visit mask.
            cache.visitMask = s_symmetryTraversalConstants.symmetryChildMask[cache.symmetry][cache.visitMask];

            cache.visitMask = cache.visitMask & compute_intersection_mask<true>(level, path, ray, invRayDirection);
        }
    }

    if constexpr (OutOfOrder) {
        return true;
    } else {
        RayHit out;
        out.initEmpty();
        if (!path.is_null())
            out.init(path, ray, invRayDirection);
        return out;
    }
}

template <>
DEVICE RayHit intersect_ray(const SymmetryAwareDAG16& dag, Ray& ray, uint32_t& materialId)
{
    materialId = 0;
    return intersect_ray_impl<false>(dag, ray);
}
template <>
DEVICE bool intersect_ray_node_out_of_order(const SymmetryAwareDAG16& dag, Ray ray)
{
    return intersect_ray_impl<true>(dag, ray);
}

template <typename TDAG>
__global__ void trace_paths(TracePathsParams traceParams, const TDAG dag)
{
    // Target pixel coordinate
    const uint2 pixel = make_uint2(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside.

    // Pre-calculate per-pixel data
    const float3 rayOrigin = make_float3(traceParams.cameraPosition);
    const float3 rayDirection = make_float3(normalize(traceParams.rayMin + pixel.x * traceParams.rayDDx + pixel.y * traceParams.rayDDy - traceParams.cameraPosition));
    Ray ray = Ray::create(rayOrigin, rayDirection);

    uint32_t materialId;
    const auto rayHit = intersect_ray(dag, ray, materialId);
    SurfaceInteraction si;
    si.initFromRayHit(rayHit);
    traceParams.surfaceInteractionSurface.write(pixel.x, imageHeight - 1 - pixel.y, si);
}

template <typename TDAG, typename TDAGColors>
__global__ void trace_colors(TraceColorsParams traceParams, const TDAG dag, const TDAGColors colors)
{
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside

    SurfaceInteraction& si = traceParams.surfaceInteractionSurface.getPixel(pixel);
    si.diffuseColorSRGB8 = ColorUtils::float3_to_rgb888(
        computeColorAtSurface(traceParams.colorParams, dag, colors, si));
}

template <typename TDAG>
__global__ void trace_shadows(TraceShadowsParams params, const TDAG dag)
{
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside

    const SurfaceInteraction& si = params.surfaceInteractionSurface.getPixel(pixel);
    if (si.path.is_null())
        return;

    const Ray shadowRay = Ray::create(si.position + params.shadowBias * sun_direction(), sun_direction());
    const bool sunOccluded = intersect_ray_node_out_of_order(dag, shadowRay);
    const float NdotL = std::max(dot(si.normal, shadowRay.direction), 0.0f);
    params.sunLightSurface.write(pixel, sunOccluded ? 0.0f : NdotL);
}

template <typename TDAG>
__global__ void trace_ambient_occlusion(TraceAmbientOcclusionParams traceParams, const TDAG dag)
{
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside

    const SurfaceInteraction& si = traceParams.surfaceInteractionSurface.getPixel(pixel);
    if (si.path.is_null())
        return;

    const uint64_t seed = (pixel.y * imageWidth + pixel.x) * std::max(traceParams.randomSeed, (uint64_t)1);
    Utils::PCG_RNG rng { seed };
    uint32_t numSamples = 0, numHits = 0;
    for (uint32_t i = 0; i < traceParams.numSamples; ++i) {
        const float2 u = rng.sampleFloat2();
        const auto hemisphereSample = cosineSampleHemisphere(u);
        const float3 worldDirection = si.transformDirectionToWorld(hemisphereSample.direction);
        Ray shadowRay = Ray::create(si.position + si.normal * traceParams.shadowBias, worldDirection);
        shadowRay.tmax = traceParams.aoRayLength;

        const float NdotL = max(dot(si.normal, shadowRay.direction), 0.0f);
        if (!intersect_ray_node_out_of_order(dag, shadowRay)) {
            numHits += 1;
        }
        numSamples += 1;
    }

    const float ambientOcclusion = (float)numHits / (float)numSamples;
    traceParams.aoSurface.write(pixel.x, pixel.y, ambientOcclusion);
}

__global__ void trace_lighting(TraceLightingParams params)
{
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside

    const SurfaceInteraction si = params.surfaceInteractionSurface.getPixel(pixel);
    float3 shadedColor;
    if (si.path.is_null()) {
        shadedColor = sky_color();
    } else {
#if EDITS_ENABLE_COLORS || EDITS_ENABLE_MATERIALS
        const float3 materialKd_sRGB = ColorUtils::rgb888_to_float3(si.diffuseColorSRGB8);
        const float3 materialKd = ColorUtils::accurateSRGBToLinear(materialKd_sRGB);
#else
        const float3 materialKd = make_float3(0.8f * glm::one_over_pi<float>());
#endif

        constexpr float ambientColorFactor = 0.5f;

        const float ambientOcclusion = params.applyAmbientOcclusion ? params.aoSurface.read(pixel) : 1.0f;
        const float sunVisibility = params.applyShadows ? params.sunLightSurface.read(pixel) : 1.0f;
        const float3 ambientLightContribution = make_float3(ambientOcclusion) * ambientColorFactor * sky_color();
        const float3 directLightContribution = sunVisibility * max(dot(si.normal, sun_direction()), 0.0f) * sun_color() * params.sunBrightness;
        shadedColor = (ambientLightContribution + directLightContribution) * materialKd;

        if (params.fogDensity > 0.0f) {
            // Reconstruct camera ray.
            const float3 rayOrigin = make_float3(params.cameraPosition);
            const float3 rayDirection = make_float3(normalize(params.rayMin + pixel.x * params.rayDDx + pixel.y * params.rayDDy - params.cameraPosition));
            const Ray cameraRay = Ray::create(rayOrigin, rayDirection);
            shadedColor = applyFog(shadedColor, cameraRay, si, params.fogDensity);
        }
    }

#ifdef TOOL_OVERLAY
    if (!si.path.is_null())
        shadedColor = params.toolInfo.addToolColor(si.path, shadedColor);
    shadedColor = min(shadedColor, make_float3(1.0f));
#endif

    const uint32 finalColor = ColorUtils::float3_to_rgb888(
        ColorUtils::accurateLinearToSRGB(shadedColor));
    params.finalColorsSurface.write(pixel.x, pixel.y, finalColor);
}

// https://pbr-book.org/3ed-2018/Color_and_Radiometry/Working_with_Radiometric_Integrals#SphericalPhi
DEVICE float SphericalTheta(const float3 v)
{
    return std::acos(clamp(v.y, -1.0f, 1.0f));
}
DEVICE float SphericalPhi(const float3 v)
{
    float p = std::atan2(v.z, v.x);
    return (p < 0) ? (p + 2 * std::numbers::pi_v<float>) : p;
}
DEVICE float2 dirToUV(const float3 dir)
{
    const float s = SphericalPhi(-dir) * 0.5f * std::numbers::inv_pi_v<float>;
    const float t = 1.0f - SphericalTheta(dir) * std::numbers::inv_pi_v<float>;
    return make_float2(s, t);
}

DEVICE float3 getEnvironmentColor(const TracePathTracingParams& traceParams, const Ray& ray, bool isPrimaryRay)
{
    if (traceParams.environmentMap.cuArray) {
        if (isPrimaryRay && !traceParams.environmentMapVisibleToPrimaryRays) {
            return make_float3(1.0f);
        } else {
            const float2 uv = dirToUV(ray.direction);
            return make_float3(tex2D<float4>(traceParams.environmentMap.cuTexture, uv.x, uv.y)) * traceParams.environmentBrightness;
        }
    } else {
        return sky_color() * traceParams.environmentBrightness;
    }
}

template <typename TDAG, typename TDAGColors>
__global__ void trace_path_tracing(TracePathTracingParams traceParams, const TDAG dag, const TDAGColors colors)
{
    // Target pixel coordinate
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside.

    // Epsilon to prevent self intersection.
    constexpr float epsilon = 0.1f;

    // Create camera ray
    const uint64_t seed = (pixel.y * imageWidth + pixel.x) * std::max(traceParams.randomSeed, (uint64_t)1);
    Utils::PCG_RNG rng { seed };
    float3 Lo = traceParams.accumulationBuffer.read(pixel);
    for (uint32_t sample = 0; sample < traceParams.numSamples; ++sample) {
        const float2 cameraSample = make_float2(pixel.x, pixel.y) + (traceParams.integratePixel ? rng.sampleFloat2() : make_float2(0.0f, 0.0f));
        const float3 cameraRayOrigin = make_float3(traceParams.cameraPosition);
        const float3 cameraRayDirection = make_float3(normalize(traceParams.rayMin + cameraSample.x * traceParams.rayDDx + (imageHeight - 1 - cameraSample.y) * traceParams.rayDDy - traceParams.cameraPosition));
        Ray continuationRay = Ray::create(cameraRayOrigin, cameraRayDirection);

        float3 currentPathContribution = make_float3(1.0f);
        for (uint32_t pathDepth = 0; pathDepth < traceParams.maxPathDepth; ++pathDepth) {
            uint32_t materialId;
            const RayHit rayHit = intersect_ray(dag, continuationRay, materialId);

            // Stop recursion if we didn't hit any geometry (ray goes into the skybox).
            if (rayHit.isEmpty()) {
                Lo = Lo + currentPathContribution * getEnvironmentColor(traceParams, continuationRay, pathDepth == 0);
                break;
            }

            SurfaceInteraction si;
            si.initFromRayHit(rayHit);
            // const float3 materialKd = computeColorAtSurface(traceParams.colorParams, dag, colors, si) * glm::one_over_pi<float>();
            float3 materialKd = make_float3(glm::one_over_pi<float>(), glm::one_over_pi<float>(), glm::one_over_pi<float>()) * 0.8f;
#ifdef TOOL_OVERLAY
            if (!si.path.is_null())
                materialKd = traceParams.toolInfo.addToolColor(si.path, materialKd);
#endif // TOOL_OVERLAY

            const auto evaluteBRDF = [&](const float3& direction) {
                const float NdotL = max(0.0f, dot(si.normal, direction));
                return NdotL * materialKd;
            };

            // Shadow ray (Next-Event Estimation).
            // const Ray shadowRay = Ray::create(si.position + epsilon * si.normal, sun_direction());
            // if (!intersect_ray_node_out_of_order(dag, shadowRay)) {
            //    const float3 brdf = evaluteBRDF(shadowRay.direction);
            //    Lo = Lo + currentPathContribution * brdf * sun_color() * traceParams.environmentBrightness;
            //}

            // Continuation ray.
            auto continuationSample = cosineSampleHemisphere(rng.sampleFloat2());
            const float3 continuationDirection = si.transformDirectionToWorld(continuationSample.direction);
            continuationRay = Ray::create(si.position + epsilon * si.normal, continuationDirection);
            const float3 brdf = evaluteBRDF(continuationRay.direction);
            currentPathContribution = currentPathContribution * brdf / continuationSample.pdf;

            /*// Russian Roulette.
            const float russianRouletteProbability = clamp(getColorLuminance(currentPathContribution), 0.01f, 1.0f); // Must be in the range of 0 to 1
            if (rng.sampleFloat() > russianRouletteProbability)
                break;
            currentPathContribution = currentPathContribution / russianRouletteProbability; */
        }
    }

    traceParams.accumulationBuffer.write(pixel, Lo);

    const float3 linearColor = Lo / make_float3(traceParams.numAccumulatedSamples);
    const float3 linearToneMappedColor = ColorUtils::aces(linearColor);
    const auto srgbToneMappedColor = ColorUtils::accurateLinearToSRGB(linearToneMappedColor);
    traceParams.finalColorsSurface.write(pixel, ColorUtils::float3_to_rgb888(srgbToneMappedColor));
}

__global__ void generate_rays_path_tracing(TracePathTracingParams traceParams, std::span<WavefrontPathState> outPathStates, std::span<Ray> outRays)
{
    // Target pixel coordinate
    const auto grid = cooperative_groups::this_grid();
    const auto threadIndex = grid.thread_index();
    const auto pixel = make_uint2(threadIndex.x, threadIndex.y);
    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside.

    // Initialize random seed.
    const auto threadRank = pixel.y * imageWidth + pixel.x;
    const uint64_t seed = threadRank * std::max(traceParams.randomSeed, (uint64_t)1);
    WavefrontPathState& pathState = outPathStates[threadRank];
    pathState.rng.init(seed);
    pathState.pixel = pixel;
    pathState.throughput = make_float3(1.0f, 1.0f, 1.0f);

    // Create camera ray
    const float2 cameraSample = make_float2(pixel.x, pixel.y) + (traceParams.integratePixel ? pathState.rng.sampleFloat2() : make_float2(0.0f, 0.0f));
    const float3 cameraRayOrigin = make_float3(traceParams.cameraPosition);
    const float3 cameraRayDirection = make_float3(normalize(traceParams.rayMin + cameraSample.x * traceParams.rayDDx + (imageHeight - 1 - cameraSample.y) * traceParams.rayDDy - traceParams.cameraPosition));
    outRays[threadRank] = Ray::create(cameraRayOrigin, cameraRayDirection);
}

static __device__ float length2(float3 v) {
    return dot(v, v);
}

template <typename TDAG>
__global__ void intersect_rays(const TDAG inDAG, const uint32_t* pNumRays, std::span<const Ray> inRays, std::span<RayHit> outRayHits)
{
    const auto grid = cooperative_groups::this_grid();
    const auto threadRank = grid.thread_rank();

    // Persistent threads.
    if (threadRank >= *pNumRays)
        return;

    uint32_t materialId;
    Ray ray = inRays[threadRank];
    outRayHits[threadRank] = intersect_ray(inDAG, ray, materialId);
    

}

__global__ void shade_and_integrate_path_tracing(
    TracePathTracingParams inParams,
    const uint32_t* pNumInPaths, std::span<const WavefrontPathState> inPathState, std::span<const RayHit> inRayHits,
    uint32_t* pNumOutPaths, std::span<WavefrontPathState> outPathState, std::span<Ray> outRemainingRays)
{
    const auto grid = cooperative_groups::this_grid();
    const auto threadRank = grid.thread_rank();
    if (threadRank >= *pNumInPaths)
        return;

    WavefrontPathState pathState = inPathState[threadRank];
    const auto addContribution = [&](float3 radiance) {
        float3 Lo = inParams.accumulationBuffer.read(pathState.pixel);
        Lo = Lo + radiance;
        inParams.accumulationBuffer.write(pathState.pixel, Lo);
    };

    const RayHit rayHit = inRayHits[threadRank];
    if (rayHit.path.is_null()) {
        addContribution(pathState.throughput * sky_color() * inParams.environmentBrightness);
        return;
    }

    SurfaceInteraction si;
    si.initFromRayHit(rayHit);
    float3 materialKd = make_float3(glm::one_over_pi<float>(), glm::one_over_pi<float>(), glm::one_over_pi<float>()) * 0.8f;
#ifdef TOOL_OVERLAY
    if (!rayHit.path.is_null())
        materialKd = inParams.toolInfo.addToolColor(rayHit.path, materialKd);
#endif // TOOL_OVERLAY

    const auto evaluteBRDF = [&](const float3& direction) {
        const float NdotL = max(0.0f, dot(si.normal, direction));
        return NdotL * materialKd;
    };

    constexpr float epsilon = 0.05f;
    // Shadow ray (Next-Event Estimation).
    // const Ray shadowRay = Ray::create(si.position + epsilon * si.normal, sun_direction());
    // if (!intersect_ray_node_out_of_order(dag, shadowRay)) {
    //    const float3 brdf = evaluteBRDF(shadowRay.direction);
    //    Lo = Lo + currentPathContribution * brdf * sun_color() * traceParams.environmentBrightness;
    //}

    // Continuation ray.
    auto continuationSample = cosineSampleHemisphere(pathState.rng.sampleFloat2());
    const float3 continuationDirection = si.transformDirectionToWorld(continuationSample.direction);
    const float3 brdf = evaluteBRDF(continuationDirection);
    // addContribution(brdf);

    const uint32_t outIdx = atomicAdd(pNumOutPaths, 1);
    pathState.throughput = pathState.throughput * (brdf / continuationSample.pdf);
    outPathState[outIdx] = pathState;
    outRemainingRays[outIdx] = Ray::create(si.position + epsilon * si.normal, continuationDirection, epsilon);
}

__global__ void accumulate_to_final_buffer(TracePathTracingParams inParams)
{
    // Target pixel coordinate
    const auto grid = cooperative_groups::this_grid();
    const auto threadIndex = grid.thread_index();
    const auto pixel = make_uint2(threadIndex.x, threadIndex.y);
    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside.

    const auto accumulatedLinearColor = inParams.accumulationBuffer.read(pixel);
    auto linearColor = accumulatedLinearColor / make_float3(inParams.numAccumulatedSamples);
#if OPTIX_DENOISING
    inParams.finalColorsSurface.write(pixel, linearColor);
#else

    const auto srgbColor = ColorUtils::accurateLinearToSRGB(linearColor);
    inParams.finalColorsSurface.write(pixel, ColorUtils::float3_to_rgb888(srgbColor));
#endif
}

template <typename TDAG>
__global__ void trace_tool_path(TDAG dag, Ray ray, ToolPath* pOutput)
{
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0)
        return;

    uint32_t materialId;
    const RayHit rayHit = intersect_ray(dag, ray, materialId);
    if (rayHit.isEmpty()) {
        pOutput->centerPath = make_uint3(0);
        pOutput->neighbourPath = make_uint3(0);
        return;
    }

    SurfaceInteraction si;
    si.initFromRayHit(rayHit);

    const uint3 centerVoxel = si.path.path;
    const uint3 neighbourVoxel = make_uint3(make_int3(centerVoxel) + make_int3(si.normal));
    pOutput->centerPath = centerVoxel;
    pOutput->neighbourPath = neighbourVoxel;
}

#define DAG_IMPLS(Dag)                                                                                           \
    template __global__ void trace_paths<Dag>(TracePathsParams, Dag);                                            \
    template __global__ void trace_shadows<Dag>(TraceShadowsParams, Dag);                                        \
    template __global__ void trace_ambient_occlusion<Dag>(TraceAmbientOcclusionParams, Dag);                     \
    template __global__ void intersect_rays<Dag>(Dag, const uint32_t*, std::span<const Ray>, std::span<RayHit>); \
    template __global__ void trace_tool_path<Dag>(Dag, Ray, ToolPath*);

#define DAG_COLOR_IMPLS(Dag, Colors)                                                    \
    template __global__ void trace_colors<Dag, Colors>(TraceColorsParams, Dag, Colors); \
    template __global__ void trace_path_tracing<Dag, Colors>(TracePathTracingParams, Dag, Colors);

// Work-around for not supporting comma in #define (which don't understand C++ syntax) :-(
using MyGPUHashDAG_T = MyGPUHashDAG<EMemoryType::GPU_Malloc>;

DAG_IMPLS(BasicDAG)
DAG_IMPLS(HashDAG)
DAG_IMPLS(TransformDAG16)
DAG_IMPLS(SymmetryAwareDAG16)
DAG_IMPLS(MyGPUHashDAG_T)

DAG_COLOR_IMPLS(BasicDAG, BasicDAGUncompressedColors)
DAG_COLOR_IMPLS(BasicDAG, BasicDAGCompressedColors)
DAG_COLOR_IMPLS(BasicDAG, BasicDAGColorErrors)
DAG_COLOR_IMPLS(TransformDAG16, BasicDAGColorErrors)
DAG_COLOR_IMPLS(SymmetryAwareDAG16, BasicDAGColorErrors)
DAG_COLOR_IMPLS(HashDAG, HashDAGColors)
DAG_COLOR_IMPLS(MyGPUHashDAG_T, HashDAGColors)

} // namespace Tracer
