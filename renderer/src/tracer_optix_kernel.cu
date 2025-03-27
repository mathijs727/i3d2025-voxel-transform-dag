#include "color_utils.h"
#include "cuda_math.h"
#include "cuda_random.h"
#include "tracer.h"
#include "tracer_impl.h"
#include "tracer_optix.h"
#include "utils.h"
#include <cuda.h>
#include <numbers>
#include <optix.h>

using namespace Tracer;

extern "C" __constant__ PathTracingOptixParams params;

extern "C" __global__ void __raygen__hello_world()
{
    const auto launchIndex3D = optixGetLaunchIndex();
    const auto launchIndex2D = make_uint2(launchIndex3D.x, launchIndex3D.y);
    TracePathTracingParams& traceParams = params.traceParams;
    optixThrowException(123);

    const float3 color = make_float3(1.0f, 0.0f, 0.0f);
    traceParams.finalColorsSurface.write(launchIndex2D, ColorUtils::float3_to_rgb888(ColorUtils::accurateLinearToSRGB(color)));
}

struct PathPayload {
    float3 radiance;
    Utils::PCG_RNG rng;
    unsigned depth;
};
static __forceinline__ __device__ PathPayload loadPathPayload()
{
    PathPayload out;
    out.radiance.x = __uint_as_float(optixGetPayload_0());
    out.radiance.y = __uint_as_float(optixGetPayload_1());
    out.radiance.z = __uint_as_float(optixGetPayload_2());
    out.rng.state = optixGetPayload_3();
    out.rng.state |= (uint64_t)optixGetPayload_4() << 32;
    out.rng.inc = optixGetPayload_5();
    out.rng.inc |= (uint64_t)optixGetPayload_6() << 32;
    out.depth = optixGetPayload_7();
    return out;
}

static __forceinline__ __device__ void storePathPayloadRadiance(const float3& radiance)
{
    optixSetPayload_0(__float_as_uint(radiance.x));
    optixSetPayload_1(__float_as_uint(radiance.y));
    optixSetPayload_2(__float_as_uint(radiance.z));
}

[[maybe_unused]] static __forceinline__ __device__ Path getCurrentIntersectionPath()
{
    Path out;
    out.path.x = optixGetAttribute_0();
    out.path.y = optixGetAttribute_1();
    out.path.z = optixGetAttribute_2();
    return out;
}
extern "C" __global__ void __intersection__TransformDAG16()
{
    const TransformDAG16OptixParams intersectParams = params.pSubTrees[optixGetPrimitiveIndex()];

    Ray ray;
    ray.origin = optixGetWorldRayOrigin();
    ray.direction = optixGetWorldRayDirection();
    ray.tmin = optixGetRayTmin();
    ray.tmax = optixGetRayTmax();

    const RayHit rayHit = intersect_ray_impl<false, OptixLevel>(params.dagTraversalConstants, params.dag, intersectParams.path, intersectParams.nodeIdx, intersectParams.transformID, ray);
    if (!rayHit.isEmpty()) {
        optixReportIntersection(ray.tmax, 0, rayHit.path.path.x, rayHit.path.path.y, rayHit.path.path.z);
    }
}

static __forceinline__ __device__ float3 intersectRadiance(const Ray& ray, const PathPayload& currentPayload)
{
    if (currentPayload.depth < params.traceParams.maxPathDepth) {
        unsigned p0, p1, p2, p3, p4, p5, p6, p7;
        p3 = uint32_t(currentPayload.rng.state);
        p4 = uint32_t(currentPayload.rng.state >> 32);
        p5 = uint32_t(currentPayload.rng.inc);
        p6 = uint32_t(currentPayload.rng.inc >> 32);
        p7 = currentPayload.depth + 1;
        optixTrace(OPTIX_PAYLOAD_TYPE_DEFAULT, params.accelerationStructure,
            ray.origin, ray.direction, ray.tmin, ray.tmax,
            0.0f, // ray time
            OptixVisibilityMask(255), // Always visible
            OPTIX_RAY_FLAG_NONE,
            0, // SBT offset
            0, // SBT stride
            0, // missSBTIndex
            p0, p1, p2, p3, p4, p5, p6, p7);
        return make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
    } else {
        return make_float3(0.0f);
    }
}

extern "C" __global__ void __raygen__camera()
{
    const auto launchIndex3D = optixGetLaunchIndex();
    const auto launchIndex2D = make_uint2(launchIndex3D.x, launchIndex3D.y);
    const auto launchDimensions3D = optixGetLaunchDimensions();
    TracePathTracingParams& traceParams = params.traceParams;

    const uint64_t seed = (launchIndex2D.y * launchDimensions3D.x + launchIndex2D.x) * std::max(traceParams.randomSeed, (uint64_t)1);
    PathPayload payload {
        .rng = Utils::PCG_RNG(seed),
        .depth = 0
    };

    const float2 cameraSample = make_float2(launchIndex2D) + (traceParams.integratePixel ? payload.rng.sampleFloat2() : make_float2(0.0f));
    const float3 cameraRayOrigin = make_float3(traceParams.cameraPosition);
    const float3 cameraRayDirection = make_float3(normalize(traceParams.rayMin + cameraSample.x * traceParams.rayDDx + (imageHeight - 1 - cameraSample.y) * traceParams.rayDDy - traceParams.cameraPosition));
    const Ray cameraRay = Ray::create(cameraRayOrigin, cameraRayDirection);

    float3 accumulatedLinearColor = traceParams.accumulationBuffer.read(launchIndex2D);
    for (unsigned i = 0; i < traceParams.numSamples; ++i)
        accumulatedLinearColor = accumulatedLinearColor + intersectRadiance(cameraRay, payload); // Perform path tracing.
    traceParams.accumulationBuffer.write(launchIndex2D, accumulatedLinearColor);

    const float3 linearColor = accumulatedLinearColor / make_float3(float(traceParams.numAccumulatedSamples));
    const uint32_t srgbColor = ColorUtils::float3_to_rgb888(ColorUtils::accurateLinearToSRGB(linearColor));
    traceParams.finalColorsSurface.write(launchIndex2D, srgbColor);
}

extern "C" __global__ void __miss__background()
{
    storePathPayloadRadiance(params.traceParams.environmentBrightness * Tracer::sky_color());
}

[[maybe_unused]] static __forceinline__ __device__ void constructOrthonormalBasis(
    const float3& normal, float3& tangent, float3& bitangent)
{
    if (normal.x > 0.5f)
        tangent = normalize(cross(normal, make_float3(0, 0, 1)));
    else
        tangent = normalize(cross(normal, make_float3(1, 0, 0)));
    bitangent = cross(normal, tangent);
}

template <bool SupportInstancing>
static __forceinline__ __device__ SurfaceInteraction getCurrentSphereSurfaceInteraction()
{
    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDirection = optixGetWorldRayDirection();
    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();

    SurfaceInteraction si;
    si.position = rayOrigin + tmax * rayDirection;

    const unsigned int prim_idx = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int sbtGASIndex = optixGetSbtGASIndex();
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    float4 q;
    optixGetSphereData(gas, prim_idx, sbtGASIndex, 0.f, &q);
    if constexpr (SupportInstancing) {
        const float3 objectIntersectPos = optixTransformPointFromWorldToObjectSpace(si.position);
        const float3 objectNormal = (objectIntersectPos - make_float3(q)) / q.w;
        si.normal = normalize(optixTransformNormalFromObjectToWorldSpace(objectNormal));
    } else {
        si.normal = normalize(si.position - make_float3(q));
    }
    constructOrthonormalBasis(si.normal, si.dpdu, si.dpdv);
    return si;
}

[[maybe_unused]] static __forceinline__ __device__ bool intersectShadow(const Ray& ray)
{
    unsigned shadowIntersect;
    optixTrace(OPTIX_PAYLOAD_TYPE_DEFAULT, params.accelerationStructure,
        ray.origin, ray.direction, ray.tmin, ray.tmax,
        0.0f, // ray time
        OptixVisibilityMask(255), // Always visible
        OPTIX_RAY_FLAG_NONE,
        1, // SBT offset
        0, // SBT stride
        1, // missSBTIndex
        shadowIntersect);
    return shadowIntersect;
}

extern "C" __global__ void __closesthit__pathTracing()
{
    // Epsilon to prevent self intersection.
    constexpr float epsilon = 0.1f;

    // const TracePathTracingParams& traceParams = params.traceParams;
    PathPayload payload = loadPathPayload();

#if OPTIX_DEBUG_DRAW_SPHERES
    const SurfaceInteraction si = getCurrentSphereSurfaceInteraction<false>();
#else
    const auto path = getCurrentIntersectionPath();
    const Ray incomingRay {
        .origin = optixGetWorldRayOrigin(),
        .direction = optixGetWorldRayDirection(),
        .tmin = optixGetRayTmin(),
        .tmax = optixGetRayTmax()
    };
    const SurfaceInteraction si = createSurfaceInteraction(incomingRay, path, 0);
#endif

    const float3 materialColor = make_float3(std::numbers::inv_pi_v<float>, std::numbers::inv_pi_v<float>, std::numbers::inv_pi_v<float>) * 0.8f;
    if (isnan(si.position.x) || isnan(si.position.y) || isnan(si.position.z))
        return;

    const auto evaluteBRDF = [&](const float3& direction) {
        const float NdotL = max(0.0f, dot(si.normal, direction));
        return NdotL * materialColor;
    };

    // Shadow ray (Next-Event Estimation).
    float3 Lo = make_float3(0.0f);
    // const Ray shadowRay = Ray::create(si.position + epsilon * si.normal, sun_direction());
    // if (!intersectShadow(shadowRay)) {
    //     const float3 brdf = evaluteBRDF(shadowRay.direction);
    //     Lo = Lo + brdf * sun_color() * traceParams.environmentBrightness;
    // }

    // Continuation ray.
    auto continuationSample = cosineSampleHemisphere(payload.rng.sampleFloat2());
    const float3 continuationDirection = si.transformDirectionToWorld(continuationSample.direction);
    const Ray continuationRay = Ray::create(si.position + epsilon * si.normal, continuationDirection);
    const float3 brdf = evaluteBRDF(continuationRay.direction);
    const float3 Li = intersectRadiance(continuationRay, payload);
    // const float3 Li = make_float3(rnd(payload.seed), rnd(payload.seed), rnd(payload.seed));
    // const float3 Li = abs(continuationRay.direction);
    Lo = Lo + (Li * brdf) / continuationSample.pdf;

    storePathPayloadRadiance(Lo);
}

extern "C" __global__ void __closesthit__shadow()
{
    optixSetPayload_0(1);
}

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(0);
}

// Copied from:
// https://github.com/NVIDIA/OptiX_Apps/blob/master/apps/nvlink_shared/shaders/exception.cu
extern "C" __global__ void __exception__all()
{
    const uint3 launchIndex = optixGetLaunchIndex();
    const int exceptionCode = optixGetExceptionCode();

    printf("Exception %i at (%u, %u)\n", exceptionCode, launchIndex.x, launchIndex.y);
}
