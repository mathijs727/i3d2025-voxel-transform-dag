#pragma once
#include "array2d.h"
#include "cuda_math.h"
#include "dags/symmetry_aware_dag/symmetry_aware_dag.h"
#include "dags/transform_dag/transform_dag.h"
#include "image.h"
#include "path.h"
#include "typedefs.h"
#include "utils.h"
#include "voxel_textures.h"
#include <cstddef>
#include <span>
#if ENABLE_OPTIX
#include <optix_host.h>
#endif

enum class EDebugColors : int {
    None,
    Index,
    Position,
    ColorTree,
    ColorBits,
    MinColor,
    MaxColor,
    Weight,
    White
};

constexpr uint32 CNumDebugColors = 9;

enum class ETool : int {
    Sphere,
    SpherePaint,
    SphereNoise,
    Cube,
    CubeCopy,
    CubeFill
};

constexpr uint32 CNumTools = 9;

struct ToolPath {
    uint3 centerPath;
    uint3 neighbourPath;
};

struct ToolInfo {
    ETool tool;
    ToolPath position;
    float radius;
    Path copySource = Path(0, 0, 0);
    Path copyDest = Path(0, 0, 0);

    ToolInfo() = default;
    ToolInfo(ETool tool, ToolPath position, float radius, uint3 copySource, uint3 copyDest)
        : tool(tool)
        , position(position)
        , radius(radius)
        , copySource(copySource)
        , copyDest(copyDest)
    {
    }

    HOST_DEVICE float3 addToolColor(const Path& path, float3 pixelColor) const
    {
        const auto addColor = [&](float toolStrength, float3 toolColor = make_float3(1, 0, 0)) {
            pixelColor = lerp(pixelColor, toolColor, clamp(100.0f * toolStrength, 0.0f, 0.5f));
        };

        switch (tool) {
        case ETool::Sphere:
        case ETool::SpherePaint:
        case ETool::SphereNoise: {
            addColor(sphere_strength(position.centerPath, path, radius));
        } break;
        case ETool::Cube:
        case ETool::CubeFill: {
            addColor(cube_strength(position.centerPath, path, radius));
        } break;
        case ETool::CubeCopy: {
            addColor(cube_strength(position.centerPath, path, radius));
            addColor(sphere_strength(copySource, path, 3), make_float3(0, 1, 0));
            addColor(sphere_strength(copyDest, path, 3), make_float3(0, 0, 1));
        } break;
        default:
            break;
        };

        return pixelColor;
    }

private:
    HOST_DEVICE static float cube_strength(const Path pos, const Path path, float radius)
    {
        return 1 - max(abs(pos.as_position() - path.as_position())) / radius;
    }
    HOST_DEVICE static float sphere_strength(const Path pos, const Path path, float radius)
    {
        return 1 - length(pos.as_position() - path.as_position()) / radius;
    }
};

namespace Tracer {
struct Ray {
    float3 origin, direction;
    float tmin, tmax;

    static HOST_DEVICE Ray create(float3 origin, float3 direction, float tmin = 0.0f, float tmax = std::numeric_limits<float>::max())
    {
        return {
            .origin = origin,
            .direction = direction,
            .tmin = tmin,
            .tmax = tmax
        };
    }
};

struct RayHit {
    Path path;
    float3 position;
    uint32_t axis : 3;
    uint32_t directionSign : 3;

    static HOST_DEVICE RayHit empty()
    {
        RayHit out;
        out.initEmpty();
        return out;
    }
    HOST_DEVICE void initEmpty()
    {
        path.path = make_uint3(0, 0, 0);
    }
    HOST_DEVICE void init(const Ray& ray, const float3& position_, const Path& path_, uint32_t normalAxis)
    {
        //   Find the face and UV coordinates of the voxel/ray intersection.
        this->path = path_;
        this->position = position_;
        this->axis = normalAxis;
        this->directionSign = (ray.direction.x < 0.0f ? 1 : 0) | (ray.direction.y < 0.0f ? 2 : 0) | (ray.direction.z < 0.0f ? 4 : 0);
    }
    HOST_DEVICE void init(const Path path_, const Ray ray, const float3 invRayDirection)
    {
        //   Find the face and UV coordinates of the voxel/ray intersection.
        this->path = path_;
        const float3 boundsMin = path.as_position();
        const float3 boundsMax = boundsMin + make_float3(1.0f);
        const float3 t1 = (boundsMin - ray.origin) * invRayDirection;
        const float3 t2 = (boundsMax - ray.origin) * invRayDirection;

        const float3 dists = min(t1, t2);
        this->axis = dists.x > dists.y ? (dists.x > dists.z ? 0u : 2u) : (dists.y > dists.z ? 1u : 2u);
        this->directionSign = (ray.direction.x < 0.0f ? 1 : 0) | (ray.direction.y < 0.0f ? 2 : 0) | (ray.direction.z < 0.0f ? 4 : 0);

        float t;
        if (this->axis == 0)
            t = dists.x;
        else if (this->axis == 1)
            t = dists.y;
        else
            t = dists.z;
        this->position = ray.origin + t * ray.direction;
    }

    HOST_DEVICE bool isEmpty() const
    {
        return path.is_null();
    }
    HOST_DEVICE bool isEmptyOrNaN() const
    {
        return isEmpty() || std::isnan(position.x) || std::isnan(position.y) || std::isnan(position.z);
    }
};

struct SurfaceInteraction {
    Path path;
    uint8_t materialId;

    float3 position;
    float3 normal;
    float3 dpdu;
    float3 dpdv;

    uint32_t diffuseColorSRGB8;

    HOST_DEVICE void initFromRayHit(const RayHit rayHit)
    {
        this->path = rayHit.path;
        if (!path.is_null()) {
            this->position = rayHit.position;
            this->dpdu = this->dpdv = this->normal = make_float3(0.0f);
            if (rayHit.axis == 0) {
                this->normal.x = ((rayHit.directionSign & 0b001) ? +1.0f : -1.0f);
                this->dpdu.z = 1.0f;
                this->dpdv.y = 1.0f;
            } else if (rayHit.axis == 1) {
                this->normal.y = ((rayHit.directionSign & 0b010) ? +1.0f : -1.0f);
                this->dpdu.x = 1.0f;
                this->dpdv.z = 1.0f;
            } else if (rayHit.axis == 2) {
                this->normal.z = ((rayHit.directionSign & 0b100) ? +1.0f : -1.0f);
                this->dpdu.x = 1.0f;
                this->dpdv.y = 1.0f;
            }
        }
    }

    HOST_DEVICE float3 transformDirectionToWorld(float3 local) const
    {
        return local.x * dpdu + local.y * dpdv + local.z * normal;
    }
};

struct TracePathsParams {
    // In
    double3 cameraPosition;
    double3 rayMin;
    double3 rayDDx;
    double3 rayDDy;

    // Out
    StaticArray2D<SurfaceInteraction> surfaceInteractionSurface;
};

template <typename TDAG>
__global__ void trace_paths(TracePathsParams traceParams, const TDAG dag);

struct ColorParams {
    StaticArray<VoxelTexturesGPU> materialTextures;
    EDebugColors debugColors;
    uint32 debugColorsIndexLevel;
};
struct TraceColorsParams {
    // In
    ColorParams colorParams;

    // In/Out
    StaticArray2D<SurfaceInteraction> surfaceInteractionSurface;
};

template <typename TDAG, typename TDAGColors>
__global__ void trace_colors(TraceColorsParams traceParams, const TDAG dag, const TDAGColors colors);

struct TraceShadowsParams {
    // In
    float shadowBias;
    StaticArray2D<SurfaceInteraction> surfaceInteractionSurface;

    // In/Out
    StaticArray2D<float> sunLightSurface;
};

template <typename TDAG>
__global__ void trace_shadows(TraceShadowsParams params, const TDAG dag);

struct TraceAmbientOcclusionParams {
    // In
    int numSamples;
    float shadowBias;
    float aoRayLength;
    uint64_t randomSeed;
    StaticArray2D<SurfaceInteraction> surfaceInteractionSurface;

    // Out
    StaticArray2D<float> aoSurface;
};
template <typename TDAG>
__global__ void trace_ambient_occlusion(TraceAmbientOcclusionParams params, const TDAG dag);

struct TraceLightingParams {
    // In
    double3 cameraPosition; // Reconstruct camera ray for fog computation.
    double3 rayMin;
    double3 rayDDx;
    double3 rayDDy;

    ToolInfo toolInfo;

    float sunBrightness;
    bool applyShadows;
    bool applyAmbientOcclusion;
    float fogDensity;
    StaticArray2D<float> aoSurface;
    StaticArray2D<float> sunLightSurface;
    StaticArray2D<SurfaceInteraction> surfaceInteractionSurface;

    // Out
    StaticArray2D<uint32_t> finalColorsSurface;
};
__global__ void trace_lighting(TraceLightingParams params);

struct TracePathTracingParams {
    // In
    double3 cameraPosition; // Reconstruct camera ray for fog computation.
    double3 rayMin;
    double3 rayDDx;
    double3 rayDDy;
    bool integratePixel;

    CUDATexture environmentMap;
    float environmentBrightness;
    bool environmentMapVisibleToPrimaryRays;

    uint64_t randomSeed;
    uint32_t numSamples;
    uint32_t maxPathDepth;
    ColorParams colorParams;
    ToolInfo toolInfo;

    // In/out
    uint32_t numAccumulatedSamples;
    StaticArray2D<float3> accumulationBuffer;

    // Out
    StaticArray2D<uint32_t> finalColorsSurface;
};
template <typename TDAG, typename TDAGColors>
__global__ void trace_path_tracing(TracePathTracingParams params, const TDAG dag, const TDAGColors colors);

struct WavefrontPathState {
    float3 throughput;
    Utils::PCG_RNG rng;
    uint2 pixel;
};
struct WaveFrontShadowState {
    uint2 pixel;
    float3 throughput;
};

__global__ void generate_rays_path_tracing(TracePathTracingParams inParams, std::span<WavefrontPathState> outPathStates, std::span<Ray> outRays);
template <typename TDAG>
__global__ void intersect_rays(const TDAG inDAG, const uint32_t* pNumRays, std::span<const Ray> inRays, std::span<RayHit> outRayHits);
__global__ void shade_and_integrate_path_tracing(
    TracePathTracingParams inParams,
    const uint32_t* pNumInPaths, std::span<const WavefrontPathState> inPathState, std::span<const RayHit> inRayHits,
    uint32_t* pNumOutPaths, std::span<WavefrontPathState> outPathState, std::span<Ray> outRemainingRays);
__global__ void accumulate_to_final_buffer(TracePathTracingParams inParams);

template <typename TDAG>
__global__ void trace_tool_path(TDAG, Ray, ToolPath*);

extern __constant__ TransformDAG16::TraversalConstants s_transformTraversalConstants;
extern __constant__ SymmetryAwareDAG16::TraversalConstants s_symmetryTraversalConstants;

struct TransformDAG16OptixParams {
    Path path;
    uint32_t nodeIdx;
    uint32_t transformID;
};
#define OPTIX_DEBUG_DRAW_SPHERES 0
#if ENABLE_OPTIX
struct OptixState_ {
    CUcontext cuContext;
    CUstream cuStream;
    OptixDeviceContext deviceContext;
};

OptixState_ createOptixState();
void destroyOptixState(const OptixState_& optixState);

struct OptixAccelerationStructure_ {
#if OPTIX_DEBUG_DRAW_SPHERES
    StaticArray<float3> sphereCenters;
    StaticArray<float> sphereRadii;
#else
    StaticArray<TransformDAG16OptixParams> subTrees;
#endif

    void* pAccelerationStructureMemory = nullptr;
    OptixTraversableHandle accelerationStructure;
};
OptixAccelerationStructure_ createOptixAccelerationStructure(const TransformDAG16& dag, const OptixState_& optixState);
void destroyOptixAccelerationStructure(OptixAccelerationStructure_&);

struct OptixProgram_ {
    OptixModule module_;
    std::vector<OptixProgramGroup> programGroups;
    OptixPipeline pipeline;

    OptixShaderBindingTable shaderBindingTableDesc;
    std::byte* pShaderBindingTable;
};
OptixProgram_ createOptixProgram(const OptixState_& optixState);
void destroyOptixProgram(OptixProgram_&);

struct OptixDenoiser_ {
    OptixDenoiser denoiser;
    std::byte *pStateMemory, *pScratchMemory;
    size_t stateMemorySizeInBytes, scratchMemorySizeInBytes;
};
OptixDenoiser_ createOptixDenoiser(const OptixState_& optixState);
void applyOptixDenoising(OptixState_ state, OptixDenoiser_& denoiser, StaticArray2D<float3> inputImage, StaticArray2D<float3> outputImage);
void destroyOptixDenoiser(OptixDenoiser_&);

#endif

}