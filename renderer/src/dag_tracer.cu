#include "dag_tracer.h"
#include "typedefs.h"
//
#include "cuda_error_check.h"
#include "cuda_helpers.h"
#include "dag_tracer.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/dag_utils.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/hash_dag/hash_dag_colors.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "dags/symmetry_aware_dag/symmetry_aware_dag.h"
#include "dags/transform_dag/transform_dag.h"
#include "events.h"
#include "image.h"
#include "memory.h"
#include "tracer.h"
#include "tracer_impl.h"
#include <limits>
#include <span>

#if ENABLE_OPTIX
#include "tracer_optix.h"
#include <cooperative_groups.h>
#include <optix_host.h>
#include <optix_stubs.h>
#endif

DAGTracer::DAGTracer(bool headLess, EventsManager* eventsManager)
    : headLess(headLess)
    , eventsManager(eventsManager)
{
    colorsBuffer = StaticArray2D<uint32_t>::allocate("colors buffer", imageWidth, imageHeight);

    if (!headLess) {
        glGenTextures(1, &colorsImage);
        glBindTexture(GL_TEXTURE_2D, colorsImage);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int32)imageWidth, (int32)imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
        colorsSurface = GLSurface2D::create(colorsImage);
    }
    // We cannot use GPU managed memory to read data from the GPU without requiring a full device synchronization on Windows:
    // "Applications running on Windows (whether in TCC or WDDM mode) will use the basic Unified Memory model as on
    //  pre-6.x architectures even when they are running on hardware with compute capability 6.x or higher."
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
    pathCache = Memory::malloc<ToolPath>("path cache", pathCacheSize * sizeof(ToolPath), EMemoryType::CPU);

#if ENABLE_OPTIX
    optixState = Tracer::createOptixState();
    optixProgram = Tracer::createOptixProgram(optixState);
#endif
}

DAGTracer::~DAGTracer()
{
    if (!headLess) {
        colorsSurface.free();
        glDeleteTextures(1, &colorsImage);

        Memory::free(pathCache);
    }

    colorsBuffer.free();

    directLightingBuffers.free();
    pathTracingBuffers.free();

#if ENABLE_OPTIX
    Tracer::destroyOptixProgram(optixProgram);
    Tracer::destroyOptixAccelerationStructure(optixAccelerationStructure);
    Tracer::destroyOptixState(optixState);
#endif
}

void DAGTracer::initOptiX(const TransformDAG16& dag)
{
#if ENABLE_OPTIX
    optixAccelerationStructure = Tracer::createOptixAccelerationStructure(dag, optixState);
#endif
}

inline Tracer::TracePathsParams get_trace_params(
    const CameraView& camera, uint32 levels, const DAGInfo& dagInfo)
{
    const double3 position = make_double3(camera.position);
    const double3 direction = make_double3(camera.forward());
    const double3 up = make_double3(camera.up());
    const double3 right = make_double3(camera.right());

    const double3 boundsMin = make_double3(dagInfo.boundsAABBMin);
    const double3 boundsMax = make_double3(dagInfo.boundsAABBMax);

    const double fov = camera.fov / 2.0 * (double(M_PI) / 180.);
    const double aspect_ratio = double(imageWidth) / double(imageHeight);

    const double3 X = right * sin(fov) * aspect_ratio;
    const double3 Y = up * sin(fov);
    const double3 Z = direction * cos(fov);

    const double3 bottomLeft = position + Z - Y - X;
    const double3 bottomRight = position + Z - Y + X;
    const double3 topLeft = position + Z + Y - X;

    const double3 translation = -boundsMin;
    const double3 scale = make_double3(double(1 << levels)) / (boundsMax - boundsMin);

    const double3 finalPosition = (position + translation) * scale;
    const double3 finalBottomLeft = (bottomLeft + translation) * scale;
    const double3 finalTopLeft = (topLeft + translation) * scale;
    const double3 finalBottomRight = (bottomRight + translation) * scale;
    const double3 dx = (finalBottomRight - finalBottomLeft) * (1.0 / imageWidth);
    const double3 dy = (finalTopLeft - finalBottomLeft) * (1.0 / imageHeight);

    Tracer::TracePathsParams params;

    params.cameraPosition = finalPosition;
    params.rayMin = finalBottomLeft;
    params.rayDDx = dx;
    params.rayDDy = dy;

    return params;
}

inline Tracer::ColorParams get_color_params(const VoxelTextures& voxelTextures, EDebugColors debugColors, uint32 debugColorsIndexLevel)
{
    return Tracer::ColorParams {
        .materialTextures = voxelTextures.gpuMaterials,
        .debugColors = debugColors,
        .debugColorsIndexLevel = debugColorsIndexLevel
    };
}

size_t DAGTracer::current_size_in_bytes() const
{
    size_t out = 0;
    if (directLightingBuffers.is_valid())
        out += directLightingBuffers.size_in_bytes();
    if (pathTracingBuffers.is_valid())
        out += pathTracingBuffers.size_in_bytes();
    return out;
}

template <typename TDAG, typename TDAGColors>
void DAGTracer::resolve_direct_lighting(
    const CameraView& camera, const DirectLightingSettings& settings,
    const TDAG& dag, const DAGInfo& dagInfo, const ToolInfo& toolInfo,
    const TDAGColors& colors, const VoxelTextures& voxelTextures)
{
    PROFILE_FUNCTION();

    if (pathTracingBuffers.is_valid())
        pathTracingBuffers.free();
    if (directLightingBuffers.is_valid() && settings != previousDirectLightingSettings)
        directLightingBuffers.free();
    if (!directLightingBuffers.is_valid())
        directLightingBuffers = DirectLightingBuffers::allocate(settings);
    previousDirectLightingSettings = settings;

    const dim3 block_dim = dim3(8, 8);
    const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);

    if constexpr (std::is_same_v<TDAG, TransformDAG16>) {
        static const auto traversalConstants = dag.createTraversalConstants();
        cudaMemcpyToSymbolAsync(Tracer::s_transformTraversalConstants, &traversalConstants, sizeof(traversalConstants), 0, cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();
    }
    if constexpr (std::is_same_v<TDAG, SymmetryAwareDAG16>) {
        static const auto traversalConstants = dag.createTraversalConstants();
        cudaMemcpyToSymbolAsync(Tracer::s_symmetryTraversalConstants, &traversalConstants, sizeof(traversalConstants), 0, cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();
    }

    // Intersect primary rays.
    {
        const auto t = eventsManager->createTiming("paths");
        auto traceParams = get_trace_params(camera, dag.levels, dagInfo);
        traceParams.surfaceInteractionSurface = directLightingBuffers.surfaceInteractionBuffer;
        Tracer::trace_paths<<<grid_dim, block_dim>>>(traceParams, dag);
        CUDA_CHECK_ERROR();
    }

    // Compute surface color at primary ray intersection.
    {
        const auto t = eventsManager->createTiming("colors");
        const Tracer::TraceColorsParams traceParams {
            .colorParams = get_color_params(voxelTextures, settings.debugColors, settings.debugColorsIndexLevel),
            .surfaceInteractionSurface = directLightingBuffers.surfaceInteractionBuffer
        };
        Tracer::trace_colors<<<grid_dim, block_dim>>>(traceParams, dag, colors);
        CUDA_CHECK_ERROR();
    }

    // Intersect shadow rays.
    if (settings.enableShadows) {
        const auto t = eventsManager->createTiming("shadows");
        const Tracer::TraceShadowsParams traceParams {
            .shadowBias = settings.shadowBias,
            .surfaceInteractionSurface = directLightingBuffers.surfaceInteractionBuffer,
            .sunLightSurface = directLightingBuffers.sunLightBuffer
        };
        Tracer::trace_shadows<<<grid_dim, block_dim>>>(traceParams, dag);
        CUDA_CHECK_ERROR();
    }

    // Intersect ambient occlusion rays.
    if (settings.enableAmbientOcclusion) {
        const auto t = eventsManager->createTiming("ambient_occlusion");
        const Tracer::TraceAmbientOcclusionParams traceParams {
            .numSamples = settings.numAmbientOcclusionSamples,
            .shadowBias = settings.shadowBias,
            .aoRayLength = settings.ambientOcclusionRayLength,
            .randomSeed = randomSeed++,
            .surfaceInteractionSurface = directLightingBuffers.surfaceInteractionBuffer,
            .aoSurface = directLightingBuffers.ambientOcclusion.ambientOcclusionBuffer,
        };
        Tracer::trace_ambient_occlusion<<<grid_dim, block_dim>>>(traceParams, dag);
        CUDA_CHECK_ERROR();
    }

    // Blur ambient occlusion.
    if (settings.enableAmbientOcclusion) {
        const auto t = eventsManager->createTiming("ambient_occlusion_blur");
        directLightingBuffers.ambientOcclusion.ambientOcclusionBlurKernel.apply(
            directLightingBuffers.ambientOcclusion.ambientOcclusionBuffer,
            directLightingBuffers.ambientOcclusion.ambientOcclusionBlurScratchBuffer);
        CUDA_CHECK_ERROR();
    }

    // Compute lighting given the previously computed surface color, shadows & ambient occlusion.
    {
        const auto t = eventsManager->createTiming("lighting");
        const auto cameraParams = get_trace_params(camera, MAX_LEVELS, dagInfo);
        const Tracer::TraceLightingParams traceParams {
            .cameraPosition = cameraParams.cameraPosition,
            .rayMin = cameraParams.rayMin,
            .rayDDx = cameraParams.rayDDx,
            .rayDDy = cameraParams.rayDDy,
            .toolInfo = toolInfo,
            .sunBrightness = 1.0f,
            .applyShadows = settings.enableShadows,
            .applyAmbientOcclusion = settings.enableAmbientOcclusion,
            .fogDensity = settings.fogDensity,
            .aoSurface = directLightingBuffers.ambientOcclusion.ambientOcclusionBuffer,
            .sunLightSurface = directLightingBuffers.sunLightBuffer,
            .surfaceInteractionSurface = directLightingBuffers.surfaceInteractionBuffer,
            .finalColorsSurface = colorsBuffer
        };

        Tracer::trace_lighting<<<grid_dim, block_dim>>>(traceParams);
        CUDA_CHECK_ERROR();
    }
}

#if ENABLE_OPTIX
void DAGTracer::resolve_OptiX(const TransformDAG16& dag, const Tracer::TracePathTracingParams& traceParams)
{
    PROFILE_FUNCTION();

    static const auto traversalConstants = dag.createTraversalConstants();
    const Tracer::PathTracingOptixParams params
    {
        .traceParams = traceParams,
        .accelerationStructure = optixAccelerationStructure.accelerationStructure,
        .dag = dag,
#if !OPTIX_DEBUG_DRAW_SPHERES
        .pSubTrees = optixAccelerationStructure.subTrees.data(),
#endif
        .dagTraversalConstants = traversalConstants
    };

    void* pParams;
    cudaMallocAsync(&pParams, sizeof(params), 0);
    cudaMemcpyAsync(pParams, &params, sizeof(params), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    optixLaunch(optixProgram.pipeline, optixState.cuStream, (CUdeviceptr)pParams, sizeof(params),
        &optixProgram.shaderBindingTableDesc, imageWidth, imageHeight, 1);
    CUDA_CHECK_ERROR();

    cudaFreeAsync(pParams, 0);
    CUDA_CHECK_ERROR();
}
#endif

__global__ void kernel_linearToSRGB(std::span<const float3> inLinearColors, std::span<uint32_t> outColors)
{
    const auto g = cooperative_groups::this_grid();
    const auto threadRank = g.thread_rank();
    if (threadRank >= inLinearColors.size())
        return;

    outColors[threadRank] = ColorUtils::float3_to_rgb888(ColorUtils::accurateLinearToSRGB(inLinearColors[threadRank]));
}
static void copy_image_linear_to_srgb(const StaticArray2D<float3> linearImage, StaticArray2D<uint32_t> srgbImage)
{
    const dim3 block_dim = dim3(8, 8);
    const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);
    kernel_linearToSRGB<<<grid_dim, block_dim>>>(linearImage.cspan(), srgbImage.span());
    CUDA_CHECK_ERROR();
}

__global__ void kernel_copy_multiply(std::span<const float3> inColors, std::span<float3> outColors, float factor)
{
    const auto g = cooperative_groups::this_grid();
    const auto threadRank = g.thread_rank();
    if (threadRank >= inColors.size())
        return;

    outColors[threadRank] = inColors[threadRank] * factor;
}
[[maybe_unused]] static void copy_image_multiply(const StaticArray2D<float3> inImage, StaticArray2D<float3> outImage, float factor)
{
    const dim3 block_dim = dim3(8, 8);
    const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);
    kernel_copy_multiply<<<grid_dim, block_dim>>>(inImage.cspan(), outImage.span(), factor);
    CUDA_CHECK_ERROR();
}

template <typename TDAG, typename TDAGColors>
void DAGTracer::resolve_path_tracing(
    const CameraView& camera, const PathTracingSettings& settings,
    const TDAG& dag, const DAGInfo& dagInfo, const ToolInfo& toolInfo,
    const TDAGColors& colors, const VoxelTextures& voxelTextures)
{
    PROFILE_FUNCTION();

    if (directLightingBuffers.is_valid()) {
        directLightingBuffers.free();
        previousDagRootNode = 0xFFFF'FFFF; // Force clear when switching from direct lighting to path tracing.
    }
    if (pathTracingBuffers.is_valid() && settings != previousPathTracingSettings)
        pathTracingBuffers.free();
    if (!pathTracingBuffers.is_valid()) {
        pathTracingBuffers = PathTracingBuffers::allocate(settings
#if ENABLE_OPTIX
            ,
            optixState
#endif
        );
    }

    const auto cameraParams = get_trace_params(camera, MAX_LEVELS, dagInfo);
    Tracer::TracePathTracingParams traceParams;
    std::memset(&traceParams, 0, sizeof(traceParams)); // Prevent undefined behaviour by memcmp padding bytes.
    traceParams.cameraPosition = cameraParams.cameraPosition;
    traceParams.rayMin = cameraParams.rayMin;
    traceParams.rayDDx = cameraParams.rayDDx;
    traceParams.rayDDy = cameraParams.rayDDy;
    traceParams.integratePixel = settings.integratePixel;
    traceParams.environmentMap = settings.environmentMap;
    traceParams.environmentBrightness = settings.environmentBrightness;
    traceParams.environmentMapVisibleToPrimaryRays = settings.environmentMapVisibleToPrimaryRays;
    traceParams.randomSeed = randomSeed++;
    traceParams.numSamples = settings.samplesPerFrame;
    traceParams.maxPathDepth = settings.maxPathDepth;
    traceParams.colorParams = get_color_params(voxelTextures, settings.debugColors, settings.debugColorsIndexLevel);
    traceParams.toolInfo = toolInfo;
    // traceParams.surfaceInteractionSurface = surfaceInteractionBuffer;
    // traceParams.numAccumulatedSamples = numAccumulatedSamples + settings.samplesPerFrame;
    traceParams.accumulationBuffer = pathTracingBuffers.accumulationBuffer;
    traceParams.finalColorsSurface = colorsBuffer;

    // Check if any of the parameters changed, and if so, clear the accumulation buffer.
    if (memcmp(&camera, &previousCamera, sizeof(previousCamera)) != 0 || memcmp(&toolInfo, &previousToolInfo, sizeof(previousToolInfo)) != 0 || memcmp(&previousPathTracingSettings, &settings, sizeof(settings)) != 0 || dag.get_first_node_index() != previousDagRootNode) {
        cudaMemsetAsync(pathTracingBuffers.accumulationBuffer.data(), 0, pathTracingBuffers.accumulationBuffer.size_in_bytes(), nullptr);
        numAccumulatedSamples = 0;
    }
    traceParams.numAccumulatedSamples = numAccumulatedSamples + settings.samplesPerFrame;

    previousCamera = camera;
    previousToolInfo = toolInfo;
    previousPathTracingSettings = settings;
    previousDagRootNode = dag.get_first_node_index();

#if !USE_REPLAY
    // After 256 paths the image should be sharp enough.
    // Stop tracing to save compute resources and to prevent precision/overflow issues with the accumulation buffer.
    // We do run the kernel so we can update the tool overlay.
    if (numAccumulatedSamples > 256)
        traceParams.numSamples = 0;
#endif

    if constexpr (std::is_same_v<TDAG, TransformDAG16>) {
        static const auto traversalConstants = dag.createTraversalConstants();
        cudaMemcpyToSymbolAsync(Tracer::s_transformTraversalConstants, &traversalConstants, sizeof(traversalConstants), 0, cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();
    }
    if constexpr (std::is_same_v<TDAG, SymmetryAwareDAG16>) {
        static const auto traversalConstants = dag.createTraversalConstants();
        cudaMemcpyToSymbolAsync(Tracer::s_symmetryTraversalConstants, &traversalConstants, sizeof(traversalConstants), 0, cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();
    }

    const dim3 block_dim = dim3(8, 8);
    const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);

    auto t1 = eventsManager->createTiming("path_tracing");
    if (settings.implementation == EPathTracerImplementation::MegaKernel) {
        Tracer::trace_path_tracing<<<grid_dim, block_dim>>>(traceParams, dag, colors);
        CUDA_CHECK_ERROR();
        // Tracer::accumulate_to_final_buffer<<<grid_dim, block_dim>>>(traceParams);
        // CUDA_CHECK_ERROR();
    } else if (settings.implementation == EPathTracerImplementation::Wavefront) {
        if (traceParams.numSamples > 0) {
            auto& wavefrontBuffers = pathTracingBuffers.wavefront;
            uint32_t *pCounter0 = wavefrontBuffers.pWavefrontCounter0, *pCounter1 = wavefrontBuffers.pWavefrontCounter1;
            auto pathStateBuffer0 = wavefrontBuffers.pathStateBuffer0, pathStateBuffer1 = wavefrontBuffers.pathStateBuffer1;
            {
                const uint32_t size = (uint32_t)pathStateBuffer0.size();
                cudaMemcpyAsync(pCounter0, &size, sizeof(size), cudaMemcpyHostToDevice);
                CUDA_CHECK_ERROR();
            }
            Tracer::generate_rays_path_tracing<<<grid_dim, block_dim>>>(
                traceParams, pathStateBuffer0.span(), wavefrontBuffers.continuationRayBuffer);
            CUDA_CHECK_ERROR();
            for (uint32_t pathDepth = 0; pathDepth < traceParams.maxPathDepth; ++pathDepth) {
                Tracer::intersect_rays<<<grid_dim, block_dim>>>(dag, pCounter0, wavefrontBuffers.continuationRayBuffer, wavefrontBuffers.rayHitBuffer);
                CUDA_CHECK_ERROR();
                cudaMemsetAsync(pCounter1, 0, sizeof(uint32_t));
                CUDA_CHECK_ERROR();
                Tracer::shade_and_integrate_path_tracing<<<grid_dim, block_dim>>>(
                    traceParams,
                    pCounter0, pathStateBuffer0, wavefrontBuffers.rayHitBuffer,
                    pCounter1, pathStateBuffer1, wavefrontBuffers.continuationRayBuffer);
                CUDA_CHECK_ERROR();
                std::swap(pCounter0, pCounter1);
                std::swap(pathStateBuffer0, pathStateBuffer1);
            }
        }

        Tracer::accumulate_to_final_buffer<<<grid_dim, block_dim>>>(traceParams);
        CUDA_CHECK_ERROR();
    }
#if ENABLE_OPTIX
    else if (settings.implementation == EPathTracerImplementation::Optix) {
        if constexpr (std::is_same_v<TDAG, TransformDAG16>)
            this->resolve_OptiX(dag, traceParams);
    }
#endif
    // Add the number of samples that we have rendered.
    numAccumulatedSamples += traceParams.numSamples;
    t1.finish();

#if ENABLE_OPTIX
    if (settings.enableDenoising) {
        const auto t2 = eventsManager->createTiming("denoising");
        copy_image_multiply(pathTracingBuffers.accumulationBuffer, pathTracingBuffers.optix.noisyColorsBuffer, 1.0f / numAccumulatedSamples);
        applyOptixDenoising(optixState, pathTracingBuffers.optix.optixDenoiser, pathTracingBuffers.optix.noisyColorsBuffer, pathTracingBuffers.optix.denoisedColorsBuffer);
        copy_image_linear_to_srgb(pathTracingBuffers.optix.denoisedColorsBuffer, colorsBuffer);
    }
#endif
}

template <typename TDAG>
void DAGTracer::intersect_rays(const TDAG& inDag, std::span<Tracer::Ray> inOutrays, std::span<Tracer::RayHit> outHits) const
{
    const uint32_t numRays = (uint32_t)inOutrays.size();
    uint32_t* pNumRaysGPU;
    CUDA_CHECKED_CALL cudaMallocAsync(&pNumRaysGPU, sizeof(uint32_t), nullptr);
    CUDA_CHECKED_CALL cudaMemcpyAsync(pNumRaysGPU, &numRays, sizeof(numRays), cudaMemcpyHostToDevice, nullptr);

    Tracer::Ray* pRaysGPU;
    CUDA_CHECKED_CALL cudaMallocAsync(&pRaysGPU, inOutrays.size_bytes(), nullptr);
    CUDA_CHECKED_CALL cudaMemcpyAsync(pRaysGPU, inOutrays.data(), inOutrays.size_bytes(), cudaMemcpyHostToDevice, nullptr);
    Tracer::RayHit* pRayHitsGPU;
    CUDA_CHECKED_CALL cudaMallocAsync(&pRayHitsGPU, outHits.size_bytes(), nullptr);

    const dim3 block_dim = dim3(64, 1, 1);
    const dim3 grid_dim = dim3((uint32_t)inOutrays.size() / block_dim.x + 1, 1, 1);
    Tracer::intersect_rays<<<grid_dim, block_dim, 0, nullptr>>>(inDag, pNumRaysGPU, std::span(pRaysGPU, inOutrays.size()), std::span(pRayHitsGPU, outHits.size()));
    CUDA_CHECK_ERROR();

    cudaMemcpyAsync(inOutrays.data(), pRaysGPU, inOutrays.size_bytes(), cudaMemcpyDeviceToHost, nullptr);
    cudaMemcpyAsync(outHits.data(), pRayHitsGPU, outHits.size_bytes(), cudaMemcpyDeviceToHost, nullptr);
    CUDA_CHECKED_CALL cudaFreeAsync(pRaysGPU, nullptr);
    CUDA_CHECKED_CALL cudaFreeAsync(pNumRaysGPU, nullptr);
    cudaStreamSynchronize(nullptr);
}

template <typename TDAG>
ToolPath DAGTracer::get_path_async(const CameraView& camera, const TDAG& dag, const DAGInfo& dagInfo, const uint2& pixel)
{
    PROFILE_FUNCTION();
    cudaStream_t stream = nullptr;

    if (headLess)
        return {};

    check(pixel.x < imageWidth);
    check(pixel.y < imageHeight);

    // Enqueue the read_path kernel which reads the 3D position of the voxel at pixel x/y
    CUDA_CHECK_ERROR();
    ++currentPathIndex;

    // cudaMemcpyAsync(&pathCache[currentPathIndex % pathCacheSize], pathsBuffer.getPixelPointer(posX, posY), sizeof(uint3), cudaMemcpyDeviceToHost);
    ToolPath* pTmp;
    cudaMallocAsync(&pTmp, sizeof(ToolPath), stream);
    CUDA_CHECK_ERROR();

    const auto cameraParams = get_trace_params(camera, dag.levels, dagInfo);
    const float2 cameraSample = make_float2(pixel);
    Tracer::Ray ray {
        .origin = make_float3(cameraParams.cameraPosition),
        .direction = make_float3(normalize(cameraParams.rayMin + cameraSample.x * cameraParams.rayDDx + (imageHeight - 1 - cameraSample.y) * cameraParams.rayDDy - cameraParams.cameraPosition)),
        .tmin = 0.0f,
        .tmax = std::numeric_limits<float>::max()
    };

    Tracer::trace_tool_path<<<1, dim3(8, 8)>>>(dag, ray, pTmp);
    CUDA_CHECK_ERROR();

    cudaMemcpyAsync(&pathCache[currentPathIndex % pathCacheSize], pTmp, sizeof(ToolPath), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(pTmp, stream);
    CUDA_CHECK_ERROR();

    eventsManager->insertFenceValue("get_path", currentPathIndex);
    CUDA_CHECK_ERROR();

    // The GPU may lag behind the CPU so take the position read for the last completed frame.
    const auto lastCompletedPathIndex = eventsManager->getLastCompletedFenceValue("get_path");
    return pathCache[lastCompletedPathIndex % pathCacheSize];
}

template <typename TDAG>
__global__ void getVoxelValues_kernel(const TDAG dag, std::span<const uint3> locations, std::span<uint32_t> outVoxelMaterials)
{
    const unsigned globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= locations.size())
        return;

    Path path;
    path.path = locations[globalThreadIdx];
    auto optValue = DAGUtils::get_value(dag, path);
    outVoxelMaterials[globalThreadIdx] = optValue.value_or((uint32_t)-1);
}

// template <typename TDAG>
// void DAGTracer::get_voxel_values(const TDAG& dag, std::span<const uint3> locations, std::span<uint32_t> outVoxelMaterials) const
//{
//     check(locations.size() == outVoxelMaterials.size());
//     cudaStream_t stream = nullptr;
//     auto locationsGPU = cudaMallocAsyncRange<uint3>(locations.size(), stream);
//     cudaMemcpyAsync(locationsGPU.data(), locations.data(), locations.size_bytes(), cudaMemcpyHostToDevice, stream);
//     auto voxelMaterialsGPU = cudaMallocAsyncRange<uint32_t>(outVoxelMaterials.size(), stream);
//
//     getVoxelValues_kernel<TDAG><<<computeNumWorkGroups(locations.size()), workGroupSize, 0, stream>>>(dag, locationsGPU, voxelMaterialsGPU);
//     CUDA_CHECK_ERROR();
//
//     cudaMemcpyAsync(outVoxelMaterials.data(), voxelMaterialsGPU.data(), outVoxelMaterials.size_bytes(), cudaMemcpyDeviceToHost, stream);
//     cudaFreeAsync(locationsGPU.data(), stream);
//     cudaFreeAsync(voxelMaterialsGPU.data(), stream);
//     cudaStreamSynchronize(stream);
// }

uint32_t DAGTracer::get_path_tracing_num_accumulated_samples() const
{
    return numAccumulatedSamples;
}

DAGTracer::AmbientOcclusionBuffers DAGTracer::AmbientOcclusionBuffers::allocate()
{
    return AmbientOcclusionBuffers {
        .ambientOcclusionBuffer = StaticArray2D<float>::allocate("ambient occlusion buffer", imageWidth, imageHeight),
        .ambientOcclusionBlurScratchBuffer = StaticArray2D<float>::allocate("ambient occlusion buffer", imageWidth, imageHeight),
        .ambientOcclusionBlurKernel = BlurKernel::allocate(4),
    };
}

void DAGTracer::AmbientOcclusionBuffers::free()
{
    ambientOcclusionBuffer.free();
    ambientOcclusionBlurScratchBuffer.free();
    ambientOcclusionBlurKernel.free();
}

bool DAGTracer::AmbientOcclusionBuffers::is_valid() const
{
    return ambientOcclusionBuffer.is_valid();
}

size_t DAGTracer::AmbientOcclusionBuffers::size_in_bytes() const
{
    return ambientOcclusionBuffer.size_in_bytes() + ambientOcclusionBlurScratchBuffer.size_in_bytes() + ambientOcclusionBlurKernel.size_in_bytes();
}

DAGTracer::DirectLightingBuffers DAGTracer::DirectLightingBuffers::allocate(const DirectLightingSettings& settings)
{
    DirectLightingBuffers out {
        .surfaceInteractionBuffer = StaticArray2D<Tracer::SurfaceInteraction>::allocate("SurfaceInteraction buffer", imageWidth, imageHeight)
    };
    if (settings.enableShadows)
        out.sunLightBuffer = StaticArray2D<float>::allocate("sun light buffer", imageWidth, imageHeight);
    if (settings.enableAmbientOcclusion)
        out.ambientOcclusion = AmbientOcclusionBuffers::allocate();
    return out;
}

void DAGTracer::DirectLightingBuffers::free()
{
    surfaceInteractionBuffer.free();
    if (sunLightBuffer.is_valid())
        sunLightBuffer.free();
    if (ambientOcclusion.is_valid())
        ambientOcclusion.free();
}

bool DAGTracer::DirectLightingBuffers::is_valid() const
{
    return surfaceInteractionBuffer.is_valid();
}

size_t DAGTracer::DirectLightingBuffers::size_in_bytes() const
{
    size_t out = surfaceInteractionBuffer.size_in_bytes();
    if (sunLightBuffer.is_valid())
        out += sunLightBuffer.size_in_bytes();
    if (ambientOcclusion.is_valid())
        out += ambientOcclusion.size_in_bytes();
    return out;
}

DAGTracer::WavefrontPathTracingBuffers DAGTracer::WavefrontPathTracingBuffers::allocate()
{
    return WavefrontPathTracingBuffers {
        .pWavefrontCounter0 = Memory::malloc<uint32_t>("pWavefrontCounter0", sizeof(uint32_t), 128, EMemoryType::GPU_Malloc),
        .pWavefrontCounter1 = Memory::malloc<uint32_t>("pWavefrontCounter1", sizeof(uint32_t), 128, EMemoryType::GPU_Malloc),
        .pathStateBuffer0 = StaticArray<Tracer::WavefrontPathState>::allocate("wavefrontPathStateBuffer0", imageWidth * imageHeight, EMemoryType::GPU_Malloc),
        .pathStateBuffer1 = StaticArray<Tracer::WavefrontPathState>::allocate("wavefrontPathStateBuffer1", imageWidth * imageHeight, EMemoryType::GPU_Malloc),
        .continuationRayBuffer = StaticArray<Tracer::Ray>::allocate("wavefrontContinuationRayBuffer", imageWidth * imageHeight, EMemoryType::GPU_Malloc),
        .rayHitBuffer = StaticArray<Tracer::RayHit>::allocate("wavefrontRayHitBuffer", imageWidth * imageHeight, EMemoryType::GPU_Malloc),
    };
}

void DAGTracer::WavefrontPathTracingBuffers::free()
{
    Memory::free(pWavefrontCounter0);
    Memory::free(pWavefrontCounter1);
    pathStateBuffer0.free();
    pathStateBuffer1.free();
    continuationRayBuffer.free();
    rayHitBuffer.free();
}

bool DAGTracer::WavefrontPathTracingBuffers::is_valid() const
{
    return rayHitBuffer.is_valid();
}

size_t DAGTracer::WavefrontPathTracingBuffers::size_in_bytes() const
{
    return pathStateBuffer0.size_in_bytes() + pathStateBuffer1.size_in_bytes() + continuationRayBuffer.size_in_bytes() + rayHitBuffer.size_in_bytes();
}

#if ENABLE_OPTIX
DAGTracer::OptixDenoisingBuffers DAGTracer::OptixDenoisingBuffers::allocate(Tracer::OptixState_ optixState)
{
    return OptixDenoisingBuffers {
        .noisyColorsBuffer = StaticArray2D<float3>::allocate("noisy colors buffer", imageWidth, imageHeight),
        .denoisedColorsBuffer = StaticArray2D<float3>::allocate("denoised colors buffer", imageWidth, imageHeight),
        .optixDenoiser = Tracer::createOptixDenoiser(optixState)
    };
}

void DAGTracer::OptixDenoisingBuffers::free()
{
    noisyColorsBuffer.free();
    denoisedColorsBuffer.free();
    Tracer::destroyOptixDenoiser(optixDenoiser);
}

bool DAGTracer::OptixDenoisingBuffers::is_valid() const
{
    return noisyColorsBuffer.is_valid();
}

size_t DAGTracer::OptixDenoisingBuffers::size_in_bytes() const
{
    return noisyColorsBuffer.size_in_bytes() + denoisedColorsBuffer.size_in_bytes();
}
#endif

DAGTracer::PathTracingBuffers DAGTracer::PathTracingBuffers::allocate(const PathTracingSettings& settings
#if ENABLE_OPTIX
    ,
    Tracer::OptixState_ optixState
#endif
)
{
    PathTracingBuffers out {
        .accumulationBuffer = StaticArray2D<float3>::allocate("path tracing accumulation buffer", imageWidth, imageHeight)
    };
    if (settings.implementation == EPathTracerImplementation::Wavefront)
        out.wavefront = WavefrontPathTracingBuffers::allocate();
#if ENABLE_OPTIX
    if (settings.enableDenoising)
        out.optix = OptixDenoisingBuffers::allocate(optixState);
#endif
    return out;
}

void DAGTracer::PathTracingBuffers::free()
{
    accumulationBuffer.free();
    if (wavefront.is_valid())
        wavefront.free();
#if ENABLE_OPTIX
    if (optix.is_valid())
        optix.free();
#endif
}

bool DAGTracer::PathTracingBuffers::is_valid() const
{
    return accumulationBuffer.is_valid();
}

size_t DAGTracer::PathTracingBuffers::size_in_bytes() const
{
    size_t out = accumulationBuffer.size_in_bytes();
    if (wavefront.is_valid())
        out += wavefront.size_in_bytes();
#if ENABLE_OPTIX
    if (optix.is_valid())
        out += optix.size_in_bytes();
#endif
    return out;
}

void PathTracingSettings::free()
{
    environmentMap.free();
}

#define DAG_IMPLS(Dag)                                                                                        \
    template ToolPath DAGTracer::get_path_async(const CameraView&, const Dag&, const DAGInfo&, const uint2&); \
    template void DAGTracer::intersect_rays(const Dag&, std::span<Tracer::Ray>, std::span<Tracer::RayHit>) const;
//template void DAGTracer::get_voxel_values<Dag>(const Dag&, std::span<const uint3>, std::span<uint32_t>) const; \

#define DAG_COLOR_IMPLS(Dag, Colors)                                                                                                                                                                   \
    template void DAGTracer::resolve_direct_lighting<Dag, Colors>(const CameraView&, const DirectLightingSettings&, const Dag&, const DAGInfo&, const ToolInfo&, const Colors&, const VoxelTextures&); \
    template void DAGTracer::resolve_path_tracing<Dag, Colors>(const CameraView&, const PathTracingSettings&, const Dag&, const DAGInfo&, const ToolInfo&, const Colors&, const VoxelTextures&);

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

