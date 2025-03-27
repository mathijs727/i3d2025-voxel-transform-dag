#pragma once
#include "array2d.h"
#include "blur.h"
#include "camera_view.h"
#include "cuda_gl_buffer.h"
#include "dag_info.h"
#include "image.h"
#include "opengl_interop.h"
#include "tracer.h"
#include "typedefs.h"
#include <chrono>
#include <optional>
#include <span>

class EventsManager;
struct VoxelTextures;

enum class EPathTracerImplementation {
    MegaKernel,
    Wavefront,
#if ENABLE_OPTIX
    Optix,
#endif
};

struct DirectLightingSettings {
    bool enableShadows = true;
    float shadowBias = 1.0f;

    bool enableAmbientOcclusion = true;
    int numAmbientOcclusionSamples = 3;
    float ambientOcclusionRayLength = 1 << (SCENE_DEPTH - 10);

    float fogDensity = 0.0f;

    EDebugColors debugColors = EDebugColors::None;
    uint32 debugColorsIndexLevel = 0;

    constexpr bool operator==(const DirectLightingSettings&) const noexcept = default;
};

struct PathTracingSettings {
    EPathTracerImplementation implementation = EPathTracerImplementation::MegaKernel;

    // bool optixDenosing = false;
    bool integratePixel = true;
    int samplesPerFrame = 1;
#ifdef DEFAULT_PATH_TRACE_DEPTH
    int maxPathDepth = DEFAULT_PATH_TRACE_DEPTH;
#else
    int maxPathDepth = 4;
#endif
    bool enableDenoising = false;

    CUDATexture environmentMap;
    float environmentBrightness = 1.0f;
    bool environmentMapVisibleToPrimaryRays = true;

    EDebugColors debugColors = EDebugColors::White;
    uint32 debugColorsIndexLevel = 0;

    void free();
    constexpr bool operator==(const PathTracingSettings&) const noexcept = default;
};

class DAGTracer {
public:
    const bool headLess;

    DAGTracer(bool headLess, EventsManager* eventsManager);
    ~DAGTracer();

    void initOptiX(const TransformDAG16& dag);

    inline GLuint get_colors_image() const { return colorsImage; }
    inline void update_colors_image() { colorsSurface.copyFrom(colorsBuffer); }
    size_t current_size_in_bytes() const;

    template <typename TDAG, typename TDAGColors>
    void resolve_direct_lighting(
        const CameraView& camera, const DirectLightingSettings& lightingSettings,
        const TDAG& dag, const DAGInfo& dagInfo, const ToolInfo& toolInfo,
        const TDAGColors& colors, const VoxelTextures& voxelTextures);

    template <typename TDAG, typename TDAGColors>
    void resolve_path_tracing(
        const CameraView& camera, const PathTracingSettings& settings,
        const TDAG& dag, const DAGInfo& dagInfo, const ToolInfo& toolInfo,
        const TDAGColors& colors, const VoxelTextures& voxelTextures);

    template <typename TDAG>
    void intersect_rays(const TDAG& inDag, std::span<Tracer::Ray> inOutrays, std::span<Tracer::RayHit> outHits) const;

    template <typename TDAG>
    ToolPath get_path_async(const CameraView& camera, const TDAG& dag, const DAGInfo& dagInfo, const uint2& pixelPos); // May be delayed by a couple frames...

    // template <typename TDAG>
    // void get_voxel_values(const TDAG& dag, std::span<const uint3> locations, std::span<uint32_t> outVoxelMaterials) const;

    uint32_t get_path_tracing_num_accumulated_samples() const;

private:
#if ENABLE_OPTIX
    void resolve_OptiX(const TransformDAG16& dag, const Tracer::TracePathTracingParams& traceParams);
#endif

public:
    struct AmbientOcclusionBuffers {
    public:
        StaticArray2D<float> ambientOcclusionBuffer;
        StaticArray2D<float> ambientOcclusionBlurScratchBuffer;
        BlurKernel ambientOcclusionBlurKernel;

    public:
        static AmbientOcclusionBuffers allocate();
        void free();

        bool is_valid() const;
        size_t size_in_bytes() const;
    };
    struct DirectLightingBuffers {
    public:
        StaticArray2D<Tracer::SurfaceInteraction> surfaceInteractionBuffer;
        StaticArray2D<float> sunLightBuffer;
        AmbientOcclusionBuffers ambientOcclusion;

    public:
        static DirectLightingBuffers allocate(const DirectLightingSettings&);
        void free();

        bool is_valid() const;
        size_t size_in_bytes() const;
    };

    struct WavefrontPathTracingBuffers {
    public:
        uint32_t *pWavefrontCounter0, *pWavefrontCounter1;
        StaticArray<Tracer::WavefrontPathState> pathStateBuffer0, pathStateBuffer1;
        StaticArray<Tracer::Ray> continuationRayBuffer;
        StaticArray<Tracer::RayHit> rayHitBuffer;

    public:
        static WavefrontPathTracingBuffers allocate();
        void free();

        bool is_valid() const;
        size_t size_in_bytes() const;
    };
#if ENABLE_OPTIX
    struct OptixDenoisingBuffers {
    public:
        StaticArray2D<float3> noisyColorsBuffer;
        StaticArray2D<float3> denoisedColorsBuffer;
        Tracer::OptixDenoiser_ optixDenoiser;

    public:
        static OptixDenoisingBuffers allocate(Tracer::OptixState_);
        void free();

        bool is_valid() const;
        size_t size_in_bytes() const;
    };
#endif
    struct PathTracingBuffers {
    public:
        WavefrontPathTracingBuffers wavefront;
        // Accumulate samples of multiple frames if none of the settings changed.
        StaticArray2D<float3> accumulationBuffer;

#if ENABLE_OPTIX
        OptixDenoisingBuffers optix;
#endif

    public:
        static PathTracingBuffers allocate(const PathTracingSettings&
#if ENABLE_OPTIX
            ,
            Tracer::OptixState_ optixState
#endif
        );
        void free();

        bool is_valid() const;
        size_t size_in_bytes() const;
    };

private:
    // OpenGL output image.
    GLuint colorsImage = 0;
    GLSurface2D colorsSurface;
    // CUDA output image.
    StaticArray2D<uint32_t> colorsBuffer;

    // Measuring GPU timings.
    EventsManager* eventsManager;

    // Random seed used for ambient occlusion & path tracing.
    uint32_t randomSeed = 123;

    // Copy 3D position of the mouse cursor.
    static constexpr size_t pathCacheSize = 16;
    size_t currentPathIndex = 0;
    ToolPath* pathCache;

    // Settings used to (path trace) render the previous frame.
    // Accumulate samples over multiple frames if none of these settings change.
    CameraView previousCamera;
    ToolInfo previousToolInfo;
    DirectLightingSettings previousDirectLightingSettings;
    PathTracingSettings previousPathTracingSettings;
    uint32_t previousDagRootNode = (uint32_t)-1;

    // Buffers
    uint32_t numAccumulatedSamples = 0;
    DirectLightingBuffers directLightingBuffers;
    PathTracingBuffers pathTracingBuffers;

    // Optix path tracer using custom primitves to traverse SVDAG.
#if ENABLE_OPTIX
    Tracer::OptixState_ optixState;
    Tracer::OptixAccelerationStructure_ optixAccelerationStructure;
    Tracer::OptixProgram_ optixProgram;
#endif
};
