#include "dags/basic_dag/basic_dag.h"
#include "dags/dag_utils.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/hash_dag/hash_dag_colors.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
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
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#endif

namespace Tracer {

#if ENABLE_OPTIX
void optixLogCallback(unsigned int level, const char* tag, const char* message, void* cbData)
{
    if (level == 1 || level == 2)
        spdlog::error("[optix] {}: {}", tag, message);
    else if (level == 3)
        spdlog::warn("[optix] {}: {}", tag, message);
    else
        spdlog::info("[optix] {}: {}", tag, message);
}

OptixState_ createOptixState()
{
    OptixState_ out {};
    // out.cuContext = 0;
    out.cuStream = 0;
    cuCtxGetCurrent(&out.cuContext);
    optixInit();

    OptixDeviceContextOptions options {};
#if ENABLE_CHECKS
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    CUDA_CHECKED_CALL optixDeviceContextCreate(out.cuContext, &options, &out.deviceContext);
    CUDA_CHECKED_CALL optixDeviceContextSetLogCallback(out.deviceContext, optixLogCallback, nullptr, 4);
    return out;
}

void destroyOptixState(const OptixState_& optixState)
{
    CUDA_CHECKED_CALL optixDeviceContextDestroy(optixState.deviceContext);
}

OptixAccelerationStructure_ createOptixAccelerationStructure(const TransformDAG16& dag, const OptixState_& optixState)
{
    const float radius = float(1u << OptixLevel) / 2.0f;

    const auto traversalConstants = dag.createTraversalConstants();
    auto dagCPU = dag.copy(EMemoryType::CPU);

#if OPTIX_DEBUG_DRAW_SPHERES
    std::vector<float3> sphereCenters;
#else
    std::vector<TransformDAG16OptixParams> subTrees;
    std::vector<OptixAabb> subTreeAABBs;
#endif

    static_assert(OptixLevel > TRANSFORM_DAG_MAX_TRANSLATION_LEVEL);

    struct StackEntry {
        Path path;
        uint32_t level;
        uint32_t nodeIdx;
        uint32_t transformID;
    };
    std::stack<StackEntry> stack;
    stack.push(StackEntry { .path = Path(0, 0, 0), .level = dagCPU.levels, .nodeIdx = dagCPU.get_first_node_index(), .transformID = 0 });
    while (!stack.empty()) {
        const auto stackEntry = stack.top();
        stack.pop();

        if (stackEntry.level == OptixLevel) {
            Path path = stackEntry.path;
#if OPTIX_DEBUG_DRAW_SPHERES
            sphereCenters.push_back(path.as_position(stackEntry.level) + radius);
#else
            subTrees.push_back({ .path = path, .nodeIdx = stackEntry.nodeIdx, .transformID = stackEntry.transformID });
            const float3 lower = path.as_position(stackEntry.level);
            subTreeAABBs.push_back({
                .minX = lower.x,
                .minY = lower.y,
                .minZ = lower.z,
                .maxX = lower.x + 2 * radius,
                .maxY = lower.y + 2 * radius,
                .maxZ = lower.z + 2 * radius,
            });
#endif
        } else {
            const auto* pNode = &dagCPU.nodes[dagCPU.levelStarts[stackEntry.level] + stackEntry.nodeIdx];
            const uint32_t childLevel = stackEntry.level - 1;
#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
            const uint32_t childMask = traversalConstants.transformMaskMappingLocalToWorld[stackEntry.transformID][TransformDAG16::convert_child_mask(*pNode)];
#else
            const uint32_t childMask = TransformDAG16::convert_child_mask(*pNode);
#endif
            for (uint8_t childIdx = 0; childIdx < 8; ++childIdx) {
                if (!((childMask >> childIdx) & 0b1))
                    continue;

#if TRANSFORM_DAG_USE_TRANSFORMATION_ID
                const uint32_t localChildIdx = traversalConstants.transformChildMappingWorldToLocal[stackEntry.transformID][childIdx];
                const TransformDAG16::TransformPointer& childPointer = dagCPU.get_child_index(childLevel, pNode, *pNode, (uint8_t)localChildIdx);
                const uint32_t childTransformID = traversalConstants.transformCombineTable[stackEntry.transformID][childPointer.transformID];
#else
                const TransformDAG16::TransformPointer& childPointer = dagCPU.get_child_index(childLevel, pNode, *pNode, childIdx);
                const uint32_t childTransformID = 0;
#endif
                // checkAlways(childPointer.transformID == 0);

                Path childPath = stackEntry.path;
                childPath.descendZYX(childIdx);
                stack.push({ .path = childPath, .level = childLevel, .nodeIdx = childPointer.index, .transformID = childTransformID });
            }
        }
    }

    // Build the Optix acceleration structure.
    OptixAccelBuildOptions buildOptions {};
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    buildOptions.motionOptions.numKeys = 0;

    OptixAccelerationStructure_ out {};

#if OPTIX_DEBUG_DRAW_SPHERES
    const std::array sphereRadii { radius };
    out.sphereCenters = StaticArray<float3>::allocate("Optix [sphere centers]", sphereCenters, EMemoryType::GPU_Malloc);
    out.sphereRadii = StaticArray<float>::allocate("Optix [sphere radii]", sphereRadii, EMemoryType::GPU_Malloc);
    CUdeviceptr pSphereCenters = (CUdeviceptr)out.sphereCenters.data();
    CUdeviceptr pSphereRadii = (CUdeviceptr)out.sphereRadii.data();
    unsigned int geometryFlags = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput buildInput {
        .type = OPTIX_BUILD_INPUT_TYPE_SPHERES,
        .sphereArray = OptixBuildInputSphereArray {
            .vertexBuffers = &pSphereCenters,
            .vertexStrideInBytes = sizeof(float3),
            .numVertices = (unsigned)sphereCenters.size(),
            .radiusBuffers = &pSphereRadii,
            .radiusStrideInBytes = sizeof(float),
            .singleRadius = true,
            .flags = &geometryFlags,
            .numSbtRecords = 1,
            .sbtIndexOffsetBuffer = (CUdeviceptr) nullptr,
            .sbtIndexOffsetSizeInBytes = 0,
            .sbtIndexOffsetStrideInBytes = 0,
            .primitiveIndexOffset = 0,
        }
    };
#else
    out.subTrees = StaticArray<TransformDAG16OptixParams>::allocate("Optix [subtrees]", subTrees, EMemoryType::GPU_Malloc);
    checkAlways(subTreeAABBs.size() == subTrees.size());
    auto subTreeAABBs_gpu = Memory::malloc<OptixAabb>("Optix [subtree AABBs]", subTreeAABBs.size() * sizeof(OptixAabb), OPTIX_AABB_BUFFER_BYTE_ALIGNMENT, EMemoryType::GPU_Malloc);
    cudaMemcpy(subTreeAABBs_gpu, subTreeAABBs.data(), subTreeAABBs.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice);
    std::array<CUdeviceptr, 1> aabbBuffers { (CUdeviceptr)subTreeAABBs_gpu };
    unsigned int geometryFlags = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput buildInput {
        .type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
        .customPrimitiveArray = OptixBuildInputCustomPrimitiveArray {
            .aabbBuffers = aabbBuffers.data(),
            .numPrimitives = (unsigned)subTrees.size(),
            .strideInBytes = 0,
            .flags = &geometryFlags,
            .numSbtRecords = 1,
            .sbtIndexOffsetBuffer = (CUdeviceptr) nullptr,
            .sbtIndexOffsetSizeInBytes = 0,
            .sbtIndexOffsetStrideInBytes = 0,
            .primitiveIndexOffset = 0,
        }
    };
#endif

    OptixAccelBufferSizes bufferSizes {};
    CUDA_CHECKED_CALL optixAccelComputeMemoryUsage(optixState.deviceContext, &buildOptions, &buildInput, 1, &bufferSizes);

    void* pScratchMemory = Memory::malloc<std::byte>("Optix [scratch acceleration structure]", bufferSizes.tempSizeInBytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT, EMemoryType::GPU_Malloc);
    out.pAccelerationStructureMemory = Memory::malloc<std::byte>("Optix [acceleration structure]", bufferSizes.outputSizeInBytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT, EMemoryType::GPU_Malloc);
    CUDA_CHECKED_CALL optixAccelBuild(optixState.deviceContext, optixState.cuStream, &buildOptions, &buildInput, 1, (CUdeviceptr)pScratchMemory, bufferSizes.tempSizeInBytes, (CUdeviceptr)out.pAccelerationStructureMemory, bufferSizes.outputSizeInBytes, &out.accelerationStructure, nullptr, 0);
    cudaDeviceSynchronize(); // Wait for Optix acceleration structure build to finish.
    Memory::free(pScratchMemory);
#if !OPTIX_DEBUG_DRAW_SPHERES
    Memory::free(subTreeAABBs_gpu);
#endif
    dagCPU.free();

    return out;
}

void destroyOptixAccelerationStructure(OptixAccelerationStructure_& accelerationStructure)
{
    if (!accelerationStructure.pAccelerationStructureMemory)
        return;

    Memory::free(accelerationStructure.pAccelerationStructureMemory);
    accelerationStructure.pAccelerationStructureMemory = nullptr;

#if OPTIX_DEBUG_DRAW_SPHERES
    accelerationStructure.sphereCenters.free();
    accelerationStructure.sphereRadii.free();
#else
    accelerationStructure.subTrees.free();
#endif
}

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SBTRecord {
    std::byte header[OPTIX_SBT_RECORD_HEADER_SIZE]; // Optix header.
    T data; // User data.
};

static size_t alignUp(size_t address, size_t alignment)
{
    while (address % alignment)
        ++address;
    return address;
}

class SBTBuilder {
public:
    template <typename T>
    void setRayGenRecord(OptixProgramGroup programGroup, const T& userData)
    {
        m_rayGenShader = createEntry(programGroup, userData);
    }
    void setRayGenRecord(OptixProgramGroup programGroup)
    {
        m_rayGenShader = createEntry(programGroup);
    }

    template <typename T>
    void setExceptionRecord(OptixProgramGroup programGroup, const T& userData)
    {
        m_exceptionShader = createEntry(programGroup, userData);
    }
    void setExceptionRecord(OptixProgramGroup programGroup)
    {
        m_exceptionShader = createEntry(programGroup);
    }

    template <typename T>
    void addMissRecord(OptixProgramGroup programGroup, const T& userData)
    {
        m_missShaders.push_back(createEntry(programGroup, userData));
    }
    void addMissRecord(OptixProgramGroup programGroup)
    {
        m_missShaders.push_back(createEntry(programGroup));
    }

    template <typename T>
    void addHitGroupRecord(OptixProgramGroup programGroup, const T& userData)
    {
        m_hitGroupShaders.push_back(createEntry(programGroup, userData));
    }
    void addHitGroupRecord(OptixProgramGroup programGroup)
    {
        m_hitGroupShaders.push_back(createEntry(programGroup));
    }

    template <typename T>
    void addCallableRecord(OptixProgramGroup programGroup, const T& userData)
    {
        m_callableShaders.push_back(createEntry(programGroup, userData));
    }
    void addCallableRecord(OptixProgramGroup programGroup)
    {
        m_callableShaders.push_back(createEntry(programGroup));
    }

    std::pair<OptixShaderBindingTable, std::byte*> finalize() const
    {
        std::vector<std::byte> shaderBindingTable;
        const auto addTableEntries = [&](std::span<const std::vector<std::byte>> sbtEntries, CUdeviceptr& outOffset, unsigned& outStride, unsigned& outCount) {
            if (sbtEntries.empty())
                return;
            outOffset = alignUp(shaderBindingTable.size(), OPTIX_SBT_RECORD_ALIGNMENT);
            shaderBindingTable.resize(outOffset);

            outStride = 0;
            for (const auto& sbtEntry : sbtEntries)
                outStride = std::max(outStride, (unsigned)sbtEntry.size());
            outStride = (unsigned)alignUp(outStride, OPTIX_SBT_RECORD_ALIGNMENT);
            outCount = (unsigned)sbtEntries.size();

            shaderBindingTable.resize(outOffset + outCount * outStride);
            for (size_t i = 0; i < sbtEntries.size(); ++i) {
                const auto& sbtEntry = sbtEntries[i];
                std::memcpy(&shaderBindingTable[outOffset + i * outStride], sbtEntry.data(), sbtEntry.size());
            }
        };
        const auto addTableEntry = [&](std::span<const std::byte> sbtEntry, CUdeviceptr& outOffset) {
            if (sbtEntry.empty())
                return;
            outOffset = alignUp(shaderBindingTable.size(), OPTIX_SBT_RECORD_ALIGNMENT);
            shaderBindingTable.resize(outOffset + alignUp(sbtEntry.size(), OPTIX_SBT_RECORD_ALIGNMENT));
            std::memcpy(shaderBindingTable.data() + outOffset, sbtEntry.data(), sbtEntry.size());
        };

        OptixShaderBindingTable shaderBindingTableDesc {};
        addTableEntry(m_rayGenShader, shaderBindingTableDesc.raygenRecord);
        addTableEntry(m_exceptionShader, shaderBindingTableDesc.exceptionRecord);
        addTableEntries(m_missShaders, shaderBindingTableDesc.missRecordBase, shaderBindingTableDesc.missRecordStrideInBytes, shaderBindingTableDesc.missRecordCount);
        addTableEntries(m_hitGroupShaders, shaderBindingTableDesc.hitgroupRecordBase, shaderBindingTableDesc.hitgroupRecordStrideInBytes, shaderBindingTableDesc.hitgroupRecordCount);
        addTableEntries(m_callableShaders, shaderBindingTableDesc.callablesRecordBase, shaderBindingTableDesc.callablesRecordStrideInBytes, shaderBindingTableDesc.callablesRecordCount);

        const auto pShaderBindingTable = Memory::malloc<std::byte>("Optix [ShaderBindingTable]", shaderBindingTable.size(), OPTIX_SBT_RECORD_ALIGNMENT, EMemoryType::GPU_Malloc);
        cudaMemcpy(pShaderBindingTable, shaderBindingTable.data(), shaderBindingTable.size(), cudaMemcpyHostToDevice);

        const auto& pointToGPU = [&](bool empty, CUdeviceptr& entry) {
            if (empty)
                entry = 0;
            else
                entry += (CUdeviceptr)pShaderBindingTable;
        };
        pointToGPU(m_rayGenShader.empty(), shaderBindingTableDesc.raygenRecord);
        pointToGPU(m_exceptionShader.empty(), shaderBindingTableDesc.exceptionRecord);
        pointToGPU(m_missShaders.empty(), shaderBindingTableDesc.missRecordBase);
        pointToGPU(m_hitGroupShaders.empty(), shaderBindingTableDesc.hitgroupRecordBase);
        pointToGPU(m_callableShaders.empty(), shaderBindingTableDesc.callablesRecordBase);

        return { shaderBindingTableDesc, pShaderBindingTable };
    }

private:
    template <typename T>
    std::vector<std::byte> createEntry(OptixProgramGroup programGroup, const T& userData)
    {
        std::vector<std::byte> outRecord(OPTIX_SBT_RECORD_HEADER_SIZE + sizeof(T));
        CUDA_CHECKED_CALL optixSbtRecordPackHeader(programGroup, outRecord.data());
        std::memcpy(outRecord.data() + OPTIX_SBT_RECORD_HEADER_SIZE, &userData, sizeof(T));
        return outRecord;
    }

    std::vector<std::byte> createEntry(OptixProgramGroup programGroup)
    {
        std::vector<std::byte> outRecord(OPTIX_SBT_RECORD_HEADER_SIZE);
        CUDA_CHECKED_CALL optixSbtRecordPackHeader(programGroup, outRecord.data());
        return outRecord;
    }

private:
    std::vector<std::byte> m_rayGenShader;
    std::vector<std::byte> m_exceptionShader;
    std::vector<std::vector<std::byte>> m_missShaders;
    std::vector<std::vector<std::byte>> m_hitGroupShaders;
    std::vector<std::vector<std::byte>> m_callableShaders;
};

OptixProgram_ createOptixProgram(const OptixState_& optixState)
{
    constexpr uint32_t maxTraceDepth = 5; // Max optixTrace recursion depth.
    [[maybe_unused]] constexpr uint32_t maxTraversalDepth = 1; // Single GAS (no instancing).
    constexpr uint32_t maxContinuationCallDepth = 1; // Maximum depth of call trees of continuation callables.
    constexpr uint32_t maxDirectCallDepth = 1; // Maximum depth of call trees of direct callables.

    OptixModuleCompileOptions moduleCompileOptions {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    printf("COMPILE_OPTIX_DEBUG = %u\n", (uint32_t)COMPILE_OPTIX_DEBUG);
#if COMPILE_OPTIX_DEBUG
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
    moduleCompileOptions.numBoundValues = 0;
    moduleCompileOptions.numPayloadTypes = 0;
    moduleCompileOptions.payloadTypes = nullptr;

    OptixPipelineCompileOptions pipelineCompileOptions {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 8;
    pipelineCompileOptions.numAttributeValues = 3;
#if COMPILE_OPTIX_DEBUG
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_USER;
#else
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
#if OPTIX_DEBUG_DRAW_SPHERES
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
#else
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
#endif

    OptixPipelineLinkOptions pipelineLinkOptions {};
    pipelineLinkOptions.maxTraceDepth = maxTraceDepth;

    const std::filesystem::path filePath { OPTIX_PROGRAM_FILE };
    std::ifstream file { filePath, std::ios::binary };
    checkAlways(file.is_open());

    std::vector<char> shaderIR((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::array<char, 4096> logString;
    size_t logStringSize = logString.size();
    OptixModule optixModule;
    optixModuleCreate(optixState.deviceContext, &moduleCompileOptions, &pipelineCompileOptions, shaderIR.data(), shaderIR.size(), logString.data(), &logStringSize, &optixModule);
    if (logStringSize > 1)
        spdlog::info("{}", logString.data());

    OptixProgramGroupDesc pgExceptionDesc {};
    pgExceptionDesc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    pgExceptionDesc.exception.entryFunctionName = "__exception__all";
    pgExceptionDesc.exception.module = optixModule;

    OptixProgramGroupDesc pgRayGenDesc {};
    pgRayGenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgRayGenDesc.raygen.entryFunctionName = "__raygen__camera";
    pgRayGenDesc.raygen.module = optixModule;

    OptixProgramGroupDesc pgMissDesc {};
    pgMissDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgMissDesc.miss.entryFunctionName = "__miss__background";
    pgMissDesc.miss.module = optixModule;

    auto pgMissShadowDesc = pgMissDesc;
    pgMissShadowDesc.miss.entryFunctionName = "__miss__shadow";

    OptixProgramGroupDesc pgClosestHitPTDesc {};
    pgClosestHitPTDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgClosestHitPTDesc.hitgroup.moduleCH = optixModule;
    pgClosestHitPTDesc.hitgroup.entryFunctionNameCH = "__closesthit__pathTracing";
    pgClosestHitPTDesc.hitgroup.moduleAH = nullptr;
    pgClosestHitPTDesc.hitgroup.entryFunctionNameAH = nullptr;
#if OPTIX_DEBUG_DRAW_SPHERES
    OptixModule optixSphereModule;
    OptixBuiltinISOptions builtin_is_options = {};
    builtin_is_options.usesMotionBlur = false;
    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    CUDA_CHECKED_CALL optixBuiltinISModuleGet(optixState.deviceContext, &moduleCompileOptions, &pipelineCompileOptions, &builtin_is_options, &optixSphereModule);

    pgClosestHitPTDesc.hitgroup.moduleIS = optixSphereModule;
    pgClosestHitPTDesc.hitgroup.entryFunctionNameIS = nullptr;
#else
    pgClosestHitPTDesc.hitgroup.moduleIS = optixModule;
    pgClosestHitPTDesc.hitgroup.entryFunctionNameIS = "__intersection__TransformDAG16";
#endif

    auto pgClosestHitShadowDesc = pgClosestHitPTDesc;
    pgClosestHitShadowDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";

    const std::array programGroupDescs { pgExceptionDesc, pgRayGenDesc, pgMissDesc, pgMissShadowDesc, pgClosestHitPTDesc, pgClosestHitShadowDesc };
    std::vector<OptixProgramGroup> programGroups(programGroupDescs.size());
    OptixProgramGroupOptions pgOptions {};
    logStringSize = logString.size();
    CUDA_CHECKED_CALL optixProgramGroupCreate(optixState.deviceContext, programGroupDescs.data(), (unsigned)programGroups.size(), &pgOptions, logString.data(), &logStringSize, programGroups.data());
    if (logStringSize > 1)
        spdlog::info("{}", logString.data());

    OptixPipeline optixPipeline;
    logStringSize = logString.size();
    CUDA_CHECKED_CALL optixPipelineCreate(optixState.deviceContext, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), (int)programGroups.size(), logString.data(), &logStringSize, &optixPipeline);
    if (logStringSize > 1)
        spdlog::info("{}", logString.data());

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stackSizes {};
    for (const auto& programGroup : programGroups)
        CUDA_CHECKED_CALL optixUtilAccumulateStackSizes(programGroup, &stackSizes, optixPipeline);

    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t continuationStackSize;
    CUDA_CHECKED_CALL optixUtilComputeStackSizes(
        &stackSizes,
        maxTraceDepth,
        maxContinuationCallDepth,
        maxDirectCallDepth,
        &directCallableStackSizeFromTraversal,
        &directCallableStackSizeFromState,
        &continuationStackSize);

    // CUDA_CHECKED_CALL optixPipelineSetStackSize(
    //     optixPipeline,
    //     directCallableStackSizeFromTraversal,
    //     directCallableStackSizeFromState,
    //     continuationStackSize,
    //     maxTraversalDepth);

    SBTBuilder sbtBuilder;
    sbtBuilder.setExceptionRecord(programGroups[0]);
    sbtBuilder.setRayGenRecord(programGroups[1]);
    sbtBuilder.addMissRecord(programGroups[2]);
    sbtBuilder.addMissRecord(programGroups[3]);
    sbtBuilder.addHitGroupRecord(programGroups[4]);
    sbtBuilder.addHitGroupRecord(programGroups[5]);
    const auto [shaderBindingTableDesc, pShaderBindingTable] = sbtBuilder.finalize();

    return OptixProgram_ {
        .module_ = optixModule,
        .programGroups = std::move(programGroups),
        .pipeline = optixPipeline,
        .shaderBindingTableDesc = shaderBindingTableDesc,
        .pShaderBindingTable = pShaderBindingTable,
    };
}

void destroyOptixProgram(OptixProgram_& program)
{
    optixPipelineDestroy(program.pipeline);
    for (auto&& programGroup : program.programGroups)
        optixProgramGroupDestroy(programGroup);
    Memory::free(program.pShaderBindingTable);
    optixModuleDestroy(program.module_);
}

OptixDenoiser_ createOptixDenoiser(const OptixState_& optixState)
{
    OptixDenoiserOptions options { .guideAlbedo = false, .guideNormal = false, .denoiseAlpha = OptixDenoiserAlphaMode::OPTIX_DENOISER_ALPHA_MODE_COPY };
    OptixDenoiser denoiser;
    CUDA_CHECKED_CALL optixDenoiserCreate(optixState.deviceContext, OPTIX_DENOISER_MODEL_KIND_LDR, &options, &denoiser);

    OptixDenoiserSizes denoiserSizes;
    CUDA_CHECKED_CALL optixDenoiserComputeMemoryResources(denoiser, imageWidth, imageHeight, &denoiserSizes);

    auto pStateMemory = Memory::malloc<std::byte>("OptixDenoiser::StateMemory", denoiserSizes.stateSizeInBytes, EMemoryType::GPU_Malloc);
    auto pScratchMemory = Memory::malloc<std::byte>("OptixDenoiser::ScratchMemory", denoiserSizes.withoutOverlapScratchSizeInBytes, EMemoryType::GPU_Malloc);
    CUDA_CHECKED_CALL optixDenoiserSetup(
        denoiser, optixState.cuStream, imageWidth, imageHeight,
        (CUdeviceptr)pStateMemory, denoiserSizes.stateSizeInBytes,
        (CUdeviceptr)pScratchMemory, denoiserSizes.withoutOverlapScratchSizeInBytes);

    return {
        .denoiser = denoiser,
        .pStateMemory = pStateMemory,
        .pScratchMemory = pScratchMemory,
        .stateMemorySizeInBytes = denoiserSizes.stateSizeInBytes,
        .scratchMemorySizeInBytes = denoiserSizes.withoutOverlapScratchSizeInBytes
    };
}

void applyOptixDenoising(OptixState_ state, OptixDenoiser_& denoiser, StaticArray2D<float3> inputImage, StaticArray2D<float3> outputImage)
{
    const OptixImage2D inputImageDesc {
        .data = (CUdeviceptr)inputImage.data(),
        .width = imageWidth,
        .height = imageHeight,
        .rowStrideInBytes = imageWidth * sizeof(float3),
        .pixelStrideInBytes = sizeof(float3),
        .format = OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT3
    };
    const OptixImage2D outputImageDesc {
        .data = (CUdeviceptr)outputImage.data(),
        .width = imageWidth,
        .height = imageHeight,
        .rowStrideInBytes = imageWidth * sizeof(float3),
        .pixelStrideInBytes = sizeof(float3),
        .format = OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT3
    };

    const OptixDenoiserGuideLayer guideLayer {};
    const OptixDenoiserLayer denoiserLayer {
        .input = inputImageDesc,
        .previousOutput = {},
        .output = outputImageDesc,
        .type = {}
    };
    const OptixDenoiserParams params {
        .blendFactor = 0,
    };

    optixDenoiserInvoke(denoiser.denoiser, state.cuStream, &params,
        (CUdeviceptr)denoiser.pStateMemory, denoiser.stateMemorySizeInBytes,
        &guideLayer, &denoiserLayer, 1, 0, 0,
        (CUdeviceptr)denoiser.pScratchMemory, denoiser.scratchMemorySizeInBytes);
}

void destroyOptixDenoiser(OptixDenoiser_& denoiser)
{
    optixDenoiserDestroy(denoiser.denoiser);
    Memory::free(denoiser.pStateMemory);
    Memory::free(denoiser.pScratchMemory);
}
#endif

} // namespace Tracer
