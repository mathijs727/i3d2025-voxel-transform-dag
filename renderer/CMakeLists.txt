set(OPTIX_DIR "" CACHE PATH "Enable OptiX ray tracing")
option(ENABLE_VULKAN_MEMORY_ALLOCATOR "Allocate GPU memory using VMA instead of calling cudaMalloc (faster)" TRUE)
option(ENABLE_TRACY "Enable traces for tracy profiler" FALSE)
option(ENABLE_GIT_COMMIT_IN_STATS "Store git commit hash in the stats output file" FALSE)

add_library(renderer_lib
    "src/configuration/script_definitions.h" 
	"src/configuration/profile_definitions.h" 
    "src/configuration/gpu_hash_dag_definitions.h"

    "src/dags/my_gpu_dags/create_edit_svo.cu"
    "src/dags/my_gpu_dags/create_edit_svo.h"
    "src/dags/my_gpu_dags/my_gpu_base_dag.h"
    "src/dags/my_gpu_dags/my_gpu_dag_editors.h"
    "src/dags/my_gpu_dags/my_gpu_dag_item.h"

    "src/dags/my_gpu_dags/cub/cub_scan.cu"
    "src/dags/my_gpu_dags/cub/cub_scan.h"
    "src/dags/my_gpu_dags/cub/cub_merge_sort.cu"
    "src/dags/my_gpu_dags/cub/cub_merge_sort.h"
    "src/dags/my_gpu_dags/cub/cub_radix_sort.cu"
    "src/dags/my_gpu_dags/cub/cub_radix_sort.h"
    "src/dags/my_gpu_dags/cub/cub_select.cu"
    "src/dags/my_gpu_dags/cub/cub_select.h"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag_edits.cu"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag_edits.h"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag_factory.cpp"
    #"src/dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag_threaded_factory.cpp"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag_factory.h"

    "src/test_shared_memory.cu"
    "src/test_shared_memory.h"
    "src/work_queue.h"

    "src/dags/hash_dag/hash_dag.h"
    "src/dags/hash_dag/hash_dag_colors.h"
    "src/dags/hash_dag/hash_dag_editors.h"
    "src/dags/hash_dag/hash_dag_factory.cpp"
    "src/dags/hash_dag/hash_dag_factory.h"
    "src/dags/hash_dag/hash_dag_globals.h"
    "src/dags/hash_dag/hash_table.cpp"
    "src/dags/hash_dag/hash_table.h"

	"src/dags/basic_dag/basic_dag.cpp"
    "src/dags/basic_dag/basic_dag.h"

    "src/dags/transform_dag/transform_dag.cpp"
    "src/dags/transform_dag/transform_dag.cu"
    "src/dags/transform_dag/transform_dag.h"

    "src/dags/base_dag.h"
    "src/dags/dag_utils.h"

    "src/gmath/Matrix3x3.h"
    "src/gmath/Quaternion.h"
    "src/gmath/Vector2.h"
    "src/gmath/Vector3.h"
    "src/array.h"
    "src/array2d.h"
    "src/depth_image.cpp"
    "src/depth_image.h"
    "src/opengl_texture.cpp"
    "src/opengl_texture.h"
    "src/binary_reader.h"
	"src/binary_writer.h"
	"src/camera_view.h"
    "src/color_utils.h"
	"src/cuda_error_check.h"
    "src/cuda_gl_buffer.h"
    "src/cuda_helpers.h"
    "src/cuda_helpers_cpp.cu"
    "src/cuda_helpers_cpp.h"
	"src/cuda_math.h"
    "src/dag_tracer.cu"
    "src/dag_tracer.h"
    "src/engine.cpp"
    "src/engine.h"
    "src/events.cpp"
    "src/events.h"
    "src/is_of_type.h"
    "src/image.cpp"
    "src/image.h"
    "src/opengl_interop.h"
    "src/path.h"
    "src/replay.cpp"
    "src/replay.h"
    "src/shader.cpp"
    "src/shader.h"
    "src/constexpr_for.h"
    "src/move_only.h"
    "src/stats.cpp"
    "src/stats.h"
    "src/timings.cpp"
    "src/timings.h"
    "src/tracer.cu"
    "src/tracer_optix.cu"
    "src/tracer.h"
    "src/typedefs.h"
    "src/utils.h"
	"src/FastNoise.cpp"
	"src/FastNoise.h"
	"src/hacky_profiler.cpp"
	"src/hacky_profiler.hpp"
	"src/hacky_profiler.inl"
	"src/dags/hash_dag/hash_dag_edits.h"
	"src/dags/hash_dag/hash_dag_utils.h"
    "src/memory.cu"
    "src/memory.h"
    "src/safe_cooperative_groups.h"
    "src/blur.cu"
    "src/blur.h"

    "src/video.h"
    "src/video.cpp"
    "src/voxel_textures.cpp"
    "src/voxel_textures.h"
	 
    "src/dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.cu"

    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/dynamic_slab_alloc.cu"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/dynamic_slab_alloc.h"

    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/individual_chaining_hash_table.h"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/ticket_board_hash_table.cu"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/ticket_board_hash_table.h"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/acceleration_hash_table.cu"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/acceleration_hash_table.h"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/compact_acceleration_hash_table.cu"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/compact_acceleration_hash_table.h"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/atomic64_hash_table.cu"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/atomic64_hash_table.h"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/duplicate_detection_hash_table.cu"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/duplicate_detection_hash_table.h"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/gpu_hash_table_base.cu"
    "src/dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/gpu_hash_table_base.h"
    "src/dags/symmetry_aware_dag/symmetry_aware_dag.cpp")

target_include_directories(renderer_lib PUBLIC "src")
target_link_libraries(renderer_lib
    PUBLIC project_options GLEW::GLEW glfw OpenGL::GL CUDA::cudart CUDA::toolkit CUDA::nvml CUDA::cuda_driver fmt::fmt magic_enum::magic_enum imgui::imgui
    PRIVATE Renderer_image TBB::tbb Eigen3::Eigen EASTL glm::glm unofficial::nativefiledialog::nfd nlohmann_json date::date)

if (OPTIX_DIR)
    add_library(Optix INTERFACE)
    target_include_directories(Optix INTERFACE "${OPTIX_DIR}/include")
    target_link_libraries(renderer_lib PUBLIC Optix)
    target_compile_definitions(renderer_lib PUBLIC "-DENABLE_OPTIX=1")

    add_library(renderer_optix OBJECT "src/tracer_optix_kernel.cu")
    target_include_directories(renderer_optix PUBLIC "src")
    target_link_libraries(renderer_optix PRIVATE nlohmann_json Optix project_options)
    target_compile_definitions(renderer_optix PRIVATE "-DIS_OPTIX_KERNEL=1")

    # Enable -G debug flag for CUDA in Debug mode
    # https://gitlab.kitware.com/cmake/cmake/-/issues/19017
    
    set_target_properties(renderer_optix PROPERTIES
        CUDA_ARCHITECTURES "86" # 86 (Ampere) required for warp reduction functions.
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_OPTIX_COMPILATION ON
    )
    target_compile_definitions(renderer_lib PUBLIC "-DOPTIX_PROGRAM_FILE=\"$<TARGET_OBJECTS:renderer_optix>\"")
    add_dependencies(renderer_lib renderer_optix)
else()
    target_compile_definitions(renderer_lib PUBLIC "-DENABLE_OPTIX=0")
endif()

# Work-around for freeimage::FreeImage adding broken compiler flags which breaks compilation of CUDA files on Windows.
add_library(Renderer_image "src/free_image_wrapper.cpp" "src/free_image_wrapper.h")
target_compile_features(Renderer_image PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(Renderer_image PRIVATE freeimage::FreeImage freeimage::FreeImagePlus)

if (ENABLE_GIT_COMMIT_IN_STATS)
    target_compile_definitions(renderer_lib PRIVATE "-DENABLE_GIT_COMMIT_IN_STATS=$<BOOL:${ENABLE_GIT_COMMIT_IN_STATS}>")
    target_link_libraries(renderer_lib PRIVATE cmake_git_version_tracking)
endif()
if (WIN32)
    target_link_libraries(renderer_lib PRIVATE ws2_32) # Link to winsock to get hostname (output of stats.json)
endif()
if (ENABLE_TRACY)
    find_package(Tracy CONFIG REQUIRED)
    target_link_libraries(renderer_lib PUBLIC Tracy::TracyClient)
    target_compile_definitions(renderer_lib PUBLIC "TRACY_ENABLE=1")
endif()
if (ENABLE_VULKAN_MEMORY_ALLOCATOR)
    find_package(Vulkan)
    if (Vulkan_FOUND)
        find_package(VulkanMemoryAllocator CONFIG REQUIRED)
    else()
        message(WARNING "Vulkan could not be found. Disabling Vulkan Memory Allocator support!")
        set(ENABLE_VULKAN_MEMORY_ALLOCATOR FALSE)
    endif()
    target_link_libraries(renderer_lib PUBLIC Vulkan::Headers GPUOpen::VulkanMemoryAllocator Vulkan::Vulkan)
    target_compile_definitions(renderer_lib PUBLIC "ENABLE_VULKAN_MEMORY_ALLOCATOR=1")
endif()
target_compile_features(renderer_lib PUBLIC cxx_std_20 cuda_std_20)
target_compile_definitions(renderer_lib PUBLIC "ROOT_FOLDER=\"${CMAKE_CURRENT_LIST_DIR}\"")

set_target_properties(renderer_lib PROPERTIES
    CUDA_ARCHITECTURES "86" # 86 (Ampere) required for warp reduction functions.
    CUDA_SEPARABLE_COMPILATION ON # Seems like a requirement for CUDA debugging?
    CUDA_RESOLVE_DEVICE_SYMBOLS ON # Required on Windows when using CMake to generate a Visual Studio Solution file.
)

# Enable -G debug flag for CUDA in Debug mode
# https://gitlab.kitware.com/cmake/cmake/-/issues/19017
target_compile_options(renderer_lib PUBLIC "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(renderer_lib PUBLIC "$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")
target_compile_definitions(renderer_lib PUBLIC "-DCOMPILE_OPTIX_DEBUG=$<IF:$<CONFIG:Debug>,1,0>")

if (NOT WIN32)
    # Tell the compiler to target the platform that it is currently running on.
    # This will automatically enable AVX2 support on supported machines.
    target_compile_options(renderer_lib PRIVATE "-march=native")
endif()

# Enable parallel compilation when using the Visual Studio generator (required for CUDA debugging on Windows).
string(FIND "${CMAKE_GENERATOR}" "Visual Studio" VS_GENERATOR_SUBSTR_POS)
if (NOT ${VS_GENERATOR_SUBSTR_POS} MATCHES -1)
    target_compile_options(renderer_lib PUBLIC "/MP")
endif()

if (NOT WIN32)
    # Add -Xcompiler to mirror some host compiler flags.
    target_compile_options(renderer_lib PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler \"-march=native -ffast-math\">")
endif()

add_executable(renderer "src/main.cpp")
target_link_libraries(renderer PUBLIC renderer_lib)
