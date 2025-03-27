add_executable(svdag_encoding
    "src/main.cpp"
    "src/transform_dag_encoding.h"
    "src/no_transform_dag_encoding.cpp"
    "src/transform_dag_encoding.cpp" 
    "src/pointer_encoding.h"
    "src/ssvdag_encoding.cpp"
    "src/huffman.h"
    "src/huffman.cpp"
    "src/pch.h"
    "src/svdag32_encoding.cpp"
    "src/svdag32_encoding.h"
    "src/svo32_encoding.cpp"
    "src/svo32_encoding.h")
target_include_directories(svdag_encoding PUBLIC "include")
target_compile_features(svdag_encoding PUBLIC cxx_std_23)

target_link_libraries(svdag_encoding
    PRIVATE
        project_options
        fmt::fmt
        spdlog::spdlog
		CLI11::CLI11
        voxel_compress_lib
        robin_hood::robin_hood
        Eigen3::Eigen
        nlohmann_json
)
if (ENABLE_GIT_COMMIT_IN_STATS)
    target_link_libraries(svdag_encoding PRIVATE cmake_git_version_tracking)
    target_compile_definitions(svdag_encoding PRIVATE "-DENABLE_GIT_COMMIT_IN_STATS=$<BOOL:${ENABLE_GIT_COMMIT_IN_STATS}>")
endif()

if (MSVC)
    target_compile_options(svdag_encoding PUBLIC "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/bigobj>")
    target_precompile_headers(svdag_encoding PRIVATE "src/pch.h")
endif()

# Enable -G debug flag for CUDA in Debug mode
# https://gitlab.kitware.com/cmake/cmake/-/issues/19017
#target_compile_options(svdag_encoding PUBLIC "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
#target_compile_options(svdag_encoding PUBLIC "$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
#set_target_properties(svdag_encoding PROPERTIES
#    CUDA_ARCHITECTURES "70"
#    CUDA_SEPARABLE_COMPILATION ON # Seems like a requirement for CUDA debugging?
#    CUDA_RESOLVE_DEVICE_SYMBOLS ON # Required on Windows when using CMake to generate a Visual Studio Solution file.
#)
