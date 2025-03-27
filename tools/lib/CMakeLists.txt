add_library(voxel_compress_lib
    "src/core/bounds.cpp"
    "src/core/mesh.cpp"
    "src/utility/bitstream.cpp"
    "src/voxel/all_hierarchical_transforms.cpp"
    "src/voxel/voxelize.cpp"
    "src/voxel/export_hashdag.cpp"
    "src/voxel/export_import_structure.cpp"
    "src/voxel/ssvdag.cpp"
    "src/voxel/export_ssvdag.cpp"
    "src/voxel/transform_dag.cpp"
    "src/voxel/transform_dag_hierarchical.cpp"
    "src/voxel/find_translation.cpp"
    "include/voxcom/core/cuckoo_hash_table.h")
target_include_directories(voxel_compress_lib PUBLIC "include" PRIVATE "src")
target_compile_features(voxel_compress_lib PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(voxel_compress_lib PRIVATE
    project_options
    assimp::assimp
    robin_hood::robin_hood
    absl::flat_hash_map
    EASTL
    fmt::fmt
    spdlog::spdlog
    #freeimage::FreeImage freeimage::FreeImagePlus
    libmorton::libmorton
    mio::mio
    glm::glm
    TBB::tbb
    OpenMP::OpenMP_CXX
)

if (MSVC)
    if(ENABLE_AVX512)
        set(MSVC_ARCH "/arch:AVX512")
    endif()
    target_compile_options(voxel_compress_lib PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${MSVC_ARCH}>")
    target_compile_definitions(voxel_compress_lib PUBLIC "-DNOMINMAX=1")
else()
    set(GCC_ARCH "-mavx" "-mavx2" "-mbmi2")
    if(ENABLE_AVX512)
        list(APPEND GCC_ARCH "-mavx512")
    endif()
    target_compile_options(voxel_compress_lib PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${GCC_ARCH}>")
endif()

if (ENABLE_CUDA)
    target_sources(voxel_compress_lib PRIVATE "src/voxel/find_translation.cu")
    target_compile_definitions(voxel_compress_lib PRIVATE "-DCUDA_ENABLED=1")
    set_target_properties(voxel_compress_lib PROPERTIES
        CUDA_ARCHITECTURES "86"
        CUDA_SEPARABLE_COMPILATION ON # Seems like a requirement for CUDA debugging?
        CUDA_RESOLVE_DEVICE_SYMBOLS ON # Required on Windows when using CMake to generate a Visual Studio Solution file.
    )
    # Enable -G debug flag for CUDA in Debug mode
    # https://gitlab.kitware.com/cmake/cmake/-/issues/19017
    target_compile_options(voxel_compress_lib PUBLIC "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
    target_compile_options(voxel_compress_lib PUBLIC "$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")
    target_compile_options(voxel_compress_lib PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>")
endif()

# Image in a separate *.lib file because FreeImage injects nasty stuff into the compiler arguments, causing CUDA compilation to fail otherwise.
add_library(voxel_compress_lib_image "src/core/image.cpp")
target_include_directories(voxel_compress_lib_image PUBLIC "include" PRIVATE "src")
target_link_libraries(voxel_compress_lib_image PRIVATE
    project_options
    fmt::fmt
    spdlog::spdlog
    freeimage::FreeImage freeimage::FreeImagePlus
    libmorton::libmorton
    glm::glm
)
target_link_libraries(voxel_compress_lib PUBLIC voxel_compress_lib_image)

if (ENABLE_TESTS)
    add_subdirectory("tests")
endif()
