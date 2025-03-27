add_executable(transform_subset_comparison "src/main.cpp")
target_compile_features(transform_subset_comparison   PUBLIC cxx_std_23)
target_link_libraries(transform_subset_comparison
    PRIVATE
        project_options
        fmt::fmt
        spdlog::spdlog
		CLI11::CLI11
        voxel_compress_lib
)
