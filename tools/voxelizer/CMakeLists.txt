add_executable(voxelizer "src/main.cpp" "src/octree_to_mesh.h" "src/octree_to_mesh.cpp")
target_link_libraries(voxelizer
    PRIVATE
        project_options
        fmt::fmt
        spdlog::spdlog
		CLI11::CLI11
        voxel_compress_lib
)
