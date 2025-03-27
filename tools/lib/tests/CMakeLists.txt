add_executable(tools_tests
	"src/test_main.cpp"
	"src/core/cuckoo_hash_table.cpp" 
	"src/core/image.cpp"
	"src/voxel/structure.cpp"
	"src/voxel/voxel_grid.cpp"
	"src/voxel/save_structure.cpp" 
	"src/utility/bitstream.cpp"
	"src/core/radix_sort.cpp"
	"src/utility/simd.cpp")
target_link_libraries(tools_tests PRIVATE 
	project_options 
	voxel_compress_lib 
	spdlog::spdlog 
	Catch2::Catch2)

# Work-around for stupid but in Visual Studio TestExplorer which hasn't been fixed for years:
# IN TEST EXPLORER SET GROUP-BY TO CLASS/OBJECT (ANYTHING WITHOUT NAMESPACE)
# 
# https://developercommunity.visualstudio.com/t/cmake-weird-test-name-prefix/793209
# https://stackoverflow.com/questions/60520767/specify-namespace-class-in-vs2019-test-explorer-while-using-ctest-cmake
#
catch_discover_tests(
    tools_tests
	TEST_PREFIX "Tools." # the last dot is required :)
)

# Copy data files used by the tests to the output directory.
add_custom_command(TARGET tools_tests POST_BUILD COMMAND 
	${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_LIST_DIR}/data" "$<TARGET_FILE_DIR:tools_tests>/data")
