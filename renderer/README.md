# Renderer
This is a fork of [a fork](https://github.com/mathijs727/GPU-SVDAG-Editing) of [the HashDAG framework by Careil et al.](https://github.com/Phyronnaz/HashDAG).

## Running
At the start of the pogram, it tries to load an existing SVDAG from a `data` folder inside the root of this repository. **Create a `data` folder and place an existing HashDAG file there**. You can use the voxelizer included in this project to generate a compatible voxel file from a triangle mesh.

### Memory Allocator
We found that CUDA malloc causes some slow-downs compared to using a custom memory allocator. We use [Vulkan Memory Allocator](https://gpuopen.com/vulkan-memory-allocator/) to perform sub-allocations from a single large `cudaMalloc` allocation performed at start-up.  Vulkan Memory Allocator is enabled by default, but can be disabled with the following CMake option: `-DENABLE_VULKAN_MEMORY_ALLOCATOR=0`.

### Settings
The program is completely configured through compile time definitions. The default parameters are defined in `typedefs.h`.

It is recommended to override them in `src/configuration/script_definitions.h`.
```c++
#pragma once
#include "hash_dag_enum.h"

#define SCENE "ssvdag_bistro_exterior" // CHANGE THIS TO THE SCENE YOU WANT TO LOAD (placed in the renderer/data/ folder)
#define SCENE_DEPTH 16 // CHANGE THIS TO THE NUMBER OF SVDAG LEVELS OF THE SCENE TO LOAD (in this case 2**16=64K voxel resolution per axis).

#define USE_REPLAY 0 // Whether to load and play a replay file.
#define REPLAY_NAME "pretty" // Name of the replay (see replays folder).
#define REPLAY_DEPTH 16 // The number of SVDAG levels when the replay was recorded.

#define ENABLE_CHECKS 1 // Enable debug assertions. May slow down rendering a bit.
#define CAPTURE_GPU_TIMINGS 1 // Capture GPU timings and display them in the User Interface
#define CAPTURE_MEMORY_STATS_SLOW 1 // Capture additional statistics about memory usage; may slow down performance.

#define SAVE_SCENE 0 // Set this to 0
#define DAG_TYPE EDag::TransformDag16 // TransformDag16 for our method, SymmetryAwareDag16 for Symmetry-Aware SVDAG method (Villanueva et al.)
#define EDITS_ENABLE_COLORS 0 // Set this to 0 (colors are not currently supported by TransformDag16)
#define EDITS_ENABLE_MATERIALS 0 // Set this to 0 (colors are not currently supported by TransformDag16)
```

Settings related to *just* the Transform-Aware SVDAG are found in `src/configuration/transform_dag_definitions.h`. These settings **MUST** match the file you are trying to load (see `src/configuration/script_definitions.h`):
```c++
#pragma once
#define TRANSFORM_DAG_USE_POINTER_TABLES 1 // Use an additional level of indirection to reduce the size of frequently used pointers.
#define TRANSFORM_DAG_USE_HUFFMAN_CODE 1 // Encode pointer type (symmetry, axis permutatin, translation) using Huffman coding.
#define TRANSFORM_DAG_USE_SYMMETRY 1 // Transform DAG supports symmetry.
#define TRANSFORM_DAG_USE_AXIS_PERMUTATION 1 // Transform DAG supports axis permutations.
#define TRANSFORM_DAG_USE_TRANSLATION 1 // Transform DAG supports translations.
#define TRANSFORM_DAG_MAX_TRANSLATION_LEVEL 4 // Highest level in the Transform DAG that supports translations.

#define TRANSFORM_DAG_USE_TRANSFORMATION_ID (TRANSFORM_DAG_USE_SYMMETRY || TRANSFORM_DAG_USE_AXIS_PERMUTATION)
#define TRANSFORM_DAG_HAS_TRANSFORMATIONS (TRANSFORM_DAG_USE_SYMMETRY || TRANSFORM_DAG_USE_AXIS_PERMUTATION || TRANSFORM_DAG_USE_TRANSLATION) 
```

### Profiling
The project contains profiling code to measure both CPU & GPU performance (see `src/stats.cpp`); which is how the results in the paper were obtained. The outputs of the profiler is configured through `src/configuration/profile_definitions.h`:
```c++
#pragma once
#include "hash_dag_enum.h"

#define PROFILING_PATH "/path/to/profile/folder/" // Folder where to output profiling results.
#define STATS_FILES_PREFIX "name_of_profile" // Filename for profiling run-time performance.
#define CONSTRUCT_STATS_FILE_PATH "name_of_profile-construction" // Filename for profiling initial SVDAG construction (e.g. memory usage).
```

The code is also instrumented for [Tracy profiler](https://github.com/wolfpld/tracy). 