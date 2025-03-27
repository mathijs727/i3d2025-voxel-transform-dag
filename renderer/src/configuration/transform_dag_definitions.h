#pragma once
#define TRANSFORM_DAG_USE_POINTER_TABLES 1 // Use an additional level of indirection to reduce the size of frequently used pointers.
#define TRANSFORM_DAG_USE_HUFFMAN_CODE 1 // Encode pointer type (symmetry, axis permutatin, translation) using Huffman coding.
#define TRANSFORM_DAG_USE_SYMMETRY 1 // Transform DAG supports symmetry.
#define TRANSFORM_DAG_USE_AXIS_PERMUTATION 1 // Transform DAG supports axis permutations.
#define TRANSFORM_DAG_USE_TRANSLATION 1 // Transform DAG supports translations.
#define TRANSFORM_DAG_MAX_TRANSLATION_LEVEL 4 // Highest level in the Transform DAG that supports translations.

#define TRANSFORM_DAG_USE_TRANSFORMATION_ID (TRANSFORM_DAG_USE_SYMMETRY || TRANSFORM_DAG_USE_AXIS_PERMUTATION)
#define TRANSFORM_DAG_HAS_TRANSFORMATIONS (TRANSFORM_DAG_USE_SYMMETRY || TRANSFORM_DAG_USE_AXIS_PERMUTATION || TRANSFORM_DAG_USE_TRANSLATION) 