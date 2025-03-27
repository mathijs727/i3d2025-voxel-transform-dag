#pragma once
#pragma once
#include "voxcom/voxel/structure.h"
#include <vector>
#include "transform_dag_encoding.h"

struct SVDAG32 {
    std::vector<std::vector<uint32_t>> nodesPerLevel;
    std::vector<voxcom::EditSubGrid<void>> subGrids; // 4^3 leaves.
};

SVDAG32 constructSVDAG32(const voxcom::EditStructure<void, uint32_t>& structure, DAGEncodingStats& stats);
void verifySVDAG32(const voxcom::EditStructure<void, uint32_t>& editStructure, const SVDAG32& svdag);
