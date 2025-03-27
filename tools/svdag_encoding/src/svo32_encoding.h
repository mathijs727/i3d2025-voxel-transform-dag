#pragma once
#pragma once
#include "transform_dag_encoding.h"
#include "voxcom/voxel/structure.h"
#include <vector>

struct SVO32 {
    struct InnerNode {
        uint32_t childMask;
        uint32_t firstChildPtr;
    };
    std::vector<std::vector<InnerNode>> nodesPerLevel;
    std::vector<voxcom::EditSubGrid<void>> subGrids; // 4^3 leaves.
};

SVO32 constructSVO32(const voxcom::EditStructure<void, uint32_t>& structure, DAGEncodingStats& stats);
void verifySVO32(const voxcom::EditStructure<void, uint32_t>& editStructure, const SVO32& svo);
