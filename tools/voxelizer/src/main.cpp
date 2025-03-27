#include "octree_to_mesh.h"
#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <bit>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <voxcom/core/bounds.h>
#include <voxcom/core/mesh.h>
#include <voxcom/utility/binary_writer.h>
#include <voxcom/voxel/export_ssvdag.h>
#include <voxcom/voxel/export_hashdag.h>
#include <voxcom/voxel/export_import_structure.h>
#include <voxcom/voxel/structure.h>
#include <voxcom/voxel/transform_dag.h>
#include <voxcom/voxel/voxelize.h>

#define GLM_ENABLE_EXPERIMENTAL 1
#include <glm/gtx/component_wise.hpp>

using namespace voxcom;

int main(int argc, char const* const* const ppArgv)
{
    std::filesystem::path meshFilePath, outFilePath, outSVDAGFile, outUSSVDAGFile, hashDAGFilePath, outTransformTDAGFile, boundsMeshFilePath, differenceMeshFilePath, voxelMeshFilePath, voxelTreeMeshFolderPath;
    unsigned octreeResolution;
    bool solidVoxelization = false, conservativeVoxelization = false;
    bool axis = false, symmetry = false;
    uint32_t translation = 0;

    CLI::App app { "Voxelize a mesh (with colours) and store it in various formats" };
    app.add_option("mesh", meshFilePath, "Input mesh file")->required(true);
    app.add_option("resolution", octreeResolution, "EditStructure resolution expressed as an exponent of 2 (e.g. 10 => 2^10 => 1024)");
    app.add_flag("--solid", solidVoxelization, "Solid voxelization (filling inside). This requires the mesh to be water tight.");
    app.add_flag("--conservative", conservativeVoxelization, "Conservative voxelization (fill any voxel that touches the mesh).");
    app.add_option("--out", outFilePath, "Output voxel **encoded optOctree** file");
    app.add_option("--svdag", outSVDAGFile, "SVDAG file compatible with the *SSVDAG* codebase");
    app.add_option("--ussvdag", outUSSVDAGFile, "Universal Symmetry-Aware SSVDAG file compatible with the *SSVDAG* codebase");
    app.add_option("--hashdag", hashDAGFilePath, "DAG file compatible with the *HashDAG* codebase");
    app.add_option("--tdag", outTransformTDAGFile, "Output TransformDAG which can be read by dag_encoding_test_bed");
    app.add_flag("--symmetry", symmetry, "Apply symmetry compression to the output TransformDAG");
    app.add_flag("--axis", axis, "Apply axis permutation compression to the output TransformDAG");
    app.add_option("--translation", translation, "Apply translation compression to the output TransformDAG");
    app.add_option("--boundsmesh", boundsMeshFilePath, "Use the bounds of a different mesh to scale the object.");
    app.add_option("--difference", differenceMeshFilePath, "(--voxel only): output difference between the main input mesh and this input mesh (may be usefull for making visualizations)");
    app.add_option("--voxel", voxelMeshFilePath, "Output voxel **mesh** file");
    app.add_option("--voxeltree", voxelTreeMeshFolderPath, "Output voxel **mesh** files for each level in the hierarchy");
    try {
        app.parse(argc, ppArgv);
    } catch (const CLI::ParseError& e) {
        app.exit(e);
    }
    octreeResolution = 1 << octreeResolution;

    // using Attribute = voxcom::RGB;
    using Attribute = void;
    std::optional<EditStructure<Attribute, uint32_t>> optOctree;
    std::optional<EditStructureOOC<Attribute, uint32_t>> optOctreeOOC;
    Bounds meshSceneBounds, octreeSceneBounds;

    // Voxel grid for resolutions smaller than 8x8x8.
    std::optional<VoxelGrid<Attribute>> optVoxelGrid;
    if (!voxelMeshFilePath.empty() || !voxelTreeMeshFolderPath.empty())
        optVoxelGrid = VoxelGrid<Attribute>(octreeResolution);

    if (meshFilePath.extension() == ".svdag") {
        optOctree = importSVDAG(meshFilePath, octreeSceneBounds);
        optOctree->toDAG();
        octreeSceneBounds = { .lower = glm::vec3(-1), .upper = glm::vec3(+1) };
    } else if (meshFilePath.extension() == ".svo") {
        optOctreeOOC = importEditStructureOOC<Attribute>(meshFilePath);
        octreeSceneBounds = { .lower = glm::vec3(-1), .upper = glm::vec3(+1) };
    } else {
        spdlog::info("Loading mesh");
        auto meshes = loadMeshes(meshFilePath);

        spdlog::info("Computing bounding box of mesh");
        if (boundsMeshFilePath.empty()) {
            for (const auto& mesh : meshes)
                meshSceneBounds.extend(mesh.computeBounds());
        } else {
            const auto boundsMeshes = loadMeshes(boundsMeshFilePath);
            for (const auto& mesh : boundsMeshes)
                meshSceneBounds.extend(mesh.computeBounds());
        }
        spdlog::info("Scaling meshes to fit optOctree");
        octreeSceneBounds = meshSceneBounds;
        meshes = prepareMeshForVoxelization(meshes, octreeResolution, octreeSceneBounds);

        spdlog::info("Voxelizing...");
        if (solidVoxelization) {
            optOctree = voxelizeSparseSolid(meshes, octreeResolution);
        } else { // Surface voxelization
#if 0 // Single threaded algorithms
            size_t meshIdx = 0;
            for (const auto& mesh : meshes) {
                spdlog::info("Mesh {} / {}", ++meshIdx, meshes.size());
                const auto voxelize = [&](auto& target) {
                    if (conservativeVoxelization)
                        voxelizeMeshOptimized<true>(target, mesh);
                    else
                        voxelizeMeshOptimized<false>(target, mesh);
                };
                if (optVoxelGrid)
                    voxelize(*optVoxelGrid);
                else
                    voxelize(*optOctree);
            }
#else // Multi threaded algorithms.
            optOctree = voxelizeHierarchical<Attribute, false>(meshes, octreeResolution);
#endif
        }
        spdlog::info("Converting optOctree to DAG");
        optOctree->toDAG();
    }

    const auto invokeWithActiveStructure = [&](auto&& f) {
        if (optOctree)
            return f(*optOctree);
        else if (optOctreeOOC)
            return f(*optOctreeOOC);
        else {
            assert_always(false);
            return f(*optOctree); // Work-around for missing return type warning.
        }
    };

    spdlog::info("Saving...");
    if (!outFilePath.empty()) {
        invokeWithActiveStructure([&](const auto& octree) { exportStructure(octree, outFilePath); });
    }
    if (!outSVDAGFile.empty()) {
        invokeWithActiveStructure([&](const auto& octree) { exportUSSVDAG(octree, octreeSceneBounds, outSVDAGFile); });
    }
    if (!outUSSVDAGFile.empty()) {
        invokeWithActiveStructure([&](const auto& octree) {
            const auto symmetryDAG = constructSSVDAG<false>(octree);
            exportUSSVDAG(symmetryDAG, octreeSceneBounds, outUSSVDAGFile);
        });
    }
    if (!hashDAGFilePath.empty()) {
        invokeWithActiveStructure([&](const auto& octree) { exportHashDAG(octree, octreeSceneBounds, hashDAGFilePath); });
    }
    if (!outTransformTDAGFile.empty()) {
        const TransformDAGConfig config { .symmetry = symmetry, .axisPermutation = axis, .translation = (translation > 0), .maxTranslationLevel = translation };
        const auto transformDAG = invokeWithActiveStructure([&](const auto& octree) { return constructTransformDAG(octree, config); });
        std::ofstream file { outTransformTDAGFile, std::ios::binary };
        BinaryWriter writer { file };
        writer.write(transformDAG);
        writer.write(config);
    }

    if (!voxelTreeMeshFolderPath.empty()) {
        if (!std::filesystem::exists(voxelTreeMeshFolderPath))
            std::filesystem::create_directories(voxelTreeMeshFolderPath);

        std::vector<VoxelGrid<Attribute>> gridPyramid;
        gridPyramid.push_back(*optVoxelGrid);
        while (gridPyramid.back().resolution > 1) {
            gridPyramid.push_back(gridPyramid.back().downSample2());
        }

        voxcom::VoxelGrid<Attribute> differenceVoxelGrid(optVoxelGrid->resolution);
        if (!differenceMeshFilePath.empty()) {
            assert_always(optVoxelGrid.has_value());

            auto differenceMeshes = loadMeshes(differenceMeshFilePath);
            auto sceneBounds = meshSceneBounds;
            differenceMeshes = prepareMeshForVoxelization(differenceMeshes, octreeResolution, sceneBounds);
            for (const auto& mesh : differenceMeshes) {
                if (conservativeVoxelization)
                    voxelizeMeshOptimized<true>(differenceVoxelGrid, mesh);
                else
                    voxelizeMeshOptimized<false>(differenceVoxelGrid, mesh);
            }

            std::vector<VoxelGrid<Attribute>> differenceGridPyramid;
            differenceGridPyramid.push_back(differenceVoxelGrid);
            while (differenceGridPyramid.back().resolution > 1) {
                differenceGridPyramid.push_back(differenceGridPyramid.back().downSample2());
            }

            // Compute difference.
            // std::transform(std::begin(gridPyramid), std::end(gridPyramid), std::begin(differenceGridPyramid),
            //    std::begin(gridPyramid),
            //    [](const VoxelGrid<Attribute>& originalGrid, const VoxelGrid<Attribute>& diferenceGrid) {
            //        VoxelGrid<Attribute> out { originalGrid.resolution };
            //        std::transform(
            //            std::begin(originalGrid.filled), std::end(originalGrid.filled), std::begin(diferenceGrid.filled), std::begin(out.filled),
            //            [](bool original, bool difference) {
            //                return original && !difference;
            //            });
            //        return out;
            //    });

            assert(gridPyramid.back().filled.size() == 1);
            gridPyramid.back().filled[0] = !differenceGridPyramid.back().filled[0];
            for (int level = (int)gridPyramid.size() - 2; level >= 0; --level) {
                auto& outGrid = gridPyramid[level];
                const auto& differenceGrid = differenceGridPyramid[level];
                const auto differenceParentGrid = differenceGridPyramid[level + 1].upSample2();
                for (size_t i = 0; i < outGrid.filled.size(); ++i) {
                    const bool original = outGrid.filled[i];
                    const bool difference = differenceGrid.filled[i];
                    const bool differenceParent = level == 0 ? true : differenceParentGrid.filled[i]; // Always output last layer (visualization in Blender).
                    outGrid.filled[i] = original && !difference && differenceParent;
                }
                // std::transform(std::begin(out.filled), std::end(out.filled), std::begin(parent.filled), std::begin(parent.filled), std::begin(out.filled),
                //     [](bool original, bool parent) { return original && !parent; });
            }

            // Hide regions already covered by a higher resolution in the pyramid.
            // auto parent = differenceGridPyramid.back();
            // for (int level = (int)gridPyramid.size() - 2; level >= 0; --level) {
            //    parent = parent.upSample2();
            //    auto& out = gridPyramid[level];
            //    std::transform(std::begin(out.filled), std::end(out.filled), std::begin(parent.filled), std::begin(out.filled),
            //        [](bool original, bool parent) { return original && !parent; });
            //}
        }

        for (const auto& voxelGrid : gridPyramid) {
            auto voxelMeshes = octreeToMesh(voxelGrid);
            const float maxExtent = glm::compMax(meshSceneBounds.extent());
            const auto scale = glm::vec3(maxExtent) / glm::vec3((float)voxelGrid.resolution);
            for (auto& mesh : voxelMeshes) {
                for (auto& pos : mesh.positions)
                    pos = meshSceneBounds.lower + pos * scale;
            }

            const auto filePath = voxelTreeMeshFolderPath / std::format("voxel_tree_{}.obj", std::bit_width(voxelGrid.resolution) - 1);
            saveMesh(voxelMeshes, filePath);
        }

        /* auto voxelGrid = *optVoxelGrid;
        while (voxelGrid.resolution >= 1) {
            auto outVoxelGrid = voxelGrid;
            std::transform(std::begin(voxelGrid.filled), std::end(voxelGrid.filled), std::begin(differenceVoxelGrid.filled), std::begin(outVoxelGrid.filled),
                [](bool originallyFilled, bool differenceFilled) {
                    return originallyFilled && !differenceFilled;
                });

            auto voxelMeshes = octreeToMesh(outVoxelGrid);
            const float maxExtent = glm::compMax(meshSceneBounds.extent());
            const auto scale = glm::vec3(maxExtent) / glm::vec3((float)voxelGrid.resolution);
            for (auto& mesh : voxelMeshes) {
                for (auto& pos : mesh.positions)
                    pos = meshSceneBounds.lower + pos * scale;
            }

            const auto filePath = voxelTreeMeshFolderPath / std::format("voxel_tree_{}.obj", std::bit_width(voxelGrid.resolution) - 1);
            saveMesh(voxelMeshes, filePath);

            voxelGrid = voxelGrid.downSample2();
            differenceVoxelGrid = differenceVoxelGrid.downSample2();
        }*/
    }

    if (!differenceMeshFilePath.empty()) {
        assert_always(optVoxelGrid.has_value());

        voxcom::VoxelGrid<Attribute> differenceTarget { optVoxelGrid->resolution };
        auto differenceMeshes = loadMeshes(differenceMeshFilePath);
        auto sceneBounds = meshSceneBounds;
        differenceMeshes = prepareMeshForVoxelization(differenceMeshes, octreeResolution, sceneBounds);
        for (const auto& mesh : differenceMeshes) {
            if (conservativeVoxelization)
                voxelizeMeshOptimized<true>(differenceTarget, mesh);
            else
                voxelizeMeshOptimized<false>(differenceTarget, mesh);
        }

        // std::transform(std::begin(optVoxelGrid->filled), std::end(optVoxelGrid->filled), std::begin(differenceTarget.filled), std::begin(optVoxelGrid->filled),
        //     [](bool originallyFilled, bool differenceFilled) {
        //         return originallyFilled differenceFilled;
        //     });
    }

    if (!voxelMeshFilePath.empty()) {
        auto voxelMeshes = octreeToMesh(*optVoxelGrid);
        // const float maxExtent = glm::compMax(meshSceneBounds.extent());
        // const auto scale = glm::vec3(maxExtent) / glm::vec3((float)optVoxelGrid->resolution);
        // for (auto& mesh : voxelMeshes) {
        //     for (auto& pos : mesh.positions)
        //         pos = meshSceneBounds.lower + pos * scale;
        // }
        saveMesh(voxelMeshes, voxelMeshFilePath);
    }

    spdlog::info("Done");
    return 0;
}
