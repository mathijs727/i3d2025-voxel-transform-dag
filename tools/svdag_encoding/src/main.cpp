#include "huffman.h"
#include "svdag32_encoding.h"
#include "svo32_encoding.h"
#include "transform_dag_encoding.h"
#include <algorithm>
#include <execution>
#include <filesystem>
#include <fstream>
#include <span>
#include <voxcom/utility/error_handling.h>
#include <voxcom/voxel/export_import_structure.h>
#include <voxcom/voxel/export_ssvdag.h>
#include <voxcom/voxel/structure.h>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#if ENABLE_GIT_COMMIT_IN_STATS
#include <git.h>
#endif
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
DISABLE_WARNINGS_POP()

using namespace voxcom;

int main(int argc, char const* const* ppArgv)
{
    std::filesystem::path inFilePath, outFilePath, statsFilePath;
    bool skipVerify = false, encodeSVDAG32 = false, encodeSVO32 = false;
    TransformDAGConfig dagConfig { .symmetry = false, .axisPermutation = false, .translation = false, .maxTranslationLevel = 0 };
    TransformEncodingConfig encodingConfig { .pointerTables = false, .huffmanCode = false };
    CLI::App app { "Testbed for DAG compression of voxel octree (without attributes)" };
    app.add_option("in", inFilePath, "Input SVO file")->required(true);
    app.add_flag("--svo32", encodeSVO32, "Construct SVO32 encoding without saving to a file.");
    app.add_flag("--svdag32", encodeSVDAG32, "Construct SVDAG32 encoding without saving to a file.");
    app.add_option("--out", outFilePath, "Output file");
    app.add_option("--stats", statsFilePath, "Statistics file");
    app.add_flag("--symmetry", dagConfig.symmetry, "Search for symmetry invariance");
    app.add_flag("--axis", dagConfig.axisPermutation, "Search for axis permutation");
    app.add_option("--translation", dagConfig.maxTranslationLevel, "Maximum level at which to search for translations (0 to disable translations)");
    app.add_flag("--pointertables", encodingConfig.pointerTables, "Reduce the size of frequently used pointers using an extra level of indirection.");
    app.add_flag("--huffman", encodingConfig.huffmanCode, "Reduce size of pointers using Huffman encoding.");
    app.add_flag("--noverify", skipVerify, "Don't verify that the produced encoding can be correctly decoded.");
    try {
        app.parse(argc, ppArgv);
    } catch (const CLI::ParseError& e) {
        app.exit(e);
    }

    dagConfig.translation = (dagConfig.maxTranslationLevel > 0);
    spdlog::info("inFilePath = {}", inFilePath.string());
    voxcom::assert_always(std::filesystem::exists(inFilePath));

    testHuffman();

    size_t numVoxels = 0;
    DAGEncodingStats encodingStats;
    if (encodeSVO32 || outFilePath.extension() == ".svo32") {
        encodingStats.dagConfig = TransformDAGConfig { .symmetry = false, .axisPermutation = false, .translation = false, .maxTranslationLevel = 0 };
        encodingStats.encodingConfig = TransformEncodingConfig { .pointerTables = false, .huffmanCode = false };
        assert_always(inFilePath.extension() == ".svo");
        auto editStructure = importEditStructure<void>(inFilePath);
        numVoxels = editStructure.computeVoxelCount();
        if (editStructure.structureType == StructureType::Tree)
            editStructure.toDAG();
        const auto svo32 = constructSVO32(editStructure, encodingStats);
        if (!skipVerify) {
            verifySVO32(editStructure, svo32);
            spdlog::info("VERIFIED");
        }
    } else if (encodeSVDAG32 || outFilePath.extension() == ".dag32") {
        encodingStats.dagConfig = TransformDAGConfig { .symmetry = false, .axisPermutation = false, .translation = false, .maxTranslationLevel = 0 };
        encodingStats.encodingConfig = TransformEncodingConfig { .pointerTables = false, .huffmanCode = false };
        assert_always(inFilePath.extension() == ".svo");
        auto editStructure = importEditStructure<void>(inFilePath);
        numVoxels = editStructure.computeVoxelCount();
        if (editStructure.structureType == StructureType::Tree)
            editStructure.toDAG();
        const auto svdag32 = constructSVDAG32(editStructure, encodingStats);
        if (!skipVerify) {
            verifySVDAG32(editStructure, svdag32);
            spdlog::info("VERIFIED");
        }
    } else if (outFilePath.empty() || outFilePath.extension() == ".tdag16") {
        TransformDAG16 encodedDAG;
        if (inFilePath.extension() == ".tdag") {
            std::ifstream file { inFilePath, std::ios::binary };
            BinaryReader reader { file };
            EditStructure<void, TransformPointer> transformDAG;
            reader.read(transformDAG);
            reader.read(dagConfig);
            encodedDAG = constructTransformDAG16(std::move(transformDAG), dagConfig, encodingConfig, encodingStats);
        } else {
            EditStructure<void, TransformPointer> transformDAG;
            EditStructure<void, uint32_t> editStructureCopy;
            std::ifstream file { inFilePath, std::ios::binary };
            BinaryReader reader { file };
            if (inFilePath.extension() == ".svo") {
                EditStructureOOC<void, uint32_t> editStructure = importEditStructureOOC<void>(inFilePath);
                numVoxels = editStructure.computeVoxelCount();
                transformDAG = constructTransformDAG(editStructure, dagConfig);
                if (!skipVerify) {
                    std::ifstream file2 { inFilePath, std::ios::binary };
                    editStructureCopy = importEditStructure<void>(inFilePath);
                }
            } else if (inFilePath.extension() == ".svdag") {
                Bounds bounds;
                EditStructure<void, uint32_t> editStructure = importSVDAG(inFilePath, bounds);
                numVoxels = editStructure.computeVoxelCount();
                transformDAG = constructTransformDAG(editStructure, dagConfig);
                if (!skipVerify)
                    editStructureCopy = editStructure;
            }
            encodedDAG = constructTransformDAG16(std::move(transformDAG), dagConfig, encodingConfig, encodingStats);

            if (!skipVerify) {
                verifyComplexTransformDAG(editStructureCopy, encodedDAG);
                spdlog::info("VERIFIED");
            }
            spdlog::info("Size in bytes: {}", encodedDAG.sizeInBytes());
        }

        if (!outFilePath.empty()) {
            const auto outParentPath = outFilePath.parent_path();
            if (!outParentPath.empty() && !std::filesystem::exists(outParentPath))
                std::filesystem::create_directories(outParentPath);

            std::ofstream file { outFilePath, std::ios::binary };
            BinaryWriter writer { file };
            writer.write(encodedDAG);
        }

        spdlog::info("Size: {}KiB", encodedDAG.sizeInBytes() >> 10);
    }

    if (!statsFilePath.empty()) {
        nlohmann::json outStats;
#if ENABLE_GIT_COMMIT_IN_STATS
        outStats["git_commit_sha"] = git_CommitSHA1();
        outStats["git_commit_date"] = git_CommitDate();
#endif
        outStats["in_file"] = inFilePath;
        outStats["num_levels"] = encodingStats.levels.size() - 1;
        outStats["num_voxels"] = numVoxels;

        nlohmann::json encodingStatsJSON;
        encodingStats.write(encodingStatsJSON);
        outStats["encoding_stats"] = encodingStatsJSON;

        std::ofstream statsFile { statsFilePath };
        statsFile << std::setfill(' ') << std::setw(4) << outStats;

        std::cout << std::setfill(' ') << std::setw(4) << outStats << std::endl;

        size_t numItems = 0;
        for (size_t level = 0; level < encodingStats.levels.size(); ++level) {
            const auto& levelStats = encodingStats.levels[level];
            numItems += levelStats.numNodes;
            spdlog::info("[{}]: {}", level, levelStats.numNodes);
        }
        spdlog::info("Total: {}", numItems);
    }
    return 0;
}