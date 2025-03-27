#include <voxcom/utility/error_handling.h>
#include <voxcom/voxel/export_import_structure.h>
#include <voxcom/voxel/structure.h>
#include <voxcom/voxel/transform_dag.h>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <spdlog/spdlog.h>
DISABLE_WARNINGS_POP()

using namespace voxcom;

int main(int argc, char const* const* ppArgv)
{
    std::filesystem::path inFilePath;
    CLI::App app { "Test the impact of using only a subset (S+A) of all 8! transformations" };
    app.add_option("in", inFilePath, "Input SVO file")->required(true);
    try {
        app.parse(argc, ppArgv);
    } catch (const CLI::ParseError& e) {
        app.exit(e);
    }

    const TransformDAGConfig config {
        .symmetry = true, .axisPermutation = true, .translation = false, .maxTranslationLevel = 0
    };
    const auto editStructure = importEditStructure<void>(inFilePath);
    const auto allTransformDAG = findAllHierarchicalTransforms(editStructure);
    const auto subsetTransformDAG = constructTransformDAG(editStructure, config);
    const auto numItemsAllTransforms = allTransformDAG.computeItemCount();
    const auto numItemsSubsetTransforms = subsetTransformDAG.computeItemCount();
    spdlog::info("{} => {} nodes & leaves (+{:.2}%)", numItemsAllTransforms, numItemsSubsetTransforms, double(numItemsSubsetTransforms - numItemsAllTransforms) / numItemsAllTransforms * 100.0);
}