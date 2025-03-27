#include <voxcom/utility/error_handling.h>
#include <voxcom/voxel/export_ssvdag.h>
#include <voxcom/voxel/export_import_structure.h>
#include <voxcom/voxel/structure.h>

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
    CLI::App app { "Test the impact of the SSVDAG bug" };
    app.add_option("in", inFilePath, "Input SVO file")->required(true);
    try {
        app.parse(argc, ppArgv);
    } catch (const CLI::ParseError& e) {
        app.exit(e);
    }

    const auto editStructure = importEditStructure<void>(inFilePath);
    const auto ssvdag_fixed = constructSSVDAG<true>(editStructure);
    const auto ssvdag_original = constructSSVDAG<false>(editStructure);
    verifySSVDAG(editStructure, ssvdag_original);
    spdlog::info("Verified original SSVDAG paper implementation");
    verifySSVDAG(editStructure, ssvdag_fixed);
    spdlog::info("Verified fixed SSVDAG paper implementation");

    const auto printSize = [](const auto& ssvdag, const char* name) {
        spdlog::info("=== {} ===", name);
        size_t numTotalItems = 0;
        for (size_t level = 0; level < ssvdag.nodesPerLevel.size(); ++level) {
            const size_t numItems = (level == ssvdag.subGridLevel ? ssvdag.subGrids.size() : ssvdag.nodesPerLevel[level].size());
            spdlog::info("[{}]: {}", level, numItems);
            numTotalItems += numItems;
        }
        spdlog::info("Total: {}", numTotalItems);
        return numTotalItems;
    };

    const auto numItemsOriginal = printSize(ssvdag_original, "Original Invariance");
    const auto numItemsFixed = printSize(ssvdag_fixed, "Fixed Invariance");
    spdlog::info("{} => {} (-{:.2}%)", numItemsOriginal, numItemsFixed, double(numItemsOriginal - numItemsFixed) / numItemsOriginal * 100.0);
}