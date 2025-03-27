#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>
#include <spdlog/spdlog.h>

// https://github.com/catchorg/Catch2/blob/devel/docs/own-main.md
int main(int argc, char* argv[])
{
    // global setup...
    spdlog::set_level(spdlog::level::err);

    int result = Catch::Session().run(argc, argv);

    // global clean-up...

    return result;
}
