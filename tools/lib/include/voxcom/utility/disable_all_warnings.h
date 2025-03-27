#pragma once

#if defined(__clang__)
#define CLANG 1
#elif defined(_MSC_VER)
#define MSVC 1
#endif

#ifdef CLANG
#define DISABLE_WARNINGS_PUSH()                                           \
    _Pragma("clang diagnostic push");                                     \
    _Pragma("clang diagnostic ignored \"-Wpragma-pack\"");                \
    _Pragma("clang diagnostic ignored \"-Woverloaded-virtual\"");         \
    _Pragma("clang diagnostic ignored \"-Wreorder-ctor\"");               \
    _Pragma("clang diagnostic ignored \"-Wmissing-field-initializers\""); \
    _Pragma("clang diagnostic ignored \"-Wsign-compare\"");               \
    _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"");    \
    _Pragma("clang diagnostic ignored \"-Wdeprecated-volatile\"");
#define DISABLE_WARNINGS_POP() _Pragma("clang diagnostic pop")
#elif MSVC
// NOTE(Mathijs): it seems like there is a bug in MSVC. Calling push() and then calling pragma(warning(disable ...))
//  does not work. Hence, we disable them before the push and enable them after the push. This means that every time
//  you want to disable an additional warning, you also have to enable it again in the POP function.
#define DISABLE_WARNINGS_PUSH() __pragma(warning(disable : 4242)) __pragma(warning(disable : 4201)) __pragma(warning(disable : 4702)) __pragma(warning(push, 0))
#define DISABLE_WARNINGS_POP() __pragma(warning(pop)) __pragma(warning(default : 4242)) __pragma(warning(default : 4201)) __pragma(warning(default : 4702))

// Disable warnings in CUDA generated code
#pragma warning(disable:4505)

#else

#define DISABLE_WARNINGS_PUSH()
#define DISABLE_WARNINGS_POP()

#endif