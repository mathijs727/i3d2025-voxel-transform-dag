#pragma once
#include "binary_reader.h"
#include "binary_writer.h"
#include "cuda_error_check.h"
#include "cuda_math.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>

struct CUDATexture {
    cudaArray* cuArray { nullptr };
    cudaTextureObject_t cuTexture;

    void free()
    {
        if (cuArray) {
            cudaDestroyTextureObject(cuTexture);
            cudaFreeArray(cuArray);
        }
    }

    constexpr bool operator==(const CUDATexture&) const noexcept = default;
};

struct Image {
    enum class PixelType {
        sRGBA8,
        HDR32
    };

    std::vector<std::byte> pixels;
    PixelType pixelType;
    int width = 0, height = 0;
    bool hasTransparentPixels;

public:
    Image() = default;
    Image(const std::filesystem::path& image);

    bool is_valid() const;
    void writeTo(BinaryWriter& writer) const;
    void readFrom(BinaryReader& reader);

    size_t numPixels() const;
    float3 getPixelLinear(size_t i) const;
    float3 getPixelsRGB(size_t i) const;

    CUDATexture createCudaTexture() const;
};