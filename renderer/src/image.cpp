#include "image.h"
#include "color_utils.h"
#include "free_image_wrapper.h"

Image::Image(const std::filesystem::path& filePath)
{
    if (filePath.extension() == ".hdr" || filePath.extension() == ".exr") {
        loadImageHDR32(filePath, pixels, width, height);
        pixelType = PixelType::HDR32;
        hasTransparentPixels = false;
    } else {
        loadImageRGBA8(filePath, pixels, width, height, hasTransparentPixels);
        pixelType = PixelType::sRGBA8;
    }
}

bool Image::is_valid() const
{
    return width > 0 && height > 0;
}

void Image::writeTo(BinaryWriter& writer) const
{
    writer.write(pixels);
    writer.write(pixelType);
    writer.write(width);
    writer.write(height);
    writer.write(hasTransparentPixels);
}

void Image::readFrom(BinaryReader& reader)
{
    reader.read(pixels);
    reader.read(pixelType);
    reader.read(width);
    reader.read(height);
    reader.read(hasTransparentPixels);
}

size_t Image::numPixels() const
{
    return (size_t)width * (size_t)height;
}

float3 Image::getPixelLinear(size_t i) const
{
    return ColorUtils::accurateSRGBToLinear(getPixelsRGB(i));
}

float3 Image::getPixelsRGB(size_t i) const
{
    if (pixelType == PixelType::sRGBA8) {
        std::byte const* pPixel = &pixels[i * 4];
        uint32_t srgb8;
        std::memcpy(&srgb8, pPixel, sizeof(srgb8));

        return ColorUtils::rgb888_to_float3(srgb8);
    } else {
        checkAlways(false);
        return make_float3(0.0f);
    }
}

CUDATexture Image::createCudaTexture() const
{
    cudaTextureDesc textureDesc {};
    textureDesc.addressMode[0] = textureDesc.addressMode[1] = textureDesc.addressMode[2] = cudaTextureAddressMode::cudaAddressModeClamp;
    textureDesc.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
    textureDesc.normalizedCoords = true;
    textureDesc.maxAnisotropy = 0;
    textureDesc.mipmapFilterMode = cudaTextureFilterMode::cudaFilterModePoint;
    textureDesc.mipmapLevelBias = 0.0f;
    textureDesc.minMipmapLevelClamp = 0.0f;
    textureDesc.maxMipmapLevelClamp = 0.0f;

    // FreeImage and CUDA don't agree on pixel layout (RGB vs BGR).
    CUDATexture out;
    if (pixelType == PixelType::sRGBA8) {
        const auto channelDesc = cudaCreateChannelDesc<uchar4>();
        textureDesc.sRGB = true;
        textureDesc.readMode = cudaTextureReadMode::cudaReadModeNormalizedFloat;

        constexpr size_t pixelSizeInBytes = 4 * sizeof(unsigned char);
        CUDA_CHECKED_CALL cudaMallocArray(&out.cuArray, &channelDesc, width, height);
        cudaMemcpy2DToArray(out.cuArray, 0, 0, pixels.data(), width * pixelSizeInBytes, width * pixelSizeInBytes, height, cudaMemcpyHostToDevice);
    } else {
        const auto channelDesc = cudaCreateChannelDesc<float4>();
        textureDesc.sRGB = false;
        textureDesc.readMode = cudaTextureReadMode::cudaReadModeElementType;

        // Convert from RGB_F32 to RGBA_32 because CUDA does not support RGB_F32
        const auto inPixels = std::span((const float3*)pixels.data(), width * height);
        std::vector<float4> outPixels;
        for (const float3 inPixel : inPixels) {
            outPixels.push_back(make_float4(inPixel.x, inPixel.y, inPixel.z, 0.0f));
        }

        constexpr size_t pixelSizeInBytes = 4 * sizeof(float);
        CUDA_CHECKED_CALL cudaMallocArray(&out.cuArray, &channelDesc, width, height);
        cudaMemcpy2DToArray(out.cuArray, 0, 0, outPixels.data(), width * pixelSizeInBytes, width * pixelSizeInBytes, height, cudaMemcpyHostToDevice);
    }

    cudaResourceDesc resourceDesc {};
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = out.cuArray;

    CUDA_CHECKED_CALL cudaCreateTextureObject(&out.cuTexture, &resourceDesc, &textureDesc, nullptr);

    return out;
}
