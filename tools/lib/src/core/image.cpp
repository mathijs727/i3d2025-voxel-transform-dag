#include "voxcom/core/image.h"
#include <cassert>
#include <random>

#include <voxcom/utility/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <spdlog/spdlog.h>
#if _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif
#define FREEIMAGE_COLORORDER FREEIMAGE_COLORORDER_RGB
#include <FreeImage.h>
#include <FreeImagePlus.h>
#include <assimp/texture.h>
DISABLE_WARNINGS_POP()

namespace voxcom {

void fillFromFreeImage(Image2D<voxcom::RGB>& outImage, fipImage& freeImage)
{
    // Convert image to RGBA8 format.
    [[maybe_unused]] const bool conversionSuccess = freeImage.convertTo32Bits();
    assert(conversionSuccess);

    outImage.resolution = { freeImage.getWidth(), freeImage.getHeight() };
    outImage.pixels.resize(outImage.resolution.x * outImage.resolution.y);

    for (int y = 0; y != outImage.resolution.y; y++) {
        for (int x = 0; x != outImage.resolution.x; x++) {
            RGBQUAD rgb;
            freeImage.getPixelColor(x, outImage.resolution.y - 1 - y, &rgb);
            outImage.pixels[y * outImage.resolution.x + x] = { .r = rgb.rgbRed, .g = rgb.rgbGreen, .b = rgb.rgbBlue };
        }
    }
}

template <>
Image2D<voxcom::RGB>::Image2D(const std::filesystem::path& filePath)
{
    assert(std::filesystem::exists(filePath));

    // Load image from disk.
    const std::string filePathString = filePath.string();
    fipImage image;
    image.load(filePathString.c_str());
    fillFromFreeImage(*this, image);
}

template <>
Image2D<voxcom::RGB>::Image2D(const aiTexture* pAssimpTexture)
{
    if (pAssimpTexture->mHeight == 0) {
        // Texture needs to be decoded.
        const std::string formatHint = pAssimpTexture->achFormatHint; // Convert to std::string to get comparison operator.
        assert(formatHint == "png");

        FREE_IMAGE_FORMAT freeImageFormat = FREE_IMAGE_FORMAT::FIF_UNKNOWN;
        if (formatHint == "png")
            freeImageFormat = FREE_IMAGE_FORMAT::FIF_PNG;
        else if (formatHint == "jpg")
            freeImageFormat = FREE_IMAGE_FORMAT::FIF_JPEG;
        else if (formatHint == "bmp")
            freeImageFormat = FREE_IMAGE_FORMAT::FIF_BMP;
        else
            spdlog::error("Unknown assimp embedded texture encoding");

        fipMemoryIO memory { (BYTE*)pAssimpTexture->pcData, pAssimpTexture->mWidth };
        fipImage image;
        image.loadFromMemory(freeImageFormat, memory);

        fillFromFreeImage(*this, image);
    } else {
        spdlog::error("Loading unformatted embedded textures is not supported yet");
    }
}

}
