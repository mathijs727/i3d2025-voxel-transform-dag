#ifdef _WIN32
#define NOMINMAX 1
#include <windows.h>
#endif
#define FREEIMAGE_LIB
#define FREEIMAGE_COLORORDER FREEIMAGE_COLORORDER_RGB
#include <FreeImage.h>
#include <FreeImagePlus.h>
// ^^^ Windows doesn't like it if we don't include this as the first file ^^^
#include "free_image_wrapper.h"
#include <array>
#include <cassert>

void loadImageRGBA8(
    const std::filesystem::path& filePath, std::vector<std::byte>& pixels, int& width, int& height, bool& hasTransparentPixels)
{
    assert(std::filesystem::exists(filePath));

    // Load image from disk.
    const std::string filePathString = filePath.string();
    fipImage image;
    [[maybe_unused]] const bool loadSuccess = image.load(filePathString.c_str());
    assert(image.isValid());
    assert(loadSuccess);
    width = image.getWidth();
    height = image.getHeight();
    hasTransparentPixels = image.isTransparent();

    // Convert image to RGBA8 format.
    [[maybe_unused]] const bool conversionSuccess = image.convertTo32Bits();
    assert(conversionSuccess);

    for (int y = 0; y != height; y++) {
        for (int x = 0; x != width; x++) {
            RGBQUAD rgb;
            image.getPixelColor(x, height - 1 - y, &rgb);
            pixels.push_back(std::bit_cast<std::byte>(rgb.rgbRed));
            pixels.push_back(std::bit_cast<std::byte>(rgb.rgbGreen));
            pixels.push_back(std::bit_cast<std::byte>(rgb.rgbBlue));
            pixels.push_back(std::bit_cast<std::byte>((unsigned char)255));
        }
    }
}

void loadImageHDR32(
    const std::filesystem::path& filePath, std::vector<std::byte>& pixels, int& width, int& height)
{
    assert(std::filesystem::exists(filePath));

    // Load image from disk.
    const std::string filePathString = filePath.string();
    fipImage image;
    [[maybe_unused]] const bool loadSuccess = image.load(filePathString.c_str());
    assert(image.isValid());
    assert(loadSuccess);
    width = image.getWidth();
    height = image.getHeight();

    // Convert image to RGB_F32 format.
    [[maybe_unused]] const bool conversionSuccess = image.convertToRGBF();
    assert(conversionSuccess);

    for (int y = 0; y != height; y++) {
        BYTE const* pScanLine = image.getScanLine(y);
        const unsigned scanLineWidthInBytes = image.getScanWidth();
        const size_t start = pixels.size();
        pixels.resize(start + scanLineWidthInBytes);
        std::memcpy(&pixels[start], pScanLine, scanLineWidthInBytes);
    }
}