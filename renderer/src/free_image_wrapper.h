#pragma once
#include <filesystem>
#include <vector>

void loadImageRGBA8(const std::filesystem::path& filePath, std::vector<std::byte>& pixels, int& width, int& height, bool& hasTransparentPixels);
void loadImageHDR32(const std::filesystem::path& filePath, std::vector<std::byte>& pixels, int& width, int& height);