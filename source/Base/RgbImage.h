#pragma once

#include <cstdint>
#include <vector>

struct RgbImage
{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;  // 3 bytes per pixel, row by row
};
