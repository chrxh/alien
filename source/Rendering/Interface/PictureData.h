#pragma once

#include <vector>

#include <Base/Interface/Definitions.h>

// RGB pixels ordered row-wise from top to bottom
struct PictureData
{
    static auto constexpr NumChannels = 3;

    IntVector2D resolution;
    std::vector<uint8_t> pixels;
};
