#pragma once

#include <imgui.h>

struct TextureData
{
    ImTextureID textureId;
    int width;
    int height;
};

enum class TextureFormat
{
    Rgba,
    Bgra,
};

enum class TextureFilter
{
    Smooth,
    Nearest,
};
