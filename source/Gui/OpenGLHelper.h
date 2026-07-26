#pragma once

#include <filesystem>

#include "Definitions.h"

class OpenGLHelper
{
public:
    // Returns id
    static TextureData loadTexture(std::filesystem::path const& filename);
};
