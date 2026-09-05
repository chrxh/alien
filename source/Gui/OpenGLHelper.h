#pragma once

#include <filesystem>
#include <string>

#include "Definitions.h"

class OpenGLHelper
{
public:
    // Returns id
    static TextureData loadTexture(std::filesystem::path const& filename);

    static TextureData loadTextureFromMemory(std::string const& encodedImage);
};
