#pragma once

#include <filesystem>

#include "Definitions.h"

class OpenGLHelper
{
public:
    //returns id
    static TextureData loadTexture(std::filesystem::path const& filename);
};
