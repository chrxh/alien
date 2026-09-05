#pragma once

#include <filesystem>
#include <string>

#include <Base/Singleton.h>

#include "PictureData.h"

class PictureGuiService
{
    MAKE_SINGLETON(PictureGuiService);

public:
    PictureData scale(PictureData const& picture, IntVector2D const& resolution);
    PictureData brighten(PictureData const& picture, float factor);

    std::string encodeJpg(PictureData const& picture);

    void savePng(PictureData const& picture, std::filesystem::path const& filename);
};
