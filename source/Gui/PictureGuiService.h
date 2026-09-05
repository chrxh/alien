#pragma once

#include <filesystem>

#include <Base/Singleton.h>

#include "PictureData.h"

class PictureGuiService
{
    MAKE_SINGLETON(PictureGuiService);

public:
    PictureData scale(PictureData const& picture, IntVector2D const& resolution);
    PictureData brighten(PictureData const& picture, float factor);

    void savePng(PictureData const& picture, std::filesystem::path const& filename);
    void saveJpg(PictureData const& picture, std::filesystem::path const& filename);
};
