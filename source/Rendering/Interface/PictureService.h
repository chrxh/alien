#pragma once

#include <filesystem>
#include <string>

#include <Base/Singleton.h>

#include "PictureData.h"

class PictureService
{
    MAKE_SINGLETON(PictureService);

public:
    static auto constexpr PreviewPictureResolution = IntVector2D{600, 400};

    IntVector2D calcPreviewRenderResolution(int width) const;
    std::string createPreviewJpg(PictureData const& picture) const;

    PictureData scale(PictureData const& picture, IntVector2D const& resolution) const;

    std::string encodeJpg(PictureData const& picture) const;
    std::string encodePng(PictureData const& picture) const;

    void savePng(PictureData const& picture, std::filesystem::path const& filename) const;

private:
    PictureData brighten(PictureData const& picture, float factor) const;
};
