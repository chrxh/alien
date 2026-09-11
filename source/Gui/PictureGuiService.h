#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/PreviewDesc.h>

#include "PictureData.h"

class PictureGuiService
{
    MAKE_SINGLETON(PictureGuiService);

public:
    static auto constexpr PreviewPictureResolution = IntVector2D{600, 350};

    // Preview picture of the currently rendered simulation, ready for upload
    std::optional<std::string> createSimulationPreviewJpg();

    // Preview picture showing the creatures of a genome as a collage, ready for upload
    std::optional<std::string> createGenomePreviewJpg(std::vector<PreviewDesc> const& previews);

    PictureData scale(PictureData const& picture, IntVector2D const& resolution);
    PictureData brighten(PictureData const& picture, float factor);

    std::string encodeJpg(PictureData const& picture);

    void savePng(PictureData const& picture, std::filesystem::path const& filename);
};
