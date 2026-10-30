#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <imgui.h>

#include <Base/Singleton.h>

#include <Data/PreviewDesc.h>

#include <Rendering/Interface/PictureData.h>

class PictureGuiService
{
    MAKE_SINGLETON(PictureGuiService);

public:
    static auto constexpr PreviewPictureResolution = IntVector2D{600, 400};

    // Preview picture of the currently rendered simulation, ready for upload
    std::optional<std::string> createSimulationPreviewJpg();

    // Preview picture showing the creatures of a genome as a collage, ready for upload
    std::optional<std::string> createGenomePreviewJpg(std::vector<PreviewDesc> const& previews);

    PictureData scale(PictureData const& picture, IntVector2D const& resolution);
    PictureData brighten(PictureData const& picture, float factor);

    // Text in the style of the overlay messages. It is rendered on black and on white, so that its transparency can be recovered when it is
    // drawn over a picture.
    struct OverlayText
    {
        PictureData onBlack;
        PictureData onWhite;
    };
    OverlayText renderOverlayText(std::string const& text, float fontSize);
    void drawOverlayText(PictureData& picture, OverlayText const& text, IntVector2D const& lowerRightCorner);
    void drawOverlayText(ImDrawList* drawList, std::string const& text, float fontSize, RealVector2D const& lowerRightCorner);

    std::string encodeJpg(PictureData const& picture);
    std::string encodePng(PictureData const& picture);

    void savePng(PictureData const& picture, std::filesystem::path const& filename);
};
