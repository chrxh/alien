#pragma once

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
    // Preview picture of the currently rendered simulation, ready for upload
    std::optional<std::string> createSimulationPreviewJpg();

    // Preview picture showing the creatures of a genome as a collage, ready for upload
    std::optional<std::string> createGenomePreviewJpg(std::vector<PreviewDesc> const& previews);

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
};
