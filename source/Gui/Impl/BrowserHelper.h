#pragma once

#include <optional>
#include <string>

#include <imgui.h>

#include <Network/NetworkResourceRawTO.h>
#include <Network/NetworkResourceTreeTO.h>

#include <Rendering/Interface/TextureData.h>

#include "Definitions.h"

class BrowserHelper
{
public:
    static auto constexpr RowHeight = 25.0f;
    static float calcFooterHeight();
    static bool ActionButton(std::string const& text);
    static void OpenButton(BrowserData const& data, BrowserLeaf const& leaf);
    static void HeartButton(BrowserData const& data, NetworkResourceRawTO const& rawTO, bool withReactionSummary);
    static NetworkResourceTreeTO createLeafTreeTO(NetworkResourceRawTO const& rawTO);

    // A preview picture fitted into the given width. Null stands for a picture that is being loaded, no value for a missing one.
    static float calcPictureHeight(float width);
    static void Picture(std::optional<TextureData> const* picture, float width);

    // Tiles are child windows, which are rendered after their parent and would cover a border drawn around them
    static void TileSelectionBorder(ImVec2 const& tileSize);
    static void PictureBadge(ImVec2 const& pos, std::string const& text, ImColor const& textColor, ImColor const& borderColor);

    // A heart with the number of reactions in the lower right corner of a picture, which works like the heart button. Returns true if it is hovered.
    static bool PictureHeart(BrowserData const& data, NetworkResourceRawTO const& rawTO, ImVec2 const& pictureMax);
};
