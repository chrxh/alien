#include "BrowserHelper.h"

#include <cfloat>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>

#include "AlienGui.h"
#include "BrowserData.h"
#include "PictureGuiService.h"
#include "StyleService.h"

namespace
{
    auto constexpr PictureHeartMargin = 6.0f;
    auto constexpr PictureHeartPaddingX = 7.0f;
    auto constexpr PictureHeartPaddingY = 2.0f;
    auto constexpr PictureBadgePaddingX = 5.0f;
    auto constexpr PictureBadgeHeight = 15.0f;
    auto constexpr PictureBadgeRounding = 4.0f;
    auto constexpr PictureBadgeFontScale = 0.72f;
}

float BrowserHelper::calcFooterHeight()
{
    return ImGui::GetFrameHeightWithSpacing();
}

bool BrowserHelper::ActionButton(std::string const& text)
{
    ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(Const::ToolbarButtonBackgroundColor));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImU32)Const::ToolbarButtonHoveredColor);
    auto result = ImGui::Button(text.c_str());
    ImGui::PopStyleColor(2);

    return result;
}

void BrowserHelper::OpenButton(BrowserData const& data, BrowserLeaf const& leaf)
{
    auto isLocal = isLocalWorkspace(leaf.rawTO->workspaceType);
    auto clicked = AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(isLocal ? ICON_FA_FOLDER_OPEN : ICON_FA_DOWNLOAD));
    AlienGui::Tooltip(AlienGui::TooltipParameters().text(isLocal ? "Open" : "Download").delay(false));
    if (clicked) {
        data->onOpenResource(leaf);
    }
}

namespace
{
    void processReactionSummary(BrowserData const& data, NetworkResourceRawTO const& rawTO)
    {
        for (auto const& [emojiType, numLikes] : rawTO->numLikesByEmojiType) {
            if (emojiType < toInt(data->getEmojis().size())) {
                auto const& emoji = data->getEmojis().at(emojiType);
                ImGui::Image(emoji.textureId, {scale(toFloat(emoji.width) / 2.5f), scale(toFloat(emoji.height) / 2.5f)});
                ImGui::SameLine();
            }
            AlienGui::Text(std::to_string(numLikes) + "   " + data->getUserNamesToEmojiType(rawTO, emojiType));
        }
    }

    void processHeartTooltip(BrowserData const& data, NetworkResourceRawTO const& rawTO, bool withReactionSummary)
    {
        ImGui::BeginTooltip();
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextTooltipColor.Value);
        if (withReactionSummary && !rawTO->numLikesByEmojiType.empty()) {
            processReactionSummary(data, rawTO);
            AlienGui::Separator();
        }
        auto clickHint = [&] {
            if (data->getOwnEmojiType(rawTO->id).has_value()) {
                return std::string("Click: take back your reaction");
            }
            return NetworkService::get().isLoggedIn() ? std::string("Click: give a heart") : std::string("Click: give a heart, no account needed");
        }();
        AlienGui::Text(clickHint);
        AlienGui::Text("Right click: choose another reaction");
        ImGui::PopStyleColor();
        ImGui::EndTooltip();
    }
}

void BrowserHelper::HeartButton(BrowserData const& data, NetworkResourceRawTO const& rawTO, bool withReactionSummary)
{
    auto isOwnReaction = data->getOwnEmojiType(rawTO->id).has_value();
    ImGui::PushStyleColor(ImGuiCol_Text, isOwnReaction ? Const::BrowserOwnReactionButtonTextColor.Value : Const::TextDimColor.Value);
    auto clicked = ActionButton((withReactionSummary ? ICON_FA_HEART " " + std::to_string(rawTO->getTotalLikes()) : std::string(ICON_FA_HEART)) + "###heart");
    ImGui::PopStyleColor();
    auto hovered = ImGui::IsItemHovered();
    auto rightClicked = ImGui::IsItemClicked(ImGuiMouseButton_Right);

    if (hovered) {
        processHeartTooltip(data, rawTO, withReactionSummary);
    }
    if (clicked) {
        data->onToggleHeart(rawTO);
    }
    if (rightClicked) {
        data->setActivateEmojiPopup(true);
        data->setEmojiPopupTO(createLeafTreeTO(rawTO));
    }
}

NetworkResourceTreeTO BrowserHelper::createLeafTreeTO(NetworkResourceRawTO const& rawTO)
{
    auto result = std::make_shared<_NetworkResourceTreeTO>();
    result->type = rawTO->resourceType;
    result->folderNames = NetworkResourceService::get().getFolderNames(rawTO->resourceName);
    result->node = BrowserLeaf{.leafName = NetworkResourceService::get().removeFoldersFromName(rawTO->resourceName), .rawTO = rawTO};
    return result;
}

float BrowserHelper::calcPictureHeight(float width)
{
    return width * toFloat(PictureGuiService::PreviewPictureResolution.y) / toFloat(PictureGuiService::PreviewPictureResolution.x);
}

void BrowserHelper::Picture(std::optional<TextureData> const* picture, float width)
{
    auto height = calcPictureHeight(width);
    auto pos = ImGui::GetCursorScreenPos();
    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(pos, {pos.x + width, pos.y + height}, (ImU32)Const::BackgroundColor);

    if (picture != nullptr && picture->has_value()) {

        // Pictures from earlier versions can have a different aspect ratio and are fitted into the area
        auto const& texture = picture->value();
        auto pictureWidth = width;
        auto pictureHeight = width * toFloat(texture.height) / toFloat(texture.width);
        if (pictureHeight > height) {
            pictureWidth = height * toFloat(texture.width) / toFloat(texture.height);
            pictureHeight = height;
        }
        auto picturePos = ImVec2{pos.x + (width - pictureWidth) / 2, pos.y + (height - pictureHeight) / 2};
        drawList->AddImage(texture.textureId, picturePos, {picturePos.x + pictureWidth, picturePos.y + pictureHeight});
    } else {
        auto text = picture == nullptr ? std::string("loading...") : std::string("no preview");
        auto textSize = ImGui::CalcTextSize(text.c_str());
        drawList->AddText({pos.x + (width - textSize.x) / 2, pos.y + (height - textSize.y) / 2}, (ImU32)Const::TextDecentColor, text.c_str());
    }
    ImGui::Dummy({width, height});
}

void BrowserHelper::TileSelectionBorder(ImVec2 const& tileSize)
{
    auto thickness = scale(2.0f);
    auto tileMin = ImGui::GetWindowPos();
    auto tileMax = ImVec2{tileMin.x + tileSize.x, tileMin.y + tileSize.y};

    auto drawList = ImGui::GetWindowDrawList();
    drawList->PushClipRect(tileMin, tileMax, false);
    drawList->AddRect(
        {tileMin.x + thickness / 2, tileMin.y + thickness / 2},
        {tileMax.x - thickness / 2, tileMax.y - thickness / 2},
        (ImU32)Const::AccentColor,
        ImGui::GetStyle().ChildRounding,
        0,
        thickness);
    drawList->PopClipRect();
}

void BrowserHelper::PictureBadge(ImVec2 const& pos, std::string const& text, ImColor const& textColor, ImColor const& borderColor)
{
    auto font = StyleService::get().getSmallBoldFont();
    auto fontSize = font->FontSize * PictureBadgeFontScale;
    auto textSize = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, text.c_str());

    auto badgeMax = ImVec2{pos.x + textSize.x + scale(PictureBadgePaddingX) * 2, pos.y + scale(PictureBadgeHeight)};

    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(pos, badgeMax, Const::BrowserNewBadgeColor, scale(PictureBadgeRounding));
    drawList->AddRect(pos, badgeMax, borderColor, scale(PictureBadgeRounding));
    drawList->AddText(font, fontSize, {pos.x + scale(PictureBadgePaddingX), pos.y + (badgeMax.y - pos.y - textSize.y) / 2}, textColor, text.c_str());
}

bool BrowserHelper::PictureHeart(BrowserData const& data, NetworkResourceRawTO const& rawTO, ImVec2 const& pictureMax)
{
    auto icon = std::string(ICON_FA_HEART);
    auto count = " " + std::to_string(rawTO->getTotalLikes());
    auto iconSize = ImGui::CalcTextSize(icon.c_str());
    auto countSize = ImGui::CalcTextSize(count.c_str());
    auto padding = ImVec2{scale(PictureHeartPaddingX), scale(PictureHeartPaddingY)};
    auto size = ImVec2{iconSize.x + countSize.x + padding.x * 2, iconSize.y + padding.y * 2};
    auto pos = ImVec2{pictureMax.x - scale(PictureHeartMargin) - size.x, pictureMax.y - scale(PictureHeartMargin) - size.y};

    // Placed over the picture, after which the layout continues below the picture
    auto nextItemPos = ImGui::GetCursorScreenPos();
    ImGui::SetCursorScreenPos(pos);
    auto clicked = ImGui::InvisibleButton("##pictureHeart", size);
    auto hovered = ImGui::IsItemHovered();
    auto rightClicked = ImGui::IsItemClicked(ImGuiMouseButton_Right);
    ImGui::SetCursorScreenPos(nextItemPos);

    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(
        pos, {pos.x + size.x, pos.y + size.y}, hovered ? Const::BrowserPictureChipHoveredColor : Const::BrowserPictureChipColor, size.y / 2);
    auto isOwnReaction = data->getOwnEmojiType(rawTO->id).has_value();
    drawList->AddText(
        {pos.x + padding.x, pos.y + padding.y}, isOwnReaction ? Const::BrowserPictureChipOwnReactionColor : Const::BrowserPictureChipIconColor, icon.c_str());
    drawList->AddText({pos.x + padding.x + iconSize.x, pos.y + padding.y}, Const::BrowserPictureChipTextColor, count.c_str());

    if (hovered) {
        processHeartTooltip(data, rawTO, true);
    }
    if (clicked) {
        data->onToggleHeart(rawTO);
    }
    if (rightClicked) {
        data->setActivateEmojiPopup(true);
        data->setEmojiPopupTO(createLeafTreeTO(rawTO));
    }
    return hovered;
}
