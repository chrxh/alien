#include "BrowserHelper.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>

#include "AlienGui.h"
#include "BrowserData.h"
#include "StyleService.h"

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

void BrowserHelper::DownloadButton(BrowserData const& data, BrowserLeaf const& leaf)
{
    auto isDownload = AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_DOWNLOAD));
    AlienGui::Tooltip(AlienGui::TooltipParameters().text("Download").delay(false));
    if (isDownload) {
        data->onDownloadResource(leaf);
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
