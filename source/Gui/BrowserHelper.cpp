#include "BrowserHelper.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "AlienGui.h"
#include "BrowserData.h"
#include "StyleService.h"

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
    AlienGui::Tooltip("Download", false);
    if (isDownload) {
        data->onDownloadResource(leaf);
    }
}
