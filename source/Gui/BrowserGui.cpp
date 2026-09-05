#include "BrowserGui.h"

#include <boost/algorithm/string/regex.hpp>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "AlienGui.h"
#include "BrowserData.h"
#include "StyleRepository.h"

namespace
{
    std::vector<std::string> splitString(const std::string& str)
    {
        std::vector<std::string> tokens;
        boost::algorithm::split_regex(tokens, str, boost::regex("(\n)+"));
        return tokens;
    }
}

void BrowserGui::ShortenedText(std::string const& text, bool bold)
{
    auto substrings = splitString(text);
    if (substrings.empty()) {
        return;
    }
    auto& styleRepository = StyleRepository::get();
    auto textSize = ImGui::CalcTextSize(substrings.at(0).c_str());
    auto needDetailButton = textSize.x > ImGui::GetContentRegionAvail().x || substrings.size() > 1;
    auto cursorPos = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - styleRepository.scale(15.0f);
    if (bold) {
        ImGui::PushFont(styleRepository.getSmallBoldFont());
    }
    AlienGui::Text(substrings.at(0));
    if (bold) {
        ImGui::PopFont();
    }
    if (needDetailButton) {
        ImGui::SameLine();
        ImGui::SetCursorPosX(cursorPos);

        DetailButton();
        AlienGui::Tooltip(text.c_str(), false);
    }
}

bool BrowserGui::ActionButton(std::string const& text)
{
    ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(Const::ToolbarButtonBackgroundColor));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImU32)Const::ToolbarButtonHoveredColor);
    auto result = ImGui::Button(text.c_str());
    ImGui::PopStyleColor(2);

    return result;
}

bool BrowserGui::DetailButton()
{
    auto color = Const::DetailButtonColor;
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(color.Value.x, color.Value.y, color.Value.z, h, s, v);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(h, s, v * 0.3f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(h, s, v * 0.4f));
    auto detailClicked = AlienGui::Button("...");
    ImGui::PopStyleColor(2);
    return detailClicked;
}

void BrowserGui::DownloadButton(BrowserData const& data, BrowserLeaf const& leaf)
{
    auto isDownload = AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_DOWNLOAD));
    AlienGui::Tooltip("Download", false);
    if (isDownload) {
        data->onDownloadResource(leaf);
    }
}
