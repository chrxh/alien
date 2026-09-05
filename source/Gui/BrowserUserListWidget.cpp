#include "BrowserUserListWidget.h"

#include <chrono>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include <Network/NetworkService.h>

#include "AlienGui.h"
#include "BrowserData.h"
#include "BrowserGui.h"
#include "LoginController.h"
#include "StyleRepository.h"

BrowserUserListWidget _BrowserUserListWidget::create(BrowserData const& data)
{
    return BrowserUserListWidget(new _BrowserUserListWidget(data));
}

_BrowserUserListWidget::_BrowserUserListWidget(BrowserData const& data)
    : _data(data)
{}

namespace
{
    AlienGui::TextStyle getTextStyle(bool bold)
    {
        return bold ? AlienGui::TextStyle::Bold : AlienGui::TextStyle::Normal;
    }

    std::string getGpuString(std::string const& gpu)
    {
        if (gpu.substr(0, 6) == "NVIDIA") {
            return gpu.substr(7);
        }
        return gpu;
    }

    void drawOnlineSymbol()
    {
        auto counter = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        counter = (((counter % 2000) + 2000) % 2000);
        auto color = ImColor::HSV(0.0f, counter < 1000 ? toFloat(counter) / 1000.0f : 2.0f - toFloat(counter) / 1000.0f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, color.Value);
        ImGui::Text(ICON_FA_GENDERLESS);
        ImGui::PopStyleColor();
    }

    void drawLastDayOnlineSymbol()
    {
        auto color = ImColor::HSV(0.16f, 0.5f, 0.66f);
        ImGui::PushStyleColor(ImGuiCol_Text, color.Value);
        ImGui::Text(ICON_FA_GENDERLESS);
        ImGui::PopStyleColor();
    }
}

void _BrowserUserListWidget::process()
{
    ImGui::PushID("User list");
    auto& styleRepository = StyleRepository::get();
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
        | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTabBar("##Simulators", ImGuiTabBarFlags_FittingPolicyResizeDown)) {
        if (ImGui::BeginTabItem("Simulators", nullptr, ImGuiTabItemFlags_None)) {

            if (ImGui::BeginTable("Browser", 5, flags, ImVec2(-1, -1), 0.0f)) {
                ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed, scale(90.0f));
                auto isLoggedIn = NetworkService::get().getLoggedInUserName().has_value();
                ImGui::TableSetupColumn(
                    isLoggedIn ? "GPU model" : "GPU (visible if logged in)",
                    ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed,
                    styleRepository.scale(200.0f));
                ImGui::TableSetupColumn("Time spent", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(80.0f));
                ImGui::TableSetupColumn(
                    "Reactions received",
                    ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending,
                    scale(120.0f));
                ImGui::TableSetupColumn("Reactions given", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(100.0f));
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableHeadersRow();
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

                ImGuiListClipper clipper;
                clipper.Begin(_data->userTOs.size());
                while (clipper.Step()) {
                    for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                        auto const& user = _data->userTOs.at(row);

                        ImGui::PushID(row);
                        ImGui::TableNextRow(0, scale(BrowserGui::RowHeight));

                        ImGui::TableNextColumn();
                        auto isBoldFont = isLoggedIn && *NetworkService::get().getLoggedInUserName() == user.userName;

                        if (user.online) {
                            drawOnlineSymbol();
                            ImGui::SameLine();
                        } else if (user.lastDayOnline) {
                            drawLastDayOnlineSymbol();
                            ImGui::SameLine();
                        }
                        AlienGui::Text(AlienGui::TextParameters().text(user.userName).style(getTextStyle(isBoldFont)).truncate(true));

                        ImGui::TableNextColumn();
                        if (isLoggedIn && LoginController::get().shareGpuInfo()) {
                            AlienGui::Text(AlienGui::TextParameters().text(getGpuString(user.gpu)).style(getTextStyle(isBoldFont)).truncate(true));
                        }

                        ImGui::TableNextColumn();
                        if (user.timeSpent > 0) {
                            // ``timeSpent`` is the cumulative online time
                            // in seconds. Format as ``Xh`` for >= 1 hour,
                            // otherwise as ``Ym`` so short-lived users do
                            // not collapse to ``0h``.
                            auto totalSeconds = user.timeSpent;
                            std::string text;
                            if (totalSeconds >= 3600) {
                                text = StringHelper::format(static_cast<uint64_t>(totalSeconds / 3600)) + "h";
                            } else {
                                text = std::to_string(totalSeconds / 60) + "m";
                            }
                            AlienGui::Text(AlienGui::TextParameters().text(text).style(getTextStyle(isBoldFont)).truncate(true));
                        }

                        ImGui::TableNextColumn();
                        AlienGui::Text(AlienGui::TextParameters().text(std::to_string(user.starsReceived)).style(getTextStyle(isBoldFont)).rightAligned(true));

                        ImGui::TableNextColumn();
                        AlienGui::Text(AlienGui::TextParameters().text(std::to_string(user.starsGiven)).style(getTextStyle(isBoldFont)).rightAligned(true));

                        ImGui::PopID();
                    }
                }
                ImGui::EndTable();
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::PopID();
}
