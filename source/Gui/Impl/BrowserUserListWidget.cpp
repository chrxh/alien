#include "BrowserUserListWidget.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <optional>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Network/NetworkService.h>

#include "AlienGui.h"
#include "BrowserData.h"
#include "BrowserHelper.h"
#include "HelpStrings.h"
#include "LoginController.h"
#include "LoginDialog.h"
#include "StyleService.h"

namespace
{
    auto constexpr GpuColumn = 1;
    auto constexpr GpuColumnName = "GPU model";
    auto constexpr HiddenGpuBarMinLength = 64.0f;
    auto constexpr HiddenGpuBarLengthVariation = 87;
    auto constexpr HiddenGpuBarOffsetY = 4.0f;
    auto constexpr HiddenGpuBarHeight = 8.0f;
}

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

    AlienGui::TabItemParameters getSimulatorsTabParameters(std::optional<int> const& numActiveSimulators)
    {
        auto result = AlienGui::TabItemParameters().name("Simulators");
        if (numActiveSimulators) {
            result.badge("  " + std::to_string(*numActiveSimulators) + " active")
                .badgeColor(ImColor::HSV(0.0f, 0.4f, 0.9f))
                .tooltip(Const::BrowserActiveSimulatorsTooltip);
        }
        return result;
    }

    void drawLastDayOnlineSymbol()
    {
        auto color = ImColor::HSV(0.16f, 0.5f, 0.66f);
        ImGui::PushStyleColor(ImGuiCol_Text, color.Value);
        ImGui::Text(ICON_FA_GENDERLESS);
        ImGui::PopStyleColor();
    }

    void drawGpuHeaderHiddenIcon()
    {
        if (!ImGui::TableSetColumnIndex(GpuColumn)) {
            return;
        }
        auto pos = ImGui::GetCursorScreenPos();
        auto offsetX = ImGui::CalcTextSize(GpuColumnName).x + ImGui::CalcTextSize("  ").x;
        AlienGui::SmallIcon(AlienGui::SmallIconParameters().icon(ICON_FA_EYE_SLASH).pos({pos.x + offsetX, pos.y}).color(Const::TextDimColor));
    }

    std::string getOwnGpuString()
    {
        return getGpuString(_SimulationFacade::get()->getGpuName());
    }

    void processOwnGpu()
    {
        AlienGui::Text(AlienGui::TextParameters().text(getOwnGpuString()).style(AlienGui::TextStyle::Bold).truncate(true));
    }

    void processSharedGpu(std::string const& gpu)
    {
        if (gpu.empty()) {
            AlienGui::Text(AlienGui::TextParameters().text("not shared").style(AlienGui::TextStyle::Decent).truncate(true));
        } else {
            AlienGui::Text(AlienGui::TextParameters().text(getGpuString(gpu)).truncate(true));
        }
    }

    void processHiddenGpu(std::string const& userName, bool isLoggedIn)
    {
        auto pos = ImGui::GetCursorScreenPos();
        auto width = ImGui::GetContentRegionAvail().x;
        ImGui::SetNextItemAllowOverlap();
        ImGui::InvisibleButton("##hiddenGpu", {std::max(width, 1.0f), ImGui::GetTextLineHeight()});

        auto length = std::min(width, scale(HiddenGpuBarMinLength + toFloat(std::hash<std::string>{}(userName) % HiddenGpuBarLengthVariation)));
        auto barTop = pos.y + scale(HiddenGpuBarOffsetY);
        auto barHeight = scale(HiddenGpuBarHeight);
        ImGui::GetWindowDrawList()->AddRectFilled(
            {pos.x + scale(1.0f), barTop}, {pos.x + length, barTop + barHeight}, Const::BrowserHiddenGpuBarColor, barHeight / 2);

        AlienGui::Tooltip(AlienGui::TooltipParameters().text(isLoggedIn ? Const::BrowserGpuHiddenNotSharedTooltip : Const::BrowserGpuHiddenLoggedOutTooltip));
    }

    std::string getGpuHintText(bool isLoggedIn)
    {
        return isLoggedIn ? ICON_FA_LOCK "  Share yours to see" : ICON_FA_LOCK "  Log in to see";
    }

    struct GpuHintPlacement
    {
        int row = 0;
        ImVec2 pos;
    };

    // Centered in the visible part of the column and submitted in the row containing its lower edge, so it never enlarges a row
    GpuHintPlacement calcGpuHintPlacement(int row, int numRows, bool isLoggedIn)
    {
        auto const& style = ImGui::GetStyle();
        auto rowHeight = scale(BrowserHelper::RowHeight);
        auto cellPos = ImGui::GetCursorScreenPos();
        auto rowsTop = cellPos.y - style.CellPadding.y - toFloat(row) * rowHeight;
        auto rowsBottom = rowsTop + toFloat(numRows) * rowHeight;

        auto drawList = ImGui::GetWindowDrawList();
        auto visibleTop = std::max(drawList->GetClipRectMin().y, rowsTop);
        auto visibleBottom = std::min(drawList->GetClipRectMax().y, rowsBottom);

        auto textSize = ImGui::CalcTextSize(getGpuHintText(isLoggedIn).c_str());
        auto width = textSize.x + style.FramePadding.x * 2;
        auto height = textSize.y + style.FramePadding.y * 2;
        auto bottom = std::min((visibleTop + visibleBottom + height) / 2, rowsBottom);

        return GpuHintPlacement{
            .row = std::clamp(toInt(std::ceil((bottom - rowsTop) / rowHeight)) - 1, 0, numRows - 1),
            .pos = {cellPos.x + std::max(0.0f, (ImGui::GetContentRegionAvail().x - width) / 2), bottom - height},
        };
    }

    // Stays in the background so that the login banner and the account chip remain the prominent login buttons
    bool processLoginHint()
    {
        auto const& style = ImGui::GetStyle();
        auto text = getGpuHintText(false);
        auto pos = ImGui::GetCursorScreenPos();
        auto textSize = ImGui::CalcTextSize(text.c_str());
        auto size = ImVec2(textSize.x + style.FramePadding.x * 2, textSize.y + style.FramePadding.y * 2);
        auto result = ImGui::InvisibleButton("##loginHint", size);
        auto hovered = ImGui::IsItemHovered();

        auto drawList = ImGui::GetWindowDrawList();
        drawList->AddRectFilled(
            pos, {pos.x + size.x, pos.y + size.y}, hovered ? Const::ActionButtonHoveredColor : Const::ActionButtonBackgroundColor, style.FrameRounding);
        drawList->AddRect(pos, {pos.x + size.x, pos.y + size.y}, Const::BrowserLoginHintFrameColor, style.FrameRounding, 0, scale(1.0f));
        drawList->AddText(
            {pos.x + style.FramePadding.x, pos.y + style.FramePadding.y},
            hovered ? Const::ActionButtonHighlightedTextColor : Const::TextDimColor,
            text.c_str());

        AlienGui::Tooltip(AlienGui::TooltipParameters().text(Const::BrowserGpuHiddenLoggedOutTooltip));
        return result;
    }

    void processGpuHint(ImVec2 const& pos, bool isLoggedIn)
    {
        ImGui::SetCursorScreenPos(pos);
        if (!isLoggedIn) {
            if (processLoginHint()) {
                LoginDialog::get().open();
            }
            return;
        }
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters()
                                       .buttonText(getGpuHintText(true))
                                       .highlighted(true)
                                       .frame(true)
                                       .transparentBackground(false)
                                       .tooltip(Const::BrowserShareGpuTooltip + getOwnGpuString()))) {
            LoginController::get().setShareGpuInfo(true);
        }
    }
}

void _BrowserUserListWidget::process()
{
    ImGui::PushID("User list");
    auto& styleService = StyleService::get();
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
        | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    auto selectedTabIndex = 0;
    AlienGui::TabBar(
        AlienGui::TabBarParameters().id("##Simulators").style(AlienGui::TabBarStyle::Group),
        {getSimulatorsTabParameters(_data->getNumActiveSimulators())},
        selectedTabIndex);

    if (ImGui::BeginTable("Browser", 5, flags, ImVec2(-1, -1), 0.0f)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed, scale(90.0f));
        auto isLoggedIn = NetworkService::get().getLoggedInUserName().has_value();
        auto isGpuVisible = isLoggedIn && LoginController::get().shareGpuInfo();
        ImGui::TableSetupColumn(GpuColumnName, ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, styleService.scale(200.0f));
        ImGui::TableSetupColumn("Time spent", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, styleService.scale(80.0f));
        ImGui::TableSetupColumn(
            "Reactions received",
            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending,
            scale(120.0f));
        ImGui::TableSetupColumn("Reactions given", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, styleService.scale(100.0f));
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();
        if (!isGpuVisible) {
            drawGpuHeaderHiddenIcon();
        }
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

        auto numRows = toInt(_data->getUserTOs().size());
        std::optional<GpuHintPlacement> gpuHintPlacement;
        ImGuiListClipper clipper;
        clipper.Begin(numRows);
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto const& user = _data->getUserTOs().at(row);

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(BrowserHelper::RowHeight));

                ImGui::TableNextColumn();
                auto isOwnRow = isLoggedIn && *NetworkService::get().getLoggedInUserName() == user.userName;

                if (user.online) {
                    drawOnlineSymbol();
                    ImGui::SameLine();
                } else if (user.lastDayOnline) {
                    drawLastDayOnlineSymbol();
                    ImGui::SameLine();
                }
                AlienGui::Text(AlienGui::TextParameters().text(user.userName).style(getTextStyle(isOwnRow)).truncate(true));

                ImGui::TableNextColumn();
                if (!isGpuVisible && !gpuHintPlacement) {
                    gpuHintPlacement = calcGpuHintPlacement(row, numRows, isLoggedIn);
                }
                if (isOwnRow) {
                    processOwnGpu();
                } else if (isGpuVisible) {
                    processSharedGpu(user.gpu);
                } else {
                    processHiddenGpu(user.userName, isLoggedIn);
                }
                if (!isGpuVisible && gpuHintPlacement->row == row) {
                    processGpuHint(gpuHintPlacement->pos, isLoggedIn);
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
                    AlienGui::Text(AlienGui::TextParameters().text(text).style(getTextStyle(isOwnRow)).truncate(true));
                }

                ImGui::TableNextColumn();
                AlienGui::Text(AlienGui::TextParameters().text(std::to_string(user.starsReceived)).style(getTextStyle(isOwnRow)).rightAligned(true));

                ImGui::TableNextColumn();
                AlienGui::Text(AlienGui::TextParameters().text(std::to_string(user.starsGiven)).style(getTextStyle(isOwnRow)).rightAligned(true));

                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
    ImGui::PopID();
}
