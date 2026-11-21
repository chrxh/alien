#include "BrowserUserListWidget.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <ranges>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>

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
    auto constexpr ActivityColumn = 0;
    auto constexpr ActivityColumnName = ICON_FA_CLOCK;
    auto constexpr ActivityColumnWidth = 30.0f;
    auto constexpr GpuColumn = 2;
    auto constexpr GpuColumnName = "GPU model";
    auto constexpr HiddenGpuBarMinLength = 64.0f;
    auto constexpr HiddenGpuBarLengthVariation = 87;
    auto constexpr HiddenGpuBarOffsetY = 4.0f;
    auto constexpr HiddenGpuBarHeight = 8.0f;
    auto constexpr SimulatorCountBadgeSpacing = 16.0f;
    auto constexpr OnlineMarkerRadius = 4.0f;
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

    // Centered below the icon in the column header
    void drawOnlineMarker(ImColor const& color)
    {
        auto pos = ImGui::GetCursorScreenPos();
        auto center = ImVec2(pos.x + AlienGui::CalcSmallIconWidth(ActivityColumnName) / 2, pos.y + ImGui::GetTextLineHeight() / 2);
        ImGui::GetWindowDrawList()->AddCircleFilled(center, scale(OnlineMarkerRadius), color);
    }

    AlienGui::TabBadgeParameters getSimulatorCountBadge(std::string const& text, ImColor const& color)
    {
        return AlienGui::TabBadgeParameters().text(text).color(color).markerRadius(OnlineMarkerRadius);
    }

    AlienGui::TabItemParameters getSimulatorsTabParameters(std::optional<ActiveSimulatorCountsTO> const& counts)
    {
        auto result = AlienGui::TabItemParameters().name("Simulators");
        if (!counts) {
            return result;
        }
        std::vector badges{getSimulatorCountBadge(std::to_string(counts->numActive) + " active", Const::BrowserActiveSimulatorsColor)};
        if (counts->numLastDay > 0) {
            badges.emplace_back(getSimulatorCountBadge(std::to_string(counts->numLastDay) + " last day", Const::BrowserLastDaySimulatorsColor));
        }
        return result.badges(badges).badgeSpacing(SimulatorCountBadgeSpacing).tooltip(Const::BrowserSimulatorCountsTooltip);
    }

    void processActivityHeader()
    {
        auto pos = ImGui::GetCursorScreenPos();
        ImGui::TableHeader("##activity");
        AlienGui::SmallIcon(AlienGui::SmallIconParameters().icon(ActivityColumnName).pos({pos.x, pos.y}));
        AlienGui::Tooltip(AlienGui::TooltipParameters().text(Const::BrowserLastActivityTooltip));
    }

    // Like ImGui::TableHeadersRow, with a small icon and a tooltip for the activity column
    void processHeaderRow()
    {
        auto rowHeight = ImGui::TableGetHeaderRowHeight();
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers, rowHeight);
        auto rowTop = ImGui::GetCursorScreenPos().y;
        auto columnCount = ImGui::TableGetColumnCount();
        for (auto column : std::views::iota(0, columnCount)) {
            if (!ImGui::TableSetColumnIndex(column)) {
                continue;
            }
            ImGui::PushID(column);
            if (column == ActivityColumn) {
                processActivityHeader();
            } else {
                ImGui::TableHeader(ImGui::TableGetColumnName(column));
            }
            ImGui::PopID();
        }

        // The empty area right of the last column opens the context menu, as in ImGui::TableHeadersRow
        auto mouseY = ImGui::GetMousePos().y;
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Right) && ImGui::TableGetHoveredColumn() == columnCount && mouseY >= rowTop
            && mouseY < rowTop + rowHeight) {
            ImGui::TableOpenContextMenu(columnCount);
        }
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
        return BrowserHelper::getShortGpuName(_SimulationFacade::get()->getGpuName());
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
            AlienGui::Text(AlienGui::TextParameters().text(BrowserHelper::getShortGpuName(gpu)).truncate(true));
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
        auto visibleLeft = std::max(drawList->GetClipRectMin().x, cellPos.x);
        auto visibleRight = std::min(drawList->GetClipRectMax().x, cellPos.x + ImGui::GetContentRegionAvail().x);

        auto textSize = ImGui::CalcTextSize(getGpuHintText(isLoggedIn).c_str());
        auto width = textSize.x + style.FramePadding.x * 2;
        auto height = textSize.y + style.FramePadding.y * 2;
        auto bottom = std::min((visibleTop + visibleBottom + height) / 2, rowsBottom);

        return GpuHintPlacement{
            .row = std::clamp(toInt(std::ceil((bottom - rowsTop) / rowHeight)) - 1, 0, numRows - 1),
            .pos = {visibleLeft + std::max(0.0f, (visibleRight - visibleLeft - width) / 2), bottom - height},
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
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable
        | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    auto selectedTabIndex = 0;
    AlienGui::TabBar(
        AlienGui::TabBarParameters().id("##Simulators").style(AlienGui::TabBarStyle::Group),
        {getSimulatorsTabParameters(_data->getActiveSimulatorCounts())},
        selectedTabIndex);

    if (ImGui::BeginTable("Simulators", 6, flags, ImVec2(-1, -1), 0.0f)) {
        ImGui::TableSetupColumn(
            ActivityColumnName,
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_DefaultSort
                | ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_NoSortAscending,
            scale(ActivityColumnWidth),
            UserColumnId_LastActivity);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, scale(90.0f), UserColumnId_UserName);
        auto isLoggedIn = NetworkService::get().getLoggedInUserName().has_value();
        auto isGpuVisible = isLoggedIn && LoginController::get().shareGpuInfo();
        ImGui::TableSetupColumn(
            GpuColumnName,
            ImGuiTableColumnFlags_WidthFixed | (isGpuVisible ? ImGuiTableColumnFlags_None : ImGuiTableColumnFlags_NoSort),
            styleService.scale(200.0f),
            UserColumnId_Gpu);
        ImGui::TableSetupColumn(
            "Time spent", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, styleService.scale(80.0f), UserColumnId_TimeSpent);
        ImGui::TableSetupColumn(
            "Reactions received", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, scale(120.0f), UserColumnId_ReactionsReceived);
        ImGui::TableSetupColumn(
            "Reactions given",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending,
            styleService.scale(100.0f),
            UserColumnId_ReactionsGiven);
        ImGui::TableSetupScrollFreeze(0, 1);
        processHeaderRow();
        if (!isGpuVisible) {
            drawGpuHeaderHiddenIcon();
        }
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

        processSortSpecs(isGpuVisible);

        auto numRows = toInt(_data->getUserTOs().size());
        std::optional<GpuHintPlacement> gpuHintPlacement;
        ImGuiListClipper clipper;
        clipper.Begin(numRows);
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto const& user = _data->getUserTOs().at(row);

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(BrowserHelper::RowHeight));

                auto isOwnRow = isLoggedIn && *NetworkService::get().getLoggedInUserName() == user.userName;

                ImGui::TableNextColumn();
                if (user.online) {
                    drawOnlineMarker(Const::BrowserActiveSimulatorsColor);
                } else if (user.lastDayOnline) {
                    drawOnlineMarker(Const::BrowserLastDaySimulatorsColor);
                }

                ImGui::TableNextColumn();
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

void _BrowserUserListWidget::processSortSpecs(bool isGpuVisible)
{
    auto sortSpecs = ImGui::TableGetSortSpecs();
    if (sortSpecs == nullptr) {
        return;
    }

    // An order by hidden GPU models would reveal them
    if (!isGpuVisible && sortSpecs->SpecsCount > 0 && sortSpecs->Specs->ColumnUserID == UserColumnId_Gpu) {
        ImGui::TableSetColumnSortDirection(ActivityColumn, ImGuiSortDirection_Descending, false);
        sortSpecs = ImGui::TableGetSortSpecs();
    }
    if (!sortSpecs->SpecsDirty) {
        return;
    }
    _data->setUserSortSpecs(std::vector(sortSpecs->Specs, sortSpecs->Specs + sortSpecs->SpecsCount));
    sortSpecs->SpecsDirty = false;
}
