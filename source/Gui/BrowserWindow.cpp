#include "BrowserWindow.h"

#include <boost/algorithm/string/join.hpp>
#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/StringHelper.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/SimulationController.h"

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "StyleRepository.h"
#include "RemoteSimulationDataParser.h"
#include "NetworkController.h"
#include "StatisticsWindow.h"
#include "Viewport.h"
#include "TemporalControlWindow.h"
#include "MessageDialog.h"

_BrowserWindow::_BrowserWindow(
    SimulationController const& simController,
    NetworkController const& networkController,
    StatisticsWindow const& statisticsWindow,
    Viewport const& viewport,
    TemporalControlWindow const& temporalControlWindow)
    : _AlienWindow("Browser", "browser.network", false)
    , _simController(simController)
    , _networkController(networkController)
    , _statisticsWindow(statisticsWindow)
    , _viewport(viewport)
    , _temporalControlWindow(temporalControlWindow)
{
    _on = false;
}

_BrowserWindow::~_BrowserWindow()
{}

void _BrowserWindow::onRefresh()
{
    try {
        if (!_networkController->getRemoteSimulationDataList(_remoteSimulationDatas)) {
            MessageDialog::getInstance().show("Error", "Failed to retrieve browser data.");
        }
        _filteredRemoteSimulationDatas = _remoteSimulationDatas;

        if (_networkController->getLoggedInUserName()) {
            std::vector<std::string> likedIds;
            if (!_networkController->getLikedSimulationIdList(likedIds)) {
                MessageDialog::getInstance().show("Error", "Failed to retrieve browser data.");
            }
            _likedIds = std::unordered_set<std::string>(likedIds.begin(), likedIds.end());
        } else {
            _likedIds.clear();
        }

        sortTable();
    }
    catch(std::exception const& e) {
        MessageDialog::getInstance().show("Error", e.what());
    }
}

void _BrowserWindow::processIntern()
{
    processTable();
    processStatus();
    processFilter();
    processRefreshButton();
    if(_scheduleRefresh) {
        onRefresh();
        _scheduleRefresh = false;
    }
}

void _BrowserWindow::processTable()
{
    auto styleRepository = StyleRepository::getInstance();
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable
        | ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody
        | ImGuiTableFlags_ScrollY;
    if (ImGui::BeginTable("Browser", 11, flags, ImVec2(0, ImGui::GetContentRegionAvail().y - styleRepository.scaleContent(90.0f)), 0.0f)) {
        ImGui::TableSetupColumn(
            "Timestamp", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Timestamp);
        ImGui::TableSetupColumn(
            "User name",
            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed,
            styleRepository.scaleContent(120.0f),
            RemoteSimulationDataColumnId_UserName);
        ImGui::TableSetupColumn(
            "Simulation name",
            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed,
            styleRepository.scaleContent(135.0f),
            RemoteSimulationDataColumnId_SimulationName);
        ImGui::TableSetupColumn(
            "Description",
            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed,
            styleRepository.scaleContent(120.0f),
            RemoteSimulationDataColumnId_Description);
        ImGui::TableSetupColumn("Likes", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Likes);
        ImGui::TableSetupColumn("Width", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Width);
        ImGui::TableSetupColumn("Height", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Height);
        ImGui::TableSetupColumn("Particles", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Particles);
        ImGui::TableSetupColumn("File size", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_FileSize);
        ImGui::TableSetupColumn("Version", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Version);
        ImGui::TableSetupColumn(
            "Actions", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthStretch, 0.0f, RemoteSimulationDataColumnId_Actions);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        //sort our data if sort specs have been changed!
        if (ImGuiTableSortSpecs* sortSpecs = ImGui::TableGetSortSpecs()) {
            if (sortSpecs->SpecsDirty || _scheduleSort) {
                if (_filteredRemoteSimulationDatas.size() > 1) {
                    std::sort(_filteredRemoteSimulationDatas.begin(), _filteredRemoteSimulationDatas.end(), [&](auto const& left, auto const& right) {
                        return RemoteSimulationData::compare(&left, &right, sortSpecs) < 0;
                    });
                }
                sortSpecs->SpecsDirty = false;
            }
        }

        ImGuiListClipper clipper;
        clipper.Begin(_filteredRemoteSimulationDatas.size());
        while (clipper.Step())
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                RemoteSimulationData* item = &_filteredRemoteSimulationDatas[row];

//                auto isItemSelected = _selectionIds.find(item->id) != _selectionIds.end();

                ImGui::PushID(row);
                ImGui::TableNextRow();

                ImGui::TableNextColumn();

/*
                ImGuiSelectableFlags selectableFlags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap;
                if (ImGui::Selectable(item->timestamp.c_str(), isItemSelected, selectableFlags, ImVec2(0, 0.0f))) {
                    _selectionIds = {item->id};
                }
*/
                AlienImGui::Text(item->timestamp);
                ImGui::TableNextColumn();
                AlienImGui::Text(item->userName);
                ImGui::TableNextColumn();
                AlienImGui::Text(item->simName);
                ImGui::TableNextColumn();
                auto textSize = ImGui::CalcTextSize(item->description.c_str());
                auto needDetailButton = textSize.x > ImGui::GetContentRegionAvailWidth();
                auto cursorPos = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvailWidth() - styleRepository.scaleContent(15.0f);
                AlienImGui::Text(item->description);
                if (needDetailButton) {
                    ImGui::SameLine();
                    ImGui::SetCursorPosX(cursorPos);

                    processDetailButton();
                    AlienImGui::Tooltip(item->description.c_str());
                }
                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(item->likes));
                if(item->likes > 0) {
                    ImGui::SameLine();
                    processDetailButton();
                    AlienImGui::Tooltip([&] { return getUserLikes(item->id); });
                }
                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(item->width));
                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(item->height));
                ImGui::TableNextColumn();
                AlienImGui::Text(StringHelper::format(item->particles / 1000) + " K");
                ImGui::TableNextColumn();
                AlienImGui::Text(StringHelper::format(item->contentSize / 1024) + " KB");
                ImGui::TableNextColumn();
                AlienImGui::Text(item->version);
                ImGui::TableNextColumn();
                if (ImGui::Button(ICON_FA_FOLDER_OPEN)) {
                    onOpenSimulation(item->id);
                }
                ImGui::SameLine();
                ImGui::BeginDisabled(!_networkController->getLoggedInUserName());
                auto liked = isLiked(item->id);
                if (liked) {
                    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::LikeTextColor);
                }
                if (ImGui::Button(ICON_FA_THUMBS_UP)) {
                    onToggleLike(*item);
                }
                if (liked) {
                    ImGui::PopStyleColor(1);
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::BeginDisabled(item->userName != _networkController->getLoggedInUserName().value_or(""));
                if (ImGui::Button(ICON_FA_TRASH)) {
                    onDeleteSimulation(item->id);
                }
                ImGui::EndDisabled();
                ImGui::PopID();
            }
        ImGui::EndTable();
    }
}

void _BrowserWindow::processStatus()
{
    auto styleRepository = StyleRepository::getInstance();

    if (ImGui::BeginChild("##", ImVec2(0, styleRepository.scaleContent(30.0f)), true, ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::LogMessageColor);
        std::string statusText;
        if (auto userName = _networkController->getLoggedInUserName()) {
            statusText = "Logged in as " + *userName + " @ " + _networkController->getServerAddress() + ": ";
        } else {
            statusText = "Not logged in to " + _networkController->getServerAddress() + ": ";
        }
        statusText += std::to_string(_remoteSimulationDatas.size()) + " simulations found";
        AlienImGui::Text(statusText);
        ImGui::PopStyleColor();
        ImGui::PopFont();
    }
    ImGui::EndChild();
}

void _BrowserWindow::processFilter()
{
    if (AlienImGui::InputText(AlienImGui::InputTextParameters().name("Filter"), _filter)) {
        _filteredRemoteSimulationDatas.clear();
        for (auto const& entry : _remoteSimulationDatas) {
            if (entry.matchWithFilter(_filter)) {
                _filteredRemoteSimulationDatas.emplace_back(entry);
            }
        }
    }
}

void _BrowserWindow::processRefreshButton()
{
    if (AlienImGui::Button("Refresh")) {
        onRefresh();
    }
}

bool _BrowserWindow::processDetailButton()
{
    auto color = Const::DetailButtonColor;
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(color.Value.x, color.Value.y, color.Value.z, h, s, v);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(h, s, v * 0.3f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(h, s, v * 0.4f));
    auto detailClicked = AlienImGui::Button("...");
    ImGui::PopStyleColor(2);
    return detailClicked;
}

void _BrowserWindow::processActivated()
{
    onRefresh();
}

void _BrowserWindow::sortTable()
{
    _scheduleSort = true;
}

void _BrowserWindow::onOpenSimulation(std::string const& id)
{
    std::string content, settings, symbolMap;
    if (!_networkController->downloadSimulation(content, settings, symbolMap, id)) {
        MessageDialog::getInstance().show("Error", "Failed to download simulation.");
        return;
    }

    DeserializedSimulation deserializedSim;
    Serializer::deserializeSimulationFromStrings(deserializedSim, content, settings, symbolMap);

    _simController->closeSimulation();
    _statisticsWindow->reset();

    _simController->newSimulation(deserializedSim.timestep, deserializedSim.settings, deserializedSim.symbolMap);
    _simController->setClusteredSimulationData(deserializedSim.content);
    _viewport->setCenterInWorldPos(
        {toFloat(deserializedSim.settings.generalSettings.worldSizeX) / 2, toFloat(deserializedSim.settings.generalSettings.worldSizeY) / 2});
    _viewport->setZoomFactor(2.0f);
    _temporalControlWindow->onSnapshot();
}

void _BrowserWindow::onDeleteSimulation(std::string const& id)
{
    if (!_networkController->deleteSimulation(id)) {
        MessageDialog::getInstance().show("Error", "Failed to delete simulation.");
        return;
    }
    _scheduleRefresh = true;
}

void _BrowserWindow::onToggleLike(RemoteSimulationData& entry)
{
    auto findResult = _likedIds.find(entry.id);
    if (findResult != _likedIds.end()) {
        _likedIds.erase(findResult);
        --entry.likes;
    } else {
        _likedIds.insert(entry.id);
        ++entry.likes;
    }
    _userLikesByIdCache.erase(entry.id); //invalidate cache entry
    _networkController->toggleLikeSimulation(entry.id);
    sortTable();
}

bool _BrowserWindow::isLiked(std::string const& id)
{
    return _likedIds.find(id) != _likedIds.end();
}

std::string _BrowserWindow::getUserLikes(std::string const& id)
{
    std::set<std::string> userLikes;

    auto findResult = _userLikesByIdCache.find(id);
    if (findResult != _userLikesByIdCache.end()) {
        userLikes = findResult->second;
    } else {
        _networkController->getUserLikesForSimulation(userLikes, id);
        _userLikesByIdCache.emplace(id, userLikes);
    }

    return boost::algorithm::join(userLikes, ", ");
}
