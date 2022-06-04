#include "BrowserWindow.h"

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
    if (_on) {
        processActivated();
    }
}

_BrowserWindow::~_BrowserWindow()
{
}

void _BrowserWindow::processIntern()
{
    processTable();
    processStatus();
    processFilter();
    processRefresh();
}

void _BrowserWindow::processTable()
{
    auto styleRepository = StyleRepository::getInstance();
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable
        | ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody
        | ImGuiTableFlags_ScrollY;
    if (ImGui::BeginTable("Browser", 9, flags, ImVec2(0, ImGui::GetContentRegionAvail().y - styleRepository.scaleContent(90.0f)), 0.0f)) {
        ImGui::TableSetupColumn(
            "Timestamp", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Timestamp);
        ImGui::TableSetupColumn("User name", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_UserName);
        ImGui::TableSetupColumn(
            "Simulation name", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_SimulationName);
        ImGui::TableSetupColumn("Likes", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Likes);
        ImGui::TableSetupColumn("Width", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Width);
        ImGui::TableSetupColumn("Height", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Height);
        ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Size);
        ImGui::TableSetupColumn("Version", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Version);
        ImGui::TableSetupColumn(
            "Actions", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthStretch, 0.0f, RemoteSimulationDataColumnId_Actions);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        //sort our data if sort specs have been changed!
        if (ImGuiTableSortSpecs* sortSpecs = ImGui::TableGetSortSpecs())
            if (sortSpecs->SpecsDirty) {
                if (_filteredRemoteSimulationDatas.size() > 1) {
                    std::sort(_filteredRemoteSimulationDatas.begin(), _filteredRemoteSimulationDatas.end(), [&](auto const& left, auto const& right) {
                        return RemoteSimulationData::compare(&left, &right, sortSpecs) < 0;
                    });
                }
                sortSpecs->SpecsDirty = false;
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
                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(item->width));
                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(item->height));
                ImGui::TableNextColumn();
                AlienImGui::Text(StringHelper::format(item->contentSize / 1024) + " KB");
                ImGui::TableNextColumn();
                AlienImGui::Text(item->version);
                ImGui::TableNextColumn();
                if (ImGui::Button(ICON_FA_FOLDER_OPEN)) {
                    onOpenSimulation(item->id);
                }
                ImGui::SameLine();
                ImGui::BeginDisabled();
                if (ImGui::Button(ICON_FA_THUMBS_UP)) {
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::BeginDisabled();
                if (ImGui::Button(ICON_FA_TRASH)) {
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
        if (auto userName = _networkController->getLoggedInUserName()) {
            AlienImGui::Text("Logged as " + *userName + " @ " + _networkController->getServerAddress() + ".");
        } else {
            AlienImGui::Text("Not logged in to " + _networkController->getServerAddress() + ".");
        }
        ImGui::PopStyleColor();
        ImGui::PopFont();
    }
    ImGui::EndChild();
}

void _BrowserWindow::processFilter()
{
    if (AlienImGui::InputText(AlienImGui::InputTextParameters().name("Filter"), _filter)) {
        _filteredRemoteSimulationDatas.clear();
        for (auto const& remoteSimulationData : _remoteSimulationDatas) {
            auto match = false;
            if (remoteSimulationData.description.find(_filter) != std::string::npos) {
                match = true;
            }
            if (std::to_string(remoteSimulationData.width).find(_filter) != std::string::npos) {
                match = true;
            }
            if (std::to_string(remoteSimulationData.height).find(_filter) != std::string::npos) {
                match = true;
            }
            if (remoteSimulationData.simName.find(_filter) != std::string::npos) {
                match = true;
            }
            if (remoteSimulationData.timestamp.find(_filter) != std::string::npos) {
                match = true;
            }
            if (remoteSimulationData.userName.find(_filter) != std::string::npos) {
                match = true;
            }
            if (remoteSimulationData.version.find(_filter) != std::string::npos) {
                match = true;
            }
            if (match) {
                _filteredRemoteSimulationDatas.emplace_back(remoteSimulationData);
            }
        }
    }
}

void _BrowserWindow::processRefresh()
{
    if (AlienImGui::Button("Refresh")) {
        processActivated();
    }
}

void _BrowserWindow::processActivated()
{
    _remoteSimulationDatas = _networkController->getRemoteSimulationDataList();
    _filteredRemoteSimulationDatas = _remoteSimulationDatas;
}

void _BrowserWindow::onOpenSimulation(std::string const& id)
{
    std::string content, settings, symbolMap;
    _networkController->downloadSimulation(content, settings, symbolMap, id);

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
