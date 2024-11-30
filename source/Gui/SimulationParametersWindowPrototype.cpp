#include "SimulationParametersWindowPrototype.h"

#include <Fonts/IconsFontAwesome5.h>

#include "Base/StringHelper.h"
#include "EngineInterface/SimulationParametersEditService.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienImGui.h"
#include "OverlayController.h"

namespace
{
    auto constexpr MasterEditorHeight = 100.0f;
    auto constexpr AddonHeight = 100.0f;

}

SimulationParametersWindowPrototype::SimulationParametersWindowPrototype()
    : AlienWindow("Simulation parameters Prototype", "windows.simulation parameters prototype", false)
{}

void SimulationParametersWindowPrototype::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    _masterHeight = GlobalSettings::get().getValue("windows.simulation parameters prototype.master height", scale(MasterEditorHeight));
    _addonHeight = GlobalSettings::get().getValue("windows.simulation parameters prototype.addon height", scale(AddonHeight));
}

void SimulationParametersWindowPrototype::processIntern()
{
    processToolbar();

    if (ImGui::BeginChild("##content", {0, -scale(50.0f)})) {

        processMasterEditor();
        processDetailEditor();
        processExpertModes();
    }
    ImGui::EndChild();

    processStatusBar();

    correctLayout();
}

void SimulationParametersWindowPrototype::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.simulation parameters prototype.master height", _masterHeight);
    GlobalSettings::get().setValue("windows.simulation parameters prototype.addon height", _addonHeight);
}

void SimulationParametersWindowPrototype::processToolbar()
{
    if (AlienImGui::ToolbarButton(ICON_FA_FOLDER_OPEN)) {
    }
    AlienImGui::Tooltip("Open simulation parameters from file");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
    }
    AlienImGui::Tooltip("Save simulation parameters to file");

    for (int i = 0; i < 6; ++i) {
        ImGui::SameLine();
        if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
        }
        AlienImGui::Tooltip("Save simulation parameters to file");
    }

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_COPY)) {
        _copiedParameters = _simulationFacade->getSimulationParameters();
        printOverlayMessage("Simulation parameters copied");
    }
    AlienImGui::Tooltip("Copy simulation parameters");

    ImGui::SameLine();
    ImGui::BeginDisabled(!_copiedParameters);
    if (AlienImGui::ToolbarButton(ICON_FA_PASTE)) {
        _simulationFacade->setSimulationParameters(*_copiedParameters);
        _simulationFacade->setOriginalSimulationParameters(*_copiedParameters);
        printOverlayMessage("Simulation parameters pasted");
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Paste simulation parameters");

    AlienImGui::Separator();
}

void SimulationParametersWindowPrototype::processMasterEditor()
{
    if (ImGui::BeginChild("##masterEditor", {0, getMasterWidgetHeight()})) {

        if (_masterOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Location").highlighted(true).defaultOpen(_masterOpen))) {
            processLocationTable();

            AlienImGui::EndTreeNode();
        }
    }
    ImGui::EndChild();
    if (_masterOpen) {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::PushID("master");
        AlienImGui::MovableSeparator(AlienImGui::MovableSeparatorParameters(), _masterHeight);
        ImGui::PopID();
    }
}

void SimulationParametersWindowPrototype::processDetailEditor()
{
    if (ImGui::BeginChild("##detailEditor", {0, getDetailWidgetHeight()})) {
        if (_detailOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Parameters").highlighted(true).defaultOpen(_detailOpen))) {
            ImGui::Button("Test2", ImGui::GetContentRegionAvail());
            //if (ImGui::BeginChild("##detailChildWindow", {0, scale(_detailHeight)})) {
            //}
            //ImGui::EndChild();
            AlienImGui::EndTreeNode();
        }
    }
    ImGui::EndChild();
    if (_detailOpen) {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::PushID("detail");
        AlienImGui::MovableSeparator(AlienImGui::MovableSeparatorParameters().additive(false), _addonHeight);
        ImGui::PopID();
    }
}

void SimulationParametersWindowPrototype::processExpertModes()
{
    if (ImGui::BeginChild("##addon", {0, 0})) {
        if (_expertModesOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Expert modes").highlighted(true).defaultOpen(_expertModesOpen))) {
            ImGui::Button("Test", ImGui::GetContentRegionAvail());
            //if (ImGui::BeginChild("##detailChildWindow", {0, scale(_detailHeight)})) {
            //}
            //ImGui::EndChild();
            AlienImGui::EndTreeNode();
        }
    }
    ImGui::EndChild();
}

void SimulationParametersWindowPrototype::processStatusBar()
{
    std::vector<std::string> statusItems;
    statusItems.emplace_back("CTRL + click on a slider to type in a precise value");

    AlienImGui::StatusBar(statusItems);
}

void SimulationParametersWindowPrototype::processLocationTable()
{
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
        | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTable("Location", 4, flags, ImVec2(-1, -1), 0)) {

        auto locations = generateLocations();

        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Position", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
        ImGui::TableSetupColumn("Strength", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        ImGuiListClipper clipper;
        clipper.Begin(locations.size());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto const& entry = locations.at(row);

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(ImGui::GetTextLineHeightWithSpacing()));

                // name
                ImGui::TableNextColumn();
                AlienImGui::Text(entry.name);

                // type
                ImGui::TableNextColumn();
                if (entry.type == LocationType::Base) {
                    AlienImGui::Text("Base parameters");
                } else if (entry.type == LocationType::ParameterZone) {
                    AlienImGui::Text("Zone");
                } else if (entry.type == LocationType::RadiationSource) {
                    AlienImGui::Text("Radiation");
                }

                // position
                ImGui::TableNextColumn();
                AlienImGui::Text(entry.position);

                // strength
                ImGui::TableNextColumn();
                AlienImGui::Text(entry.strength);

                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
}

auto SimulationParametersWindowPrototype::generateLocations() const -> std::vector<Location>
{
    auto parameters = _simulationFacade->getSimulationParameters();

    std::vector<Location> result;
    auto strength = SimulationParametersEditService::get().getRadiationStrengths(parameters);
    auto pinnedString = strength.pinned.contains(0) ? ICON_FA_THUMBTACK " ": " ";
    result.emplace_back("Main", LocationType::Base, "-", pinnedString + StringHelper::format(strength.values.front() * 100 + 0.5f, 0) + "%");
    for (int i = 0; i < parameters.numSpots; ++i) {
        auto const& spot = parameters.spots[i];
        auto position = "(" + StringHelper::format(spot.posX, 0) + ", " + StringHelper::format(spot.posY, 0) + ")";
        result.emplace_back(spot.name, LocationType::ParameterZone, position);
    }
    for (int i = 0; i < parameters.numRadiationSources; ++i) {
        auto const& source = parameters.radiationSources[i];
        auto position = "(" + StringHelper::format(source.posX, 0) + ", " + StringHelper::format(source.posY, 0) + ")";
        auto pinnedString = strength.pinned.contains(i + 1) ? ICON_FA_THUMBTACK " " : " ";
        result.emplace_back(
            source.name, LocationType::RadiationSource, position, pinnedString + StringHelper::format(strength.values.at(i + 1) * 100 + 0.5f, 0) + "%");
    }

    return result;
}

void SimulationParametersWindowPrototype::correctLayout()
{
}

float SimulationParametersWindowPrototype::getMasterWidgetRefHeight() const
{
    return _masterOpen ? _masterHeight : scale(25.0f);
}

float SimulationParametersWindowPrototype::getAddonWidgetRefHeight() const
{
    return _expertModesOpen ? _addonHeight : scale(47.0f);
}

float SimulationParametersWindowPrototype::getMasterWidgetHeight() const
{
    if (_masterOpen && !_detailOpen && !_expertModesOpen) {
        return ImGui::GetContentRegionAvail().y - getDetailWidgetHeight() - getAddonWidgetRefHeight();
    }
    return getMasterWidgetRefHeight();
}

float SimulationParametersWindowPrototype::getDetailWidgetHeight() const
{
    return _detailOpen ? ImGui::GetContentRegionAvail().y - getAddonWidgetRefHeight() : scale(25.0f);
}

float SimulationParametersWindowPrototype::getAddonWidgetHeight() const
{
    if (!_masterOpen && !_detailOpen) {
        return ImGui::GetContentRegionAvail().y;
    }
    return getAddonWidgetRefHeight();
}
