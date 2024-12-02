#include "SimulationParametersWindowPrototype.h"

#include <Fonts/IconsFontAwesome5.h>

#include "Base/StringHelper.h"
#include "EngineInterface/SimulationParametersEditService.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienImGui.h"
#include "OverlayController.h"

namespace
{
    auto constexpr MasterHeight = 100.0f;
    auto constexpr MasterMinHeight = 50.0f;

    auto constexpr DetailWidgetMinHeight = 60.0f;

    auto constexpr ExpertWidgetHeight = 100.0f;
    auto constexpr ExpertWidgetMinHeight = 60.0f;
}

SimulationParametersWindowPrototype::SimulationParametersWindowPrototype()
    : AlienWindow("Simulation parameters (new)", "windows.simulation parameters prototype", false)
{}

void SimulationParametersWindowPrototype::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    _masterHeight = GlobalSettings::get().getValue("windows.simulation parameters prototype.master height", scale(MasterHeight));
    _expertWidgetHeight = GlobalSettings::get().getValue("windows.simulation parameters prototype.addon height", scale(ExpertWidgetHeight));
}

void SimulationParametersWindowPrototype::processIntern()
{
    processToolbar();

    if (ImGui::BeginChild("##content", {0, -scale(50.0f)})) {

        updateLocations();

        auto origMasterHeight = _masterHeight;
        auto origExpertWidgetHeight = _expertWidgetHeight;

        processMasterEditor();
        processDetailEditor();
        processExpertModes();

        correctLayout(origMasterHeight, origExpertWidgetHeight);
    }
    ImGui::EndChild();

    processStatusBar();

}

void SimulationParametersWindowPrototype::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.simulation parameters prototype.master height", _masterHeight);
    GlobalSettings::get().setValue("windows.simulation parameters prototype.addon height", _expertWidgetHeight);
}

void SimulationParametersWindowPrototype::processToolbar()
{
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_FOLDER_OPEN).tooltip("Open simulation parameters from file"))) {
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_SAVE).tooltip("Save simulation parameters to file"))) {
    }

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_COPY).tooltip("Copy simulation parameters"))) {
        _copiedParameters = _simulationFacade->getSimulationParameters();
        printOverlayMessage("Simulation parameters copied");
    }

    ImGui::SameLine();
    ImGui::BeginDisabled(!_copiedParameters);
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_PASTE).tooltip("Paste simulation parameters"))) {
        _simulationFacade->setSimulationParameters(*_copiedParameters);
        _simulationFacade->setOriginalSimulationParameters(*_copiedParameters);
        printOverlayMessage("Simulation parameters pasted");
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_PLUS).secondText(ICON_FA_LAYER_GROUP).tooltip("Add parameter zone"))) {
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_PLUS).secondText(ICON_FA_SUN).tooltip("Add radiation source"))) {
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_PLUS).secondText(ICON_FA_CLONE).tooltip("Clone selected location"))) {
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_MINUS).tooltip("Delete selected location"))) {
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_CHEVRON_UP)
                                      .tooltip("Move selected location upward")
                                      .disabled(!_selectedLocationIndex.has_value() || _selectedLocationIndex.value() <= 1))) {
        onDecreaseLocationIndex();
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_CHEVRON_DOWN)
                                      .tooltip("Move selected location downward")
                                      .disabled(!_selectedLocationIndex.has_value() || _selectedLocationIndex.value() >= _locations.size() - 1))) {
        onIncreaseLocationIndex();
    }

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    AlienImGui::Separator();
}

void SimulationParametersWindowPrototype::processMasterEditor()
{
    if (ImGui::BeginChild("##masterEditor", {0, getMasterWidgetHeight()})) {

        if (_masterOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Locations").highlighted(true).defaultOpen(_masterOpen))) {
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
    auto height = getDetailWidgetHeight();
    if (ImGui::BeginChild("##detailEditor", {0, height})) {
        if (_detailOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Parameters").highlighted(true).defaultOpen(_detailOpen))) {
            if (ImGui::BeginChild("##detailChildWindow", {0, 0})) {
            }
            ImGui::EndChild();
            AlienImGui::EndTreeNode();
        }
    }
    ImGui::EndChild();
    if (_detailOpen) {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::PushID("detail");
        AlienImGui::MovableSeparator(AlienImGui::MovableSeparatorParameters().additive(false), _expertWidgetHeight);
        ImGui::PopID();
    }
}

void SimulationParametersWindowPrototype::processExpertModes()
{
    if (ImGui::BeginChild("##addon", {0, 0})) {
        if (_expertModesOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Expert modes").highlighted(true).defaultOpen(_expertModesOpen))) {
            if (ImGui::BeginChild("##detailChildWindow", {0, 0})) {
            }
            ImGui::EndChild();
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

    if (ImGui::BeginTable("Locations", 4, flags, ImVec2(-1, -1), 0)) {

        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Position", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
        ImGui::TableSetupColumn("Strength", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        ImGuiListClipper clipper;
        clipper.Begin(_locations.size());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto const& entry = _locations.at(row);

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(ImGui::GetTextLineHeightWithSpacing()));

                // name
                ImGui::TableNextColumn();
                AlienImGui::Text(entry.name);

                ImGui::SameLine();
                auto selected = _selectedLocationIndex.has_value() ? _selectedLocationIndex.value() == row : false;
                if (ImGui::Selectable(
                        "",
                        &selected,
                        ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
                        ImVec2(0, 0))) {
                    _selectedLocationIndex = selected ? std::make_optional(row) : std::nullopt;
                }


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

namespace
{
    std::variant<SimulationParametersSpot*, RadiationSource*> findLocation(SimulationParameters& parameters, int locationIndex)
    {
        for (int i = 0; i < parameters.numSpots; ++i) {
            if (parameters.spot[i].locationIndex == locationIndex) {
                return &parameters.spot[i];
            }
        }
        for (int i = 0; i < parameters.numRadiationSources; ++i) {
            if (parameters.radiationSource[i].locationIndex == locationIndex) {
                return &parameters.radiationSource[i];
            }
        }
        THROW_NOT_IMPLEMENTED();
    }

    void onDecreaseLocationIndexIntern(SimulationParameters& parameters, int locationIndex)
    {
        std::variant<SimulationParametersSpot*, RadiationSource*> zoneOrSource1 = findLocation(parameters, locationIndex);
        std::variant<SimulationParametersSpot*, RadiationSource*> zoneOrSource2 = findLocation(parameters, locationIndex - 1);
        if (std::holds_alternative<SimulationParametersSpot*>(zoneOrSource1)) {
            std::get<SimulationParametersSpot*>(zoneOrSource1)->locationIndex -= 1;
        } else {
            std::get<RadiationSource*>(zoneOrSource1)->locationIndex -= 1;
        }
        if (std::holds_alternative<SimulationParametersSpot*>(zoneOrSource2)) {
            std::get<SimulationParametersSpot*>(zoneOrSource2)->locationIndex += 1;
        } else {
            std::get<RadiationSource*>(zoneOrSource2)->locationIndex += 1;
        }
    }

    void onIncreaseLocationIndexIntern(SimulationParameters& parameters, int locationIndex)
    {
        std::variant<SimulationParametersSpot*, RadiationSource*> zoneOrSource1 = findLocation(parameters, locationIndex);
        std::variant<SimulationParametersSpot*, RadiationSource*> zoneOrSource2 = findLocation(parameters, locationIndex + 1);
        if (std::holds_alternative<SimulationParametersSpot*>(zoneOrSource1)) {
            std::get<SimulationParametersSpot*>(zoneOrSource1)->locationIndex += 1;
        } else {
            std::get<RadiationSource*>(zoneOrSource1)->locationIndex += 1;
        }
        if (std::holds_alternative<SimulationParametersSpot*>(zoneOrSource2)) {
            std::get<SimulationParametersSpot*>(zoneOrSource2)->locationIndex -= 1;
        } else {
            std::get<RadiationSource*>(zoneOrSource2)->locationIndex -= 1;
        }
    }
}

void SimulationParametersWindowPrototype::onDecreaseLocationIndex()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    onDecreaseLocationIndexIntern(parameters, _selectedLocationIndex.value());
    _simulationFacade->setSimulationParameters(parameters);

    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    onDecreaseLocationIndexIntern(origParameters, _selectedLocationIndex.value());
    _simulationFacade->setOriginalSimulationParameters(parameters);

    --_selectedLocationIndex.value();
}

void SimulationParametersWindowPrototype::onIncreaseLocationIndex()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    onIncreaseLocationIndexIntern(parameters, _selectedLocationIndex.value());
    _simulationFacade->setSimulationParameters(parameters);

    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    onIncreaseLocationIndexIntern(origParameters, _selectedLocationIndex.value());
    _simulationFacade->setOriginalSimulationParameters(parameters);

    ++_selectedLocationIndex.value();
}

void SimulationParametersWindowPrototype::updateLocations()
{
    auto parameters = _simulationFacade->getSimulationParameters();

    _locations = std::vector<Location>(1 + parameters.numSpots + parameters.numRadiationSources);
    auto strength = SimulationParametersEditService::get().getRadiationStrengths(parameters);
    auto pinnedString = strength.pinned.contains(0) ? ICON_FA_THUMBTACK " " : " ";
    _locations.at(0) = Location{"Main", LocationType::Base, "-", pinnedString + StringHelper::format(strength.values.front() * 100 + 0.05f, 1) + "%"};
    for (int i = 0; i < parameters.numSpots; ++i) {
        auto const& spot = parameters.spot[i];
        auto position = "(" + StringHelper::format(spot.posX, 0) + ", " + StringHelper::format(spot.posY, 0) + ")";
        _locations.at(spot.locationIndex) = Location{spot.name, LocationType::ParameterZone, position};
    }
    for (int i = 0; i < parameters.numRadiationSources; ++i) {
        auto const& source = parameters.radiationSource[i];
        auto position = "(" + StringHelper::format(source.posX, 0) + ", " + StringHelper::format(source.posY, 0) + ")";
        auto pinnedString = strength.pinned.contains(i + 1) ? ICON_FA_THUMBTACK " " : " ";
        _locations.at(source.locationIndex) = Location{
            source.name, LocationType::RadiationSource, position, pinnedString + StringHelper::format(strength.values.at(i + 1) * 100 + 0.05f, 1) + "%"};
    }
}

void SimulationParametersWindowPrototype::correctLayout(float origMasterHeight, float origExpertWidgetHeight)
{
    auto detailHeight = ImGui::GetWindowSize().y - getMasterWidgetRefHeight() - getExpertWidgetRefHeight();

    if (detailHeight < scale(DetailWidgetMinHeight)
        || _masterHeight < scale(MasterMinHeight)
        || _expertWidgetHeight < scale(ExpertWidgetMinHeight)) {
        _masterHeight = origMasterHeight;
        _expertWidgetHeight = origExpertWidgetHeight;
    }
}

float SimulationParametersWindowPrototype::getMasterWidgetRefHeight() const
{
    return _masterOpen ? _masterHeight : scale(25.0f);
}

float SimulationParametersWindowPrototype::getExpertWidgetRefHeight() const
{
    return _expertModesOpen ? _expertWidgetHeight : scale(47.0f);
}

float SimulationParametersWindowPrototype::getMasterWidgetHeight() const
{
    if (_masterOpen && !_detailOpen && !_expertModesOpen) {
        return std::max(scale(MasterMinHeight), ImGui::GetContentRegionAvail().y - getDetailWidgetHeight() - getExpertWidgetRefHeight());
    }
    return getMasterWidgetRefHeight();
}

float SimulationParametersWindowPrototype::getDetailWidgetHeight() const
{
    return _detailOpen ? std::max(scale(MasterMinHeight), ImGui::GetContentRegionAvail().y - getExpertWidgetRefHeight()) : scale(25.0f);
}
