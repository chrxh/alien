#include "SimulationParametersMainWindow.h"

#include <ImFileDialog.h>
#include <Fonts/IconsFontAwesome5.h>

#include "Base/StringHelper.h"
#include "EngineInterface/SimulationParametersEditService.h"
#include "EngineInterface/SimulationFacade.h"
#include "PersisterInterface/SerializerService.h"

#include "AlienImGui.h"
#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "LocationController.h"
#include "LocationHelper.h"
#include "OverlayController.h"
#include "SimulationParametersSourceWidgets.h"
#include "SimulationParametersZoneWidgets.h"
#include "Viewport.h"

namespace
{
    auto constexpr MasterHeight = 130.0f;
    auto constexpr MasterMinHeight = 50.0f;
    auto constexpr MasterRowHeight = 25.0f;

    auto constexpr DetailWidgetMinHeight = 0.0f;

    auto constexpr ExpertWidgetHeight = 130.0f;
    auto constexpr ExpertWidgetMinHeight = 60.0f;
}

SimulationParametersMainWindow::SimulationParametersMainWindow()
    : AlienWindow("Simulation parameters", "windows.simulation parameters", false)
{}

void SimulationParametersMainWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    _masterWidgetOpen = GlobalSettings::get().getValue("windows.simulation parameters.master widget.open", _masterWidgetOpen);
    _detailWidgetOpen = GlobalSettings::get().getValue("windows.simulation parameters.detail widget.open", _detailWidgetOpen);
    _expertWidgetOpen = GlobalSettings::get().getValue("windows.simulation parameters.expert widget.open", _expertWidgetOpen);
    _masterWidgetHeight = GlobalSettings::get().getValue("windows.simulation parameters.master widget.height", scale(MasterHeight));
    _expertWidgetHeight = GlobalSettings::get().getValue("windows.simulation parameters.expert widget height", scale(ExpertWidgetHeight));

    auto baseWidgets = std::make_shared<_SimulationParametersBaseWidgets>();
    baseWidgets->init(_simulationFacade);
    _baseWidgets = baseWidgets;

    auto zoneWidgets = std::make_shared<_SimulationParametersZoneWidgets>();
    zoneWidgets->init(_simulationFacade, 0);
    _zoneWidgets = zoneWidgets;


    auto sourceWidgets = std::make_shared<_SimulationParametersSourceWidgets>();
    sourceWidgets->init(_simulationFacade, 0);
    _sourceWidgets = sourceWidgets;
}

void SimulationParametersMainWindow::processIntern()
{
    if (!_sessionId.has_value() || _sessionId.value() != _simulationFacade->getSessionId()) {
        _selectedLocationIndex = 0;
    }

    processToolbar();

    if (ImGui::BeginChild("##content", {0, -scale(50.0f)})) {

        updateLocations();

        auto origMasterHeight = _masterWidgetHeight;
        auto origExpertWidgetHeight = _expertWidgetHeight;

        processMasterWidget();
        processDetailWidget();
        processExpertWidget();

        correctLayout(origMasterHeight, origExpertWidgetHeight);
    }
    ImGui::EndChild();

    processStatusBar();

    _sessionId = _simulationFacade->getSessionId();
}

void SimulationParametersMainWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.simulation parameters.master widget.open", _masterWidgetOpen);
    GlobalSettings::get().setValue("windows.simulation parameters.detail widget.open", _detailWidgetOpen);
    GlobalSettings::get().setValue("windows.simulation parameters.expert widget.open", _expertWidgetOpen);
    GlobalSettings::get().setValue("windows.simulation parameters.master widget.height", _masterWidgetHeight);
    GlobalSettings::get().setValue("windows.simulation parameters.expert widget height", _expertWidgetHeight);
}

void SimulationParametersMainWindow::processToolbar()
{
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_FOLDER_OPEN).tooltip("Open simulation parameters from file"))) {
        onOpenParameters();
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_SAVE).tooltip("Save simulation parameters to file"))) {
        onSaveParameters();
    }

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_COPY).tooltip("Copy simulation parameters to clipboard"))) {
        _copiedParameters = _simulationFacade->getSimulationParameters();
        printOverlayMessage("Simulation parameters copied");
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(
            AlienImGui::ToolbarButtonParameters().text(ICON_FA_PASTE).tooltip("Paste simulation parameters from clipboard").disabled(!_copiedParameters))) {
        _simulationFacade->setSimulationParameters(*_copiedParameters);
        _simulationFacade->setOriginalSimulationParameters(*_copiedParameters);
        printOverlayMessage("Simulation parameters pasted");
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_PASTE)
                                      .secondText(ICON_FA_UNDO)
                                      .secondTextOffset(RealVector2D{32.0f, 28.0f})
                                      .secondTextScale(0.3f)
                                      .tooltip("Replace reference values by values from the clipboard")
                                      .disabled(!_copiedParameters))) {
        auto parameters = _simulationFacade->getSimulationParameters();
        if (_copiedParameters->numZones == parameters.numZones && _copiedParameters->numRadiationSources == parameters.numRadiationSources) {
            _simulationFacade->setOriginalSimulationParameters(*_copiedParameters);
            printOverlayMessage("Reference simulation parameters replaced");
        } else {
            GenericMessageDialog::get().information(
                "Error", "The number of zones and radiation sources of the current simulation parameters must match with those from the clipboard.");
        }
    }

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_PLUS)
                                      .secondText(ICON_FA_LAYER_GROUP)
                                      .tooltip("Add parameter zone"))) {
        onAddZone();
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_PLUS)
                                      .secondText(ICON_FA_SUN)
                                      .tooltip("Add radiation source"))) {
        onAddSource();
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_PLUS)
                                      .secondText(ICON_FA_CLONE)
                                      .disabled(_selectedLocationIndex == 0)
                                      .tooltip("Clone selected zone/radiation source")))
        {
        onCloneLocation();
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_MINUS)
                                      .disabled(_selectedLocationIndex == 0)
                                      .tooltip("Delete selected zone/radiation source"))) {
        onDeleteLocation();
    }

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_CHEVRON_UP)
                                      .disabled(_selectedLocationIndex <= 1)
                                      .tooltip("Move selected zone/radiation source upward"))) {
        onDecreaseLocationIndex();
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_CHEVRON_DOWN)
                                      .tooltip("Move selected zone/radiation source downward")
                                      .disabled(_selectedLocationIndex >= _locations.size() - 1 || _selectedLocationIndex == 0))) {
        onIncreaseLocationIndex();
    }

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(
            AlienImGui::ToolbarButtonParameters().text(ICON_FA_EXTERNAL_LINK_SQUARE_ALT).tooltip("Open parameters for selected zone/radiation source in a new window"))) {
        onOpenInLocationWindow();
    }

    AlienImGui::Separator();
}

void SimulationParametersMainWindow::processMasterWidget()
{
    if (ImGui::BeginChild("##master", {0, getMasterWidgetHeight()})) {

        if (_masterWidgetOpen = AlienImGui::BeginTreeNode(
                AlienImGui::TreeNodeParameters().text("Overview").rank(AlienImGui::TreeNodeRank::High).defaultOpen(_masterWidgetOpen))) {
            ImGui::Spacing();
            if (ImGui::BeginChild("##master2", {0, -ImGui::GetStyle().FramePadding.y})) {
                processLocationTable();
            }
            ImGui::EndChild();
        }
        AlienImGui::EndTreeNode();
    }
    ImGui::EndChild();

    if (_masterWidgetOpen && (_detailWidgetOpen || _expertWidgetOpen)) {
        ImGui::PushID("master");
        AlienImGui::MovableSeparator(AlienImGui::MovableSeparatorParameters(), _masterWidgetHeight);
        ImGui::PopID();
    }
}

void SimulationParametersMainWindow::processDetailWidget()
{
    auto height = getDetailWidgetHeight();
    if (ImGui::BeginChild("##detail", {0, height})) {
        auto title = _filter.empty() ? "Parameters" : "Parameters (filtered)";
        if (_detailWidgetOpen =
                AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text((std::string(title) + "###parameters").c_str()).rank(AlienImGui::TreeNodeRank::High).defaultOpen(_detailWidgetOpen))) {
            ImGui::Spacing();
            AlienImGui::SetFilterText(_filter);
            if (ImGui::BeginChild(
                    "##detail2", {0, -ImGui::GetStyle().FramePadding.y - scale(33.0f)}, ImGuiChildFlags_Border, ImGuiWindowFlags_HorizontalScrollbar)) {
                auto type = _locations.at(_selectedLocationIndex).type;
                if (type == LocationType::Base) {
                    _baseWidgets->process();
                } else if (type == LocationType::ParameterZone) {
                    _zoneWidgets->setLocationIndex(_selectedLocationIndex);
                    _zoneWidgets->process();
                } else if (type == LocationType::RadiationSource) {
                    _sourceWidgets->setLocationIndex(_selectedLocationIndex);
                    _sourceWidgets->process();
                }
            }
            ImGui::EndChild();
            AlienImGui::ResetFilterText();

            ImGui::Spacing();
            AlienImGui::InputFilter(_filter);
        }
        AlienImGui::EndTreeNode();
    }
    ImGui::EndChild();

    if (_detailWidgetOpen && _expertWidgetOpen) {
        ImGui::PushID("detail");
        AlienImGui::MovableSeparator(AlienImGui::MovableSeparatorParameters().additive(false), _expertWidgetHeight);
        ImGui::PopID();
    }
}

void SimulationParametersMainWindow::processExpertWidget()
{
    if (ImGui::BeginChild("##expert", {0, 0})) {
        if (_expertWidgetOpen = AlienImGui::BeginTreeNode(
                AlienImGui::TreeNodeParameters().text("Expert settings").rank(AlienImGui::TreeNodeRank::High).defaultOpen(_expertWidgetOpen))) {
            if (ImGui::BeginChild("##expert2", {0, 0}, ImGuiChildFlags_Border, ImGuiWindowFlags_HorizontalScrollbar)) {
                    processExpertSettings();
            }
            ImGui::EndChild();
        }
        AlienImGui::EndTreeNode();
    }
    ImGui::EndChild();
}

void SimulationParametersMainWindow::processStatusBar()
{
    std::vector<std::string> statusItems;
    statusItems.emplace_back("CTRL + click on a slider to type in a precise value");

    AlienImGui::StatusBar(statusItems);
}

void SimulationParametersMainWindow::processLocationTable()
{
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
        | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTable("Locations", 4, flags, ImVec2(-1, -1), 0)) {

        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Position", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(115.0f));
        ImGui::TableSetupColumn("Strength", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        ImGuiListClipper clipper;
        clipper.Begin(_locations.size());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto const& entry = _locations.at(row);

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(MasterRowHeight));

                // name
                ImGui::TableNextColumn();
                auto selected = _selectedLocationIndex == row;
                if (ImGui::Selectable(
                        "",
                        &selected,
                        ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
                        ImVec2(0, scale(MasterRowHeight) - ImGui::GetStyle().FramePadding.y))) {
                    _selectedLocationIndex = row;
                }
                ImGui::SameLine();
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
                if (row > 0) {
                    if (AlienImGui::ActionButton(
                            AlienImGui::ActionButtonParameters().buttonText(ICON_FA_SEARCH))) {
                        onCenterLocation(row);
                    }
                    ImGui::SameLine();
                }
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

void SimulationParametersMainWindow::processExpertSettings()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origFeatures = _simulationFacade->getOriginalSimulationParameters().features;
    auto lastFeatures = parameters.features;

    AlienImGui::Checkbox(
        AlienImGui::CheckboxParameters()
            .name("Advanced absorption control")
            .textWidth(0)
            .defaultValue(origFeatures.advancedAbsorptionControl)
            .tooltip("These settings offer extended possibilities for controlling the absorption of energy particles by cells."),
        parameters.features.advancedAbsorptionControl);
    AlienImGui::Checkbox(
        AlienImGui::CheckboxParameters()
            .name("Advanced attacker control")
            .textWidth(0)
            .defaultValue(origFeatures.advancedAttackerControl)
            .tooltip("It contains further settings that influence how much energy can be obtained from an attack by attacker cells."),
        parameters.features.advancedAttackerControl);
    AlienImGui::Checkbox(
        AlienImGui::CheckboxParameters()
            .name("Cell age limiter")
            .textWidth(0)
            .defaultValue(origFeatures.cellAgeLimiter)
            .tooltip("It enables additional possibilities to control the maximal cell age."),
        parameters.features.cellAgeLimiter);
    AlienImGui::Checkbox(
        AlienImGui::CheckboxParameters()
            .name("Cell color transition rules")
            .textWidth(0)
            .defaultValue(origFeatures.cellColorTransitionRules)
            .tooltip("This can be used to define color transitions for cells depending on their age."),
        parameters.features.cellColorTransitionRules);
    AlienImGui::Checkbox(
        AlienImGui::CheckboxParameters()
            .name("Cell glow")
            .textWidth(0)
            .defaultValue(origFeatures.cellGlow)
            .tooltip("It enables an additional rendering step that makes the cells glow."),
        parameters.features.cellGlow);
    AlienImGui::Checkbox(
        AlienImGui::CheckboxParameters()
            .name("Customize neuron mutations")
            .textWidth(0)
            .defaultValue(origFeatures.customizeNeuronMutations),
        parameters.features.customizeNeuronMutations);
    AlienImGui::Checkbox(
        AlienImGui::CheckboxParameters()
            .name("External energy control")
            .textWidth(0)
            .defaultValue(origFeatures.externalEnergyControl)
            .tooltip("These settings are used to add and control an external energy source. Its energy can be gradually transferred to the constructor cells in the "
                     "simulation. Vice versa, the energy from radiation and dying cells can also be transferred back to the external source."),
        parameters.features.externalEnergyControl);
    AlienImGui::Checkbox(
        AlienImGui::CheckboxParameters()
            .name("Genome complexity measurement")
            .textWidth(0)
            .defaultValue(origFeatures.genomeComplexityMeasurement)
            .tooltip("Parameters for the calculation of genome complexity are activated here. This genome complexity can be used for 'Advanced "
                     "absorption control' "
                     "and 'Advanced attacker control' to favor more complex genomes in natural selection. If it is deactivated, default values are "
                     "used that simply take the genome size into account."),
        parameters.features.genomeComplexityMeasurement);
    AlienImGui::Checkbox(
        AlienImGui::CheckboxParameters()
            .name("Legacy behavior")
            .textWidth(0)
            .defaultValue(origFeatures.legacyModes)
            .tooltip("It contains features for compatibility with older versions."),
        parameters.features.legacyModes);

    if (parameters.features != lastFeatures) {
        _simulationFacade->setSimulationParameters(parameters);
    }
}

void SimulationParametersMainWindow::onOpenParameters()
{
    GenericFileDialog::get().showOpenFileDialog(
        "Open simulation parameters", "Simulation parameters (*.parameters){.parameters},.*", _fileDialogPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _fileDialogPath = firstFilenameCopy.remove_filename().string();

            SimulationParameters parameters;
            if (!SerializerService::get().deserializeSimulationParametersFromFile(parameters, firstFilename.string())) {
                GenericMessageDialog::get().information("Open simulation parameters", "The selected file could not be opened.");
            } else {
                _simulationFacade->setSimulationParameters(parameters);
                _simulationFacade->setOriginalSimulationParameters(parameters);
            }
        });
}

void SimulationParametersMainWindow::onSaveParameters()
{
    GenericFileDialog::get().showSaveFileDialog(
        "Save simulation parameters", "Simulation parameters (*.parameters){.parameters},.*", _fileDialogPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _fileDialogPath = firstFilenameCopy.remove_filename().string();

            auto parameters = _simulationFacade->getSimulationParameters();
            if (!SerializerService::get().serializeSimulationParametersToFile(firstFilename.string(), parameters)) {
                GenericMessageDialog::get().information("Save simulation parameters", "The selected file could not be saved.");
            }
        });
}

void SimulationParametersMainWindow::onAddZone()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    if (!checkNumZones(parameters)) {
        return;
    }

    ++_selectedLocationIndex;
    LocationHelper::adaptLocationIndex(parameters, _selectedLocationIndex, 1);
    LocationHelper::adaptLocationIndex(origParameters, _selectedLocationIndex, 1);

    auto worldSize = _simulationFacade->getWorldSize();

    SimulationParametersZone zone;
    StringHelper::copy(zone.name, sizeof(zone.name), LocationHelper::generateZoneName(parameters));
    zone.locationIndex = _selectedLocationIndex;
    zone.posX = toFloat(worldSize.x / 2);
    zone.posY = toFloat(worldSize.y / 2);
    auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    zone.shapeType = SpotShapeType_Circular;
    zone.fadeoutRadius = maxRadius / 3;
    zone.color = _zoneColorPalette.getColor((2 + parameters.numZones) * 8);
    zone.values = parameters.baseValues;

    setDefaultShapeDataForZone(zone);

    int index = parameters.numZones;
    parameters.zone[index] = zone;
    origParameters.zone[index] = zone;
    ++parameters.numZones;
    ++origParameters.numZones;
    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);
}

void SimulationParametersMainWindow::onAddSource()
{
    auto& editService = SimulationParametersEditService::get();

    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    if (!checkNumSources(parameters)) {
        return;
    }

    ++_selectedLocationIndex;
    LocationHelper::adaptLocationIndex(parameters, _selectedLocationIndex, 1);
    LocationHelper::adaptLocationIndex(origParameters, _selectedLocationIndex, 1);

    auto strengths = editService.getRadiationStrengths(parameters);
    auto newStrengths = editService.calcRadiationStrengthsForAddingZone(strengths);

    auto worldSize = _simulationFacade->getWorldSize();

    RadiationSource source;
    StringHelper::copy(source.name, sizeof(source.name), LocationHelper::generateSourceName(parameters));
    source.locationIndex = _selectedLocationIndex;
    source.posX = toFloat(worldSize.x / 2);
    source.posY = toFloat(worldSize.y / 2);

    auto index = parameters.numRadiationSources;
    parameters.radiationSource[index] = source;
    origParameters.radiationSource[index] = source;
    ++parameters.numRadiationSources;
    ++origParameters.numRadiationSources;

    editService.applyRadiationStrengths(parameters, newStrengths);
    editService.applyRadiationStrengths(origParameters, newStrengths);

    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);
}

void SimulationParametersMainWindow::onCloneLocation()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    auto location = LocationHelper::findLocation(parameters, _selectedLocationIndex);

    if (std::holds_alternative<SimulationParametersZone*>(location)) {
        if (!checkNumZones(parameters)) {
            return;
        }
    } else {
        if (!checkNumSources(parameters)) {
            return;
        }
    }

    ++_selectedLocationIndex;
    LocationHelper::adaptLocationIndex(parameters, _selectedLocationIndex, 1);
    LocationHelper::adaptLocationIndex(origParameters, _selectedLocationIndex, 1);

    if (std::holds_alternative<SimulationParametersZone*>(location)) {
        auto zone = std::get<SimulationParametersZone*>(location);
        auto clone = *zone;

        StringHelper::copy(clone.name, sizeof(clone.name), LocationHelper::generateZoneName(parameters));
        clone.locationIndex = _selectedLocationIndex;

        int index = parameters.numZones;
        parameters.zone[index] = clone;
        origParameters.zone[index] = clone;
        ++parameters.numZones;
        ++origParameters.numZones;
    } else {
        auto source = std::get<RadiationSource*>(location);
        auto clone = *source;

        auto& editService = SimulationParametersEditService::get();
        auto strengths = editService.getRadiationStrengths(parameters);
        auto newStrengths = editService.calcRadiationStrengthsForAddingZone(strengths);

        StringHelper::copy(clone.name, sizeof(clone.name), LocationHelper::generateSourceName(parameters));
        clone.locationIndex = _selectedLocationIndex;
        auto index = parameters.numRadiationSources;
        parameters.radiationSource[index] = clone;
        origParameters.radiationSource[index] = clone;
        ++parameters.numRadiationSources;
        ++origParameters.numRadiationSources;

        editService.applyRadiationStrengths(parameters, newStrengths);
        editService.applyRadiationStrengths(origParameters, newStrengths);
    }

    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);
}

void SimulationParametersMainWindow::onDeleteLocation()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    LocationController::get().deleteLocationWindow(_selectedLocationIndex);
    auto location = LocationHelper::findLocation(parameters, _selectedLocationIndex);

    if (std::holds_alternative<SimulationParametersZone*>(location)) {
        std::optional<int> zoneIndex;
        for (int i = 0; i < parameters.numZones; ++i) {
            if (parameters.zone[i].locationIndex == _selectedLocationIndex) {
                zoneIndex = i;
                break;
            }
        }
        if (zoneIndex.has_value()) {
            for (int i = zoneIndex.value(); i < parameters.numZones - 1; ++i) {
                parameters.zone[i] = parameters.zone[i + 1];
                origParameters.zone[i] = origParameters.zone[i + 1];
            }
            --parameters.numZones;
            --origParameters.numZones;
        }
    } else {
        std::optional<int> sourceIndex;
        for (int i = 0; i < parameters.numRadiationSources; ++i) {
            if (parameters.radiationSource[i].locationIndex == _selectedLocationIndex) {
                sourceIndex = i;
                break;
            }
        }
        if (sourceIndex.has_value()) {
            for (int i = sourceIndex.value(); i < parameters.numRadiationSources - 1; ++i) {
                parameters.radiationSource[i] = parameters.radiationSource[i + 1];
                origParameters.radiationSource[i] = origParameters.radiationSource[i + 1];
            }
            --parameters.numRadiationSources;
            --origParameters.numRadiationSources;
        }
    }

    auto newByOldLocationIndex = LocationHelper::adaptLocationIndex(parameters, _selectedLocationIndex, -1);
    LocationHelper::adaptLocationIndex(origParameters, _selectedLocationIndex, -1);

    if (_locations.size() - 1 == _selectedLocationIndex) {
        --_selectedLocationIndex;
    }

    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);

    LocationController::get().remapLocationIndices(newByOldLocationIndex);
}

void SimulationParametersMainWindow::onDecreaseLocationIndex()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto newByOldLocationIndex = LocationHelper::onDecreaseLocationIndex(parameters, _selectedLocationIndex);
    _simulationFacade->setSimulationParameters(parameters);

    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    LocationHelper::onDecreaseLocationIndex(origParameters, _selectedLocationIndex);
    _simulationFacade->setOriginalSimulationParameters(parameters);

    --_selectedLocationIndex;
    LocationController::get().remapLocationIndices(newByOldLocationIndex);
}

void SimulationParametersMainWindow::onIncreaseLocationIndex()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto newByOldLocationIndex = LocationHelper::onIncreaseLocationIndex(parameters, _selectedLocationIndex);
    _simulationFacade->setSimulationParameters(parameters);

    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    LocationHelper::onIncreaseLocationIndex(origParameters, _selectedLocationIndex);
    _simulationFacade->setOriginalSimulationParameters(parameters);

    ++_selectedLocationIndex;
    LocationController::get().remapLocationIndices(newByOldLocationIndex);
}

void SimulationParametersMainWindow::onOpenInLocationWindow()
{
    auto mousePos = ImGui::GetMousePos();
    auto offset = RealVector2D{50.0f + toFloat(_locationWindowCounter) * 15, toFloat(_locationWindowCounter) * 15};
    LocationController::get().addLocationWindow(_selectedLocationIndex, {mousePos.x + offset.x, mousePos.y + offset.y});
    _locationWindowCounter = (_locationWindowCounter + 1) % 8;
}

void SimulationParametersMainWindow::onCenterLocation(int locationIndex)
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto location = LocationHelper::findLocation(parameters, locationIndex);
    RealVector2D pos;
    if (std::holds_alternative<SimulationParametersZone*>(location)) {
        auto zone = std::get<SimulationParametersZone*>(location);
        pos = {zone->posX, zone->posY};
    } else {
        auto source = std::get<RadiationSource*>(location);
        pos = {source->posX, source->posY};
    }
    Viewport::get().setCenterInWorldPos(pos);
}

void SimulationParametersMainWindow::updateLocations()
{
    auto parameters = _simulationFacade->getSimulationParameters();

    _locations = std::vector<Location>(1 + parameters.numZones + parameters.numRadiationSources);
    auto strength = SimulationParametersEditService::get().getRadiationStrengths(parameters);
    auto pinnedString = strength.pinned.contains(0) ? ICON_FA_THUMBTACK " " : " ";
    _locations.at(0) = Location{"Base", LocationType::Base, "-", pinnedString + StringHelper::format(strength.values.front() * 100 + 0.05f, 1) + "%"};
    for (int i = 0; i < parameters.numZones; ++i) {
        auto const& zone = parameters.zone[i];
        auto position = "(" + StringHelper::format(zone.posX, 0) + ", " + StringHelper::format(zone.posY, 0) + ")";
        _locations.at(zone.locationIndex) = Location{zone.name, LocationType::ParameterZone, position};
    }
    for (int i = 0; i < parameters.numRadiationSources; ++i) {
        auto const& source = parameters.radiationSource[i];
        auto position = "(" + StringHelper::format(source.posX, 0) + ", " + StringHelper::format(source.posY, 0) + ")";
        auto pinnedString = strength.pinned.contains(i + 1) ? ICON_FA_THUMBTACK " " : " ";
        _locations.at(source.locationIndex) = Location{
            source.name, LocationType::RadiationSource, position, pinnedString + StringHelper::format(strength.values.at(i + 1) * 100 + 0.05f, 1) + "%"};
    }
}

void SimulationParametersMainWindow::setDefaultShapeDataForZone(SimulationParametersZone& spot) const
{
    auto worldSize = _simulationFacade->getWorldSize();

    auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    if (spot.shapeType == SpotShapeType_Circular) {
        spot.shapeData.circularSpot.coreRadius = maxRadius / 3;
    } else {
        spot.shapeData.rectangularSpot.height = maxRadius / 3;
        spot.shapeData.rectangularSpot.width = maxRadius / 3;
    }
}

void SimulationParametersMainWindow::correctLayout(float origMasterHeight, float origExpertWidgetHeight)
{
    auto detailHeight = ImGui::GetWindowSize().y - getMasterWidgetRefHeight() - getExpertWidgetRefHeight();

    if (detailHeight < scale(DetailWidgetMinHeight)
        || _masterWidgetHeight < scale(MasterMinHeight)
        || _expertWidgetHeight < scale(ExpertWidgetMinHeight)) {
        _masterWidgetHeight = origMasterHeight;
        _expertWidgetHeight = origExpertWidgetHeight;
    }
}

bool SimulationParametersMainWindow::checkNumZones(SimulationParameters const& parameters)
{
    if (parameters.numZones == MAX_ZONES) {
        showMessage("Error", "The maximum number of zones has been reached.");
        return false;
    }
    return true;
}

bool SimulationParametersMainWindow::checkNumSources(SimulationParameters const& parameters)
{
    if (parameters.numRadiationSources == MAX_RADIATION_SOURCES) {
        showMessage("Error", "The maximum number of radiation sources has been reached.");
        return false;
    }
    return true;
}

float SimulationParametersMainWindow::getMasterWidgetRefHeight() const
{
    return _masterWidgetOpen ? _masterWidgetHeight : scale(25.0f);
}

float SimulationParametersMainWindow::getExpertWidgetRefHeight() const
{
    return _expertWidgetOpen ? _expertWidgetHeight : scale(30.0f);
}

float SimulationParametersMainWindow::getMasterWidgetHeight() const
{
    if (_masterWidgetOpen && !_detailWidgetOpen && !_expertWidgetOpen) {
        return std::max(scale(MasterMinHeight), ImGui::GetContentRegionAvail().y - getDetailWidgetHeight() - getExpertWidgetRefHeight());
    }
    return getMasterWidgetRefHeight();
}

float SimulationParametersMainWindow::getDetailWidgetHeight() const
{
    return _detailWidgetOpen ? std::max(scale(MasterMinHeight), ImGui::GetContentRegionAvail().y - getExpertWidgetRefHeight() + scale(4.0f)) : scale(25.0f);
}
