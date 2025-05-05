#include "SimulationParametersMainWindow.h"

#include <ImFileDialog.h>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/StringHelper.h"
#include "EngineInterface/LocationHelper.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/ParametersEditService.h"
#include "PersisterInterface/SerializerService.h"

#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "LocationController.h"
#include "OverlayController.h"
#include "SimulationParametersSourceWidgets.h"
#include "SimulationParametersLayerWidgets.h"
#include "AlienImGui.h"
#include "SpecificationGuiService.h"
#include "Viewport.h"

namespace
{
    auto constexpr MasterHeight = 130.0f;
    auto constexpr MasterMinHeight = 50.0f;
    auto constexpr MasterRowHeight = 23.0f;

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

    auto layerWidgets = std::make_shared<_SimulationParametersLayerWidgets>();
    layerWidgets->init(_simulationFacade, 0);
    _layerWidgets = layerWidgets;


    auto sourceWidgets = std::make_shared<_SimulationParametersSourceWidgets>();
    sourceWidgets->init(_simulationFacade, 0);
    _sourceWidgets = sourceWidgets;
}

void SimulationParametersMainWindow::processIntern()
{
    if (!_sessionId.has_value() || _sessionId.value() != _simulationFacade->getSessionId()) {
        _selectedOrderNumber = 0;
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
                                      .tooltip("Replace reference values by values from the clipboard. This is useful to see the diff between the current "
                                               "parameters and those from the clipboard.")
                                      .disabled(!_copiedParameters))) {
        auto parameters = _simulationFacade->getSimulationParameters();
        if (_copiedParameters->numLayers == parameters.numLayers
            && _copiedParameters->numSources == parameters.numSources) {
            _simulationFacade->setOriginalSimulationParameters(*_copiedParameters);
            printOverlayMessage("Reference simulation parameters replaced");
        } else {
            GenericMessageDialog::get().information(
                "Error", "The number of layers and radiation sources of the current simulation parameters must match with those from the clipboard.");
        }
    }

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_PLUS).secondText(ICON_FA_LAYER_GROUP).tooltip("Add parameter layer"))) {
        onInsertDefaultLayer();
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_PLUS).secondText(ICON_FA_SUN).tooltip("Add radiation source"))) {
        onInsertDefaultSource();
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_PLUS)
                                      .secondText(ICON_FA_CLONE)
                                      .disabled(_selectedOrderNumber == 0)
                                      .tooltip("Clone selected layer/radiation source"))) {
        onCloneLocation();
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(
            AlienImGui::ToolbarButtonParameters().text(ICON_FA_MINUS).disabled(_selectedOrderNumber == 0).tooltip("Delete selected layer/radiation source"))) {
        onDeleteLocation();
    }

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_CHEVRON_UP)
                                      .disabled(_selectedOrderNumber <= 1)
                                      .tooltip("Move selected layer/radiation source upward"))) {
        onDecreaseOrderNumber();
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_CHEVRON_DOWN)
                                      .tooltip("Move selected layer/radiation source downward")
                                      .disabled(_selectedOrderNumber >= _locations.size() - 1 || _selectedOrderNumber == 0))) {
        onIncreaseOrderNumber();
    }

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters()
                                      .text(ICON_FA_EXTERNAL_LINK_SQUARE_ALT)
                                      .tooltip("Open parameters for selected layer/radiation source in a new window"))) {
        onOpenInLocationWindow();
    }

    AlienImGui::Separator();
}

void SimulationParametersMainWindow::processMasterWidget()
{
    if (ImGui::BeginChild("##master", {0, getMasterWidgetHeight()})) {

        if (_masterWidgetOpen = AlienImGui::BeginTreeNode(
                AlienImGui::TreeNodeParameters().name("Overview").rank(AlienImGui::TreeNodeRank::High).defaultOpen(_masterWidgetOpen))) {
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
        if (_detailWidgetOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                                              .name((std::string(title) + "###parameters").c_str())
                                                              .rank(AlienImGui::TreeNodeRank::High)
                                                              .defaultOpen(_detailWidgetOpen))) {
            ImGui::Spacing();
            AlienImGui::SetFilterText(_filter);
            if (ImGui::BeginChild(
                    "##detail2", {0, -ImGui::GetStyle().FramePadding.y - scale(33.0f)}, ImGuiChildFlags_Border, ImGuiWindowFlags_HorizontalScrollbar)) {
                auto type = _locations.at(_selectedOrderNumber).type;
                if (type == LocationType::Base) {
                    _baseWidgets->process();
                } else if (type == LocationType::Layer) {
                    _layerWidgets->setOrderNumber(_selectedOrderNumber);
                    _layerWidgets->process();
                } else if (type == LocationType::Source) {
                    _sourceWidgets->setOrderNumber(_selectedOrderNumber);
                    _sourceWidgets->process();
                }
            }
            ImGui::EndChild();
            AlienImGui::ResetFilterText();

            ImGui::Spacing();
            AlienImGui::InputFilter(AlienImGui::InputFilterParameters().width(250.0f), _filter);
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
                AlienImGui::TreeNodeParameters().name("Expert settings").rank(AlienImGui::TreeNodeRank::High).defaultOpen(_expertWidgetOpen))) {
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
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

        ImGuiListClipper clipper;
        clipper.Begin(toInt(_locations.size()));
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto const& entry = _locations.at(row);

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(MasterRowHeight));

                // name
                ImGui::TableNextColumn();
                auto selected = _selectedOrderNumber == row;
                if (ImGui::Selectable(
                        "",
                        &selected,
                        ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
                        ImVec2(0, scale(MasterRowHeight) - ImGui::GetStyle().FramePadding.y))) {
                    _selectedOrderNumber = row;
                }
                ImGui::SameLine();
                std::string icon;
                if (entry.type == LocationType::Base) {
                    icon = "";
                } else if (entry.type == LocationType::Layer) {
                    icon = ICON_FA_LAYER_GROUP " ";
                } else if (entry.type == LocationType::Source) {
                    icon = ICON_FA_SUN " ";
                }
                AlienImGui::Text(icon + entry.name);


                // type
                ImGui::TableNextColumn();
                if (entry.type == LocationType::Base) {
                    AlienImGui::Text("Base parameters");
                } else if (entry.type == LocationType::Layer) {
                    AlienImGui::Text("Layer");
                } else if (entry.type == LocationType::Source) {
                    AlienImGui::Text("Radiation");
                }

                // position
                ImGui::TableNextColumn();
                if (row > 0) {
                    if (AlienImGui::ActionButton(AlienImGui::ActionButtonParameters().buttonText(ICON_FA_SEARCH))) {
                        onCenterLocation(row);
                        _selectedOrderNumber = row;
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
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    SpecificationGuiService::get().createWidgetsForExpertToggles(parameters, origParameters);

    if (parameters != lastParameters) {
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

void SimulationParametersMainWindow::onInsertDefaultLayer()
{
    auto& editService = ParametersEditService::get();
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    if (!checkNumLayers(parameters)) {
        return;
    }

    auto newByOldOrderNumber = editService.insertDefaultLayer(parameters, _selectedOrderNumber);
    editService.insertDefaultLayer(origParameters, _selectedOrderNumber);

    ++_selectedOrderNumber;

    auto worldSize = _simulationFacade->getWorldSize();
    auto minRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;

    auto index = LocationHelper::findLocationArrayIndex(parameters, _selectedOrderNumber);
    parameters.backgroundColor.layerValues[index].enabled = true;
    parameters.backgroundColor.layerValues[index].value = _layerColorPalette.getColor((2 + parameters.numLayers) * 8);
    parameters.layerShape.layerValues[index] = LayerShapeType_Circular;
    parameters.layerPosition.layerValues[index] = {toFloat(worldSize.x / 2), toFloat(worldSize.y / 2)};
    parameters.layerCoreRadius.layerValues[index] = minRadius / 3;
    parameters.layerCoreRect.layerValues[index] = {minRadius / 3, minRadius / 3};
    parameters.layerFadeoutRadius.layerValues[index] = minRadius / 3;
    parameters.layerForceFieldType.layerValues[index] = ForceField_None;
    parameters.layerRadialForceFieldOrientation.layerValues[index] = Orientation_Clockwise;
    parameters.layerRadialForceFieldStrength.layerValues[index] = 0.001f;
    parameters.layerRadialForceFieldDriftAngle.layerValues[index] = 0.0f;

    origParameters.backgroundColor.layerValues[index] = parameters.backgroundColor.layerValues[index];
    origParameters.layerShape.layerValues[index] = parameters.layerShape.layerValues[index];
    origParameters.layerPosition.layerValues[index] = parameters.layerPosition.layerValues[index];
    origParameters.layerCoreRadius.layerValues[index] = parameters.layerCoreRadius.layerValues[index];
    origParameters.layerCoreRect.layerValues[index] = parameters.layerCoreRect.layerValues[index];
    origParameters.layerFadeoutRadius.layerValues[index] = parameters.layerFadeoutRadius.layerValues[index];
    origParameters.layerForceFieldType.layerValues[index] = parameters.layerForceFieldType.layerValues[index];
    origParameters.layerRadialForceFieldOrientation.layerValues[index] = parameters.layerRadialForceFieldOrientation.layerValues[index];
    origParameters.layerRadialForceFieldStrength.layerValues[index] = parameters.layerRadialForceFieldStrength.layerValues[index];
    origParameters.layerRadialForceFieldDriftAngle.layerValues[index] = parameters.layerRadialForceFieldDriftAngle.layerValues[index];

    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);

    LocationController::get().remapLocationIndices(newByOldOrderNumber);
}

void SimulationParametersMainWindow::onInsertDefaultSource()
{
    auto& editService = ParametersEditService::get();
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    if (!checkNumSources(parameters)) {
        return;
    }
    auto strengths = editService.getRadiationStrengths(parameters);
    auto newStrengths = editService.calcRadiationStrengthsForAddingSource(strengths);

    auto newByOldOrderNumber = editService.insertDefaultSource(parameters, _selectedOrderNumber);
    editService.insertDefaultSource(origParameters, _selectedOrderNumber);

    ++_selectedOrderNumber;

    editService.applyRadiationStrengths(parameters, newStrengths);
    editService.applyRadiationStrengths(origParameters, newStrengths);

    auto index = LocationHelper::findLocationArrayIndex(parameters, _selectedOrderNumber);
    auto worldSize = _simulationFacade->getWorldSize();
    parameters.sourcePosition.sourceValues[index] = {toFloat(worldSize.x / 2), toFloat(worldSize.y / 2)};
    origParameters.sourcePosition.sourceValues[index] = parameters.sourcePosition.sourceValues[index];

    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);

    LocationController::get().remapLocationIndices(newByOldOrderNumber);
}

void SimulationParametersMainWindow::onCloneLocation()
{
    auto& editService = ParametersEditService::get();
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    auto locationType = LocationHelper::getLocationType(_selectedOrderNumber, parameters);
    if (locationType == LocationType::Layer) {
        if (!checkNumLayers(parameters)) {
            return;
        }
    } else {
        if (!checkNumSources(parameters)) {
            return;
        }
    }

    auto strengths = editService.getRadiationStrengths(parameters);
    auto newStrengths = editService.calcRadiationStrengthsForAddingSource(strengths);

    auto newByOldOrderNumber = editService.cloneLocation(parameters, _selectedOrderNumber);
    editService.cloneLocation(origParameters, _selectedOrderNumber);

    if (locationType == LocationType::Source) {
        editService.applyRadiationStrengths(parameters, newStrengths);
        editService.applyRadiationStrengths(origParameters, newStrengths);
    }

    ++_selectedOrderNumber;
    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);

    LocationController::get().remapLocationIndices(newByOldOrderNumber);
}

void SimulationParametersMainWindow::onDeleteLocation()
{
    auto& editService = ParametersEditService::get();
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    LocationController::get().deleteLocationWindow(_selectedOrderNumber);

    auto newByOldOrderNumber = editService.deleteLocation(parameters, _selectedOrderNumber);
    editService.deleteLocation(origParameters, _selectedOrderNumber);

    if (_locations.size() - 1 == _selectedOrderNumber) {
        --_selectedOrderNumber;
    }

    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);

    LocationController::get().remapLocationIndices(newByOldOrderNumber);
}

void SimulationParametersMainWindow::onDecreaseOrderNumber()
{
    auto& editService = ParametersEditService::get();
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    auto newByOldOrderNumber = editService.moveLocationUpwards(parameters, _selectedOrderNumber);
    editService.moveLocationUpwards(origParameters, _selectedOrderNumber);

    --_selectedOrderNumber;

    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);

    LocationController::get().remapLocationIndices(newByOldOrderNumber);
}

void SimulationParametersMainWindow::onIncreaseOrderNumber()
{
    auto& editService = ParametersEditService::get();
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    auto newByOldOrderNumber = editService.moveLocationDownwards(parameters, _selectedOrderNumber);
    editService.moveLocationDownwards(origParameters, _selectedOrderNumber);

    ++_selectedOrderNumber;

    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);

    LocationController::get().remapLocationIndices(newByOldOrderNumber);
}

void SimulationParametersMainWindow::onOpenInLocationWindow()
{
    auto mousePos = ImGui::GetMousePos();
    auto offset = RealVector2D{50.0f + toFloat(_locationWindowCounter) * 15, toFloat(_locationWindowCounter) * 15};
    LocationController::get().addLocationWindow(_selectedOrderNumber, {mousePos.x + offset.x, mousePos.y + offset.y});
    _locationWindowCounter = (_locationWindowCounter + 1) % 8;
}

void SimulationParametersMainWindow::onCenterLocation(int orderNumber)
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
    auto arrayIndex = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
    RealVector2D pos;
    if (locationType == LocationType::Layer) {
        pos = parameters.layerPosition.layerValues[arrayIndex];
    } else if (locationType == LocationType::Source) {
        pos = parameters.sourcePosition.sourceValues[arrayIndex];
    }
    Viewport::get().setCenterInWorldPos(pos);
}

void SimulationParametersMainWindow::updateLocations()
{
    auto parameters = _simulationFacade->getSimulationParameters();

    _locations = std::vector<Location>(1 + parameters.numLayers + parameters.numSources);
    auto strength = ParametersEditService::get().getRadiationStrengths(parameters);
    auto pinnedString = strength.pinned.contains(0) ? ICON_FA_THUMBTACK " " : " ";
    _locations.at(0) = Location{"Base", LocationType::Base, "-", pinnedString + StringHelper::format(strength.values.front() * 100 + 0.05f, 1) + "%"};
    for (int i = 0; i < parameters.numLayers; ++i) {
        auto position =
            "(" + StringHelper::format(parameters.layerPosition.layerValues[i].x, 0) + ", " + StringHelper::format(parameters.layerPosition.layerValues[i].y, 0) + ")";
        _locations.at(parameters.layerOrderNumbers[i]) = Location{parameters.layerName.layerValues[i], LocationType::Layer, position};
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        auto position = "(" + StringHelper::format(parameters.sourcePosition.sourceValues[i].x, 0) + ", "
            + StringHelper::format(parameters.sourcePosition.sourceValues[i].y, 0) + ")";
        auto pinnedString = strength.pinned.contains(i + 1) ? ICON_FA_THUMBTACK " " : " ";
        _locations.at(parameters.sourceOrderNumbers[i]) =
            Location{
            parameters.sourceName.sourceValues[i],
            LocationType::Source,
            position,
            pinnedString + StringHelper::format(strength.values.at(i + 1) * 100 + 0.05f, 1) + "%"};
    }
}

void SimulationParametersMainWindow::correctLayout(float origMasterHeight, float origExpertWidgetHeight)
{
    auto detailHeight = ImGui::GetWindowSize().y - getMasterWidgetRefHeight() - getExpertWidgetRefHeight();

    if (detailHeight < scale(DetailWidgetMinHeight) || _masterWidgetHeight < scale(MasterMinHeight) || _expertWidgetHeight < scale(ExpertWidgetMinHeight)) {
        _masterWidgetHeight = origMasterHeight;
        _expertWidgetHeight = origExpertWidgetHeight;
    }
}

bool SimulationParametersMainWindow::checkNumLayers(SimulationParameters const& parameters)
{
    if (parameters.numLayers == MAX_LAYERS) {
        showMessage("Error", "The maximum number of layers has been reached.");
        return false;
    }
    return true;
}

bool SimulationParametersMainWindow::checkNumSources(SimulationParameters const& parameters)
{
    if (parameters.numSources == MAX_SOURCES) {
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

