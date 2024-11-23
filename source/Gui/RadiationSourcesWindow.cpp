#include <imgui.h>

#include "AlienImGui.h"
#include "Base/Definitions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SimulationParametersEditService.h"

#include "RadiationSourcesWindow.h"
#include "StyleRepository.h"
#include "SimulationInteractionController.h"

namespace
{
    auto const RightColumnWidth = 170.0f;
}

void RadiationSourcesWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;
}

RadiationSourcesWindow::RadiationSourcesWindow()
    : AlienWindow("Radiation sources", "windows.radiation sources", false)
{}

void RadiationSourcesWindow::processIntern()
{
    std::optional<bool> scheduleAppendTab;
    std::optional<int> scheduleDeleteTabAtIndex;

    if (ImGui::BeginTabBar("##ParticleSources", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {
        auto parameters = _simulationFacade->getSimulationParameters();

        //add source
        if (parameters.numRadiationSources < MAX_RADIATION_SOURCES) {
            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                scheduleAppendTab = true;
            }
            AlienImGui::Tooltip("Add radiation source");
        }

        processBaseTab();

        for (int tab = 0; tab < parameters.numRadiationSources; ++tab) {
            if (!processSourceTab(tab)) {
                scheduleDeleteTabAtIndex = tab;
            }
        }

        ImGui::EndTabBar();
    }

    if (scheduleAppendTab.has_value()) {
        onAppendTab();
    }
    if (scheduleDeleteTabAtIndex.has_value()) {
        onDeleteTab(scheduleDeleteTabAtIndex.value());
    }

    auto currentSessionId = _simulationFacade->getSessionId();
    _focusBaseTab = !_sessionId.has_value() || currentSessionId != *_sessionId;
    _sessionId = currentSessionId;
}

void RadiationSourcesWindow::processBaseTab()
{
    if (ImGui::BeginTabItem("Base", nullptr, _focusBaseTab ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None)) {
        auto parameters = _simulationFacade->getSimulationParameters();
        auto lastParameters = parameters;
        auto origParameters = _simulationFacade->getOriginalSimulationParameters();

        auto& editService = SimulationParametersEditService::get();

        auto strength = editService.getRadiationStrengths(parameters);
        auto origStrengths = editService.getRadiationStrengths(origParameters);
        auto editedStrength = strength;
        if (AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Relative strength")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.3f")
                    .defaultValue(&origStrengths.values.front())
                    .tooltip("Cells can emit energy particles over time. A portion of this energy can be released directly near the cell, while the rest is "
                             "utilized by one of the available radiation sources. This parameter determines the fraction of energy assigned to the emitted "
                             "energy particle in the vicinity of the cell. Values between 0 and 1 are permitted."),
                &editedStrength.values.front(),
                nullptr,
                &parameters.baseStrengthRatioPinned)) {
            editService.adaptRadiationStrengths(editedStrength, strength, 0);
            editService.applyRadiationStrengths(parameters, editedStrength);
        }

        if (parameters != lastParameters) {
            _simulationFacade->setSimulationParameters(parameters, SimulationParametersUpdateConfig::AllExceptChangingPositions);
        }
        ImGui::EndTabItem();
    }
}

bool RadiationSourcesWindow::processSourceTab(int index)
{
    auto& editService = SimulationParametersEditService::get();

    auto parameters = _simulationFacade->getSimulationParameters();
    auto lastParameters = parameters;
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    auto worldSize = _simulationFacade->getWorldSize();

    RadiationSource& source = parameters.radiationSources[index];
    RadiationSource& origSource = origParameters.radiationSources[index];

    bool isOpen = true;
    static char name[20] = {};

    snprintf(name, IM_ARRAYSIZE(name), "Source %01d", index + 1);
    if (ImGui::BeginTabItem(name, &isOpen, ImGuiTabItemFlags_None)) {
        if (AlienImGui::Switcher(
                AlienImGui::SwitcherParameters()
                    .name("Shape")
                    .values({"Circular", "Rectangular"})
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSource.shapeType),
                source.shapeType)) {
            if (source.shapeType == RadiationSourceShapeType_Circular) {
                source.shapeData.circularRadiationSource.radius = 1;
            } else {
                source.shapeData.rectangularRadiationSource.width = 40;
                source.shapeData.rectangularRadiationSource.height = 10;
            }
        }

        auto origStrengths = editService.getRadiationStrengths(parameters);
        if (AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Relative strength")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.3f")
                    .defaultValue(&origSource.strength)
                    .tooltip("Cells can emit energy particles over time. A portion of this energy can be released directly near the cell, while the rest is "
                             "utilized by one of the available radiation sources. This parameter determines the fraction of energy assigned to the emitted "
                             "energy particle for the selected radiation source. Values between 0 and 1 are permitted."),
                &source.strength,
                nullptr,
                &source.strengthPinned)) {
            auto editedStrengths = origStrengths;
            editedStrengths.values.at(index + 1) = source.strength;   // Set new strength
            editService.adaptRadiationStrengths(editedStrengths, origStrengths, index + 1);
            editService.applyRadiationStrengths(parameters, editedStrengths);
        }

        auto getMousePickerEnabledFunc = [&]() { return SimulationInteractionController::get().isPositionSelectionMode(); };
        auto setMousePickerEnabledFunc = [&](bool value) { SimulationInteractionController::get().setPositionSelectionMode(value); };
        auto getMousePickerPositionFunc = [&]() { return SimulationInteractionController::get().getPositionSelectionData(); };
        AlienImGui::SliderFloat2(
            AlienImGui::SliderFloat2Parameters()
                .name("Position (x,y)")
                .textWidth(RightColumnWidth)
                .min({0, 0})
                .max(toRealVector2D(worldSize))
                .defaultValue(RealVector2D{origSource.posX, origSource.posY})
                .format("%.2f")
                .getMousePickerEnabledFunc(getMousePickerEnabledFunc)
                .setMousePickerEnabledFunc(setMousePickerEnabledFunc)
                .getMousePickerPositionFunc(getMousePickerPositionFunc),
            source.posX,
            source.posY);
        AlienImGui::SliderFloat2(
            AlienImGui::SliderFloat2Parameters()
                .name("Velocity (x,y)")
                .textWidth(RightColumnWidth)
                .min({-4.0f, -4.0f})
                .max({4.0f, 4.0f})
                .defaultValue(RealVector2D{origSource.velX, origSource.velY})
                .format("%.2f"),
            source.velX,
            source.velY);
        if (source.shapeType == RadiationSourceShapeType_Circular) {
            auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y));
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Radius")
                    .textWidth(RightColumnWidth)
                    .min(1)
                    .max(maxRadius)
                    .format("%.0f")
                    .defaultValue(&origSource.shapeData.circularRadiationSource.radius),
                &source.shapeData.circularRadiationSource.radius);
        }
        if (source.shapeType == RadiationSourceShapeType_Rectangular) {
            AlienImGui::SliderFloat2(
                AlienImGui::SliderFloat2Parameters()
                    .name("Size (x,y)")
                    .textWidth(RightColumnWidth)
                    .min({0, 0})
                    .max({toFloat(worldSize.x), toFloat(worldSize.y)})
                    .defaultValue(RealVector2D{origSource.shapeData.rectangularRadiationSource.height, origSource.shapeData.rectangularRadiationSource.height})
                    .format("%.1f"),
                source.shapeData.rectangularRadiationSource.width,
                source.shapeData.rectangularRadiationSource.height);
        }
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radiation angle")
                .textWidth(RightColumnWidth)
                .min(-180.0f)
                .max(180.0f)
                .defaultEnabledValue(&origSource.useAngle)
                .defaultValue(&origSource.angle)
                .disabledValue(&source.angle)
                .format("%.1f"),
            &source.angle,
            &source.useAngle);
        ImGui::EndTabItem();
        validateAndCorrect(source);
    }

    if (parameters != lastParameters) {
        _simulationFacade->setSimulationParameters(parameters);
    }

    return isOpen;
}

void RadiationSourcesWindow::onAppendTab()
{
    auto& editService = SimulationParametersEditService::get();

    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    auto strengths = editService.getRadiationStrengths(parameters);
    auto newStrengths = editService.calcRadiationStrengthsForAddingSpot(strengths);

    auto index = parameters.numRadiationSources;
    parameters.radiationSources[index] = createParticleSource();
    origParameters.radiationSources[index] = createParticleSource();
    ++parameters.numRadiationSources;
    ++origParameters.numRadiationSources;

    editService.applyRadiationStrengths(parameters, newStrengths);
    editService.applyRadiationStrengths(origParameters, newStrengths);

    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);
}

void RadiationSourcesWindow::onDeleteTab(int index)
{
    auto& editService = SimulationParametersEditService::get();
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    auto strengths = editService.getRadiationStrengths(parameters);
    auto newStrengths = editService.calcRadiationStrengthsForDeletingSpot(strengths, index + 1);

    for (int i = index; i < parameters.numRadiationSources - 1; ++i) {
        parameters.radiationSources[i] = parameters.radiationSources[i + 1];
        origParameters.radiationSources[i] = origParameters.radiationSources[i + 1];
    }
    --parameters.numRadiationSources;
    --origParameters.numRadiationSources;

    editService.applyRadiationStrengths(parameters, newStrengths);
    editService.applyRadiationStrengths(origParameters, newStrengths);

    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);
}

RadiationSource RadiationSourcesWindow::createParticleSource() const
{
    RadiationSource result;
    auto worldSize = _simulationFacade->getWorldSize();
    result.posX = toFloat(worldSize.x / 2);
    result.posY = toFloat(worldSize.y / 2);
    return result;
}

void RadiationSourcesWindow::validateAndCorrect(RadiationSource& source) const
{
    if (source.shapeType == RadiationSourceShapeType_Circular) {
        source.shapeData.circularRadiationSource.radius = std::max(1.0f, source.shapeData.circularRadiationSource.radius);
    }
    if (source.shapeType == RadiationSourceShapeType_Rectangular) {
        source.shapeData.rectangularRadiationSource.width = std::max(1.0f, source.shapeData.rectangularRadiationSource.width);
        source.shapeData.rectangularRadiationSource.height = std::max(1.0f, source.shapeData.rectangularRadiationSource.height);
    }
}
