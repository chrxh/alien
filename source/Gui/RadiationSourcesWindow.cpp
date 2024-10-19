#include <imgui.h>

#include "AlienImGui.h"
#include "Base/Definitions.h"
#include "EngineInterface/SimulationFacade.h"

#include "RadiationSourcesWindow.h"
#include "StyleRepository.h"
#include "SimulationInteractionController.h"

namespace
{
    auto const RightColumnWidth = 140.0f;
}

_RadiationSourcesWindow::_RadiationSourcesWindow(SimulationFacade const& simulationFacade)
    : AlienWindow("Radiation sources", "windows.radiation sources", false)
    , _simulationFacade(simulationFacade)
{}

void _RadiationSourcesWindow::processIntern()
{
    auto parameters = _simulationFacade->getSimulationParameters();

    std::optional<bool> scheduleAppendTab;
    std::optional<int> scheduleDeleteTabAtIndex;

    if (ImGui::BeginTabBar("##ParticleSources", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (parameters.numRadiationSources < MAX_RADIATION_SOURCES) {

            //add source
            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                scheduleAppendTab = true;
            }
            AlienImGui::Tooltip("Add source");
        }

        for (int tab = 0; tab < parameters.numRadiationSources; ++tab) {
            if (!processTab(tab)) {
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
}

bool _RadiationSourcesWindow::processTab(int index)
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto lastParameters = parameters;
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    auto worldSize = _simulationFacade->getWorldSize();

    RadiationSource& source = parameters.radiationSources[index];
    RadiationSource& origSource = origParameters.radiationSources[index];

    bool isOpen = true;
    char name[20] = {};

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
        validationAndCorrection(source);
    }

    if (parameters != lastParameters) {
        _simulationFacade->setSimulationParameters(parameters);
    }

    return isOpen;
}

void _RadiationSourcesWindow::onAppendTab()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    auto index = parameters.numRadiationSources;
    parameters.radiationSources[index] = createParticleSource();
    origParameters.radiationSources[index] = createParticleSource();
    ++parameters.numRadiationSources;
    ++origParameters.numRadiationSources;

    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);
}

void _RadiationSourcesWindow::onDeleteTab(int index)
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    for (int i = index; i < parameters.numRadiationSources - 1; ++i) {
        parameters.radiationSources[i] = parameters.radiationSources[i + 1];
        origParameters.radiationSources[i] = origParameters.radiationSources[i + 1];
    }
    --parameters.numRadiationSources;
    --origParameters.numRadiationSources;
    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);
}

RadiationSource _RadiationSourcesWindow::createParticleSource() const
{
    RadiationSource result;
    auto worldSize = _simulationFacade->getWorldSize();
    result.posX = toFloat(worldSize.x / 2);
    result.posY = toFloat(worldSize.y / 2);
    return result;
}

void _RadiationSourcesWindow::validationAndCorrection(RadiationSource& source) const
{
    if (source.shapeType == RadiationSourceShapeType_Circular) {
        source.shapeData.circularRadiationSource.radius = std::max(1.0f, source.shapeData.circularRadiationSource.radius);
    }
    if (source.shapeType == RadiationSourceShapeType_Rectangular) {
        source.shapeData.rectangularRadiationSource.width = std::max(1.0f, source.shapeData.rectangularRadiationSource.width);
        source.shapeData.rectangularRadiationSource.height = std::max(1.0f, source.shapeData.rectangularRadiationSource.height);
    }
}
