#include <imgui.h>

#include "AlienImGui.h"
#include "Base/Definitions.h"
#include "EngineInterface/SimulationController.h"

#include "RadiationSourcesWindow.h"
#include "StyleRepository.h"

namespace
{
    auto const RightColumnWidth = 120.0f;
}

_RadiationSourcesWindow::_RadiationSourcesWindow(SimulationController const& simController)
    : _AlienWindow("Radiation sources", "windows.radiation sources", false)
    , _simController(simController)
{}

void _RadiationSourcesWindow::processIntern()
{
    auto parameters = _simController->getSimulationParameters();

    std::optional<bool> scheduleAddTab;
    std::optional<int> scheduleDelTabAtIndex;

    if (ImGui::BeginTabBar("##ParticleSources", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (parameters.numRadiationSources < MAX_RADIATION_SOURCES) {

            //add source
            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                scheduleAddTab = true;
            }
            AlienImGui::Tooltip("Add source");
        }

        for (int tab = 0; tab < parameters.numRadiationSources; ++tab) {
            if (!processTab(tab)) {
                scheduleDelTabAtIndex = tab;
            }
        }

        ImGui::EndTabBar();
    }

    if (scheduleAddTab.has_value()) {
        processAppendTab();
    }
    if (scheduleDelTabAtIndex.has_value()) {
        processDelTab(scheduleDelTabAtIndex.value());
    }
}

bool _RadiationSourcesWindow::processTab(int index)
{
    auto parameters = _simController->getSimulationParameters();
    auto lastParameters = parameters;
    auto origParameters = _simController->getOriginalSimulationParameters();

    auto worldSize = _simController->getWorldSize();

    RadiationSource& source = parameters.radiationSources[index];
    RadiationSource& origSource = origParameters.radiationSources[index];

    bool isOpen = true;
    char name[18] = {};

    snprintf(name, IM_ARRAYSIZE(name), "Source %01d", index + 1);
    if (ImGui::BeginTabItem(name, &isOpen, ImGuiTabItemFlags_None)) {

        if (AlienImGui::Combo(
                AlienImGui::ComboParameters().name("Shape").values({"Circular", "Rectangular"}).textWidth(RightColumnWidth).defaultValue(origSource.shapeType),
                source.shapeType)) {
            if (source.shapeType == RadiationSourceShapeType_Circular) {
                source.shapeData.circularRadiationSource.radius = 1;
            } else {
                source.shapeData.rectangularRadiationSource.width = 40;
                source.shapeData.rectangularRadiationSource.height = 10;
            }
        }

        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Position X")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(toFloat(worldSize.x))
                .format("%.0f")
                .defaultValue(&origSource.posX),
            &source.posX);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Position Y")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(toFloat(worldSize.y))
                .format("%.0f")
                .defaultValue(&origSource.posY),
            &source.posY);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Velocity X")
                .textWidth(RightColumnWidth)
                .min(-4.0f)
                .max(4.0f)
                .format("%.3f")
                .defaultValue(&origSource.velX),
            &source.velX);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Velocity Y")
                .textWidth(RightColumnWidth)
                .min(-4.0f)
                .max(4.0f)
                .format("%.3f")
                .defaultValue(&origSource.velY),
            &source.velY);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Angle")
                .textWidth(RightColumnWidth)
                .min(-180.0f)
                .max(180.0f)
                .defaultEnabledValue(&origSource.useAngle)
                .defaultValue(&origSource.angle)
                .disabledValue(&source.angle)
                .format("%.1f"),
            &source.angle,
            &source.useAngle);
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
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Width")
                    .textWidth(RightColumnWidth)
                    .min(1)
                    .max(toFloat(worldSize.x))
                    .format("%.0f")
                    .defaultValue(&origSource.shapeData.rectangularRadiationSource.width),
                &source.shapeData.rectangularRadiationSource.width);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Height")
                    .textWidth(RightColumnWidth)
                    .min(1)
                    .max(toFloat(worldSize.y))
                    .format("%.0f")
                    .defaultValue(&origSource.shapeData.rectangularRadiationSource.height),
                &source.shapeData.rectangularRadiationSource.height);
        }
        ImGui::EndTabItem();
        validationAndCorrection(source);
    }

    if (parameters != lastParameters) {
        _simController->setSimulationParameters(parameters);
    }

    return isOpen;
}

void _RadiationSourcesWindow::processAppendTab()
{
    auto parameters = _simController->getSimulationParameters();
    auto origParameters = _simController->getOriginalSimulationParameters();

    auto index = parameters.numRadiationSources;
    parameters.radiationSources[index] = createParticleSource();
    origParameters.radiationSources[index] = createParticleSource();
    ++parameters.numRadiationSources;
    ++origParameters.numRadiationSources;

    _simController->setSimulationParameters(parameters);
    _simController->setOriginalSimulationParameters(origParameters);
}

void _RadiationSourcesWindow::processDelTab(int index)
{
    auto parameters = _simController->getSimulationParameters();
    auto origParameters = _simController->getOriginalSimulationParameters();

    for (int i = index; i < parameters.numRadiationSources - 1; ++i) {
        parameters.radiationSources[i] = parameters.radiationSources[i + 1];
        origParameters.radiationSources[i] = origParameters.radiationSources[i + 1];
    }
    --parameters.numRadiationSources;
    --origParameters.numRadiationSources;
    _simController->setSimulationParameters(parameters);
    _simController->setOriginalSimulationParameters(origParameters);
}

RadiationSource _RadiationSourcesWindow::createParticleSource() const
{
    RadiationSource result;
    auto worldSize = _simController->getWorldSize();
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
