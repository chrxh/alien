#include "FlowGeneratorWindow.h"

#include <imgui.h>

#include "Base/Definitions.h"
#include "EngineInterface/SimulationController.h"
#include "StyleRepository.h"
#include "AlienImGui.h"
#include "GlobalSettings.h"

namespace
{
    auto const MaxContentTextWidth = 120.0f;
}

_FlowGeneratorWindow::_FlowGeneratorWindow(SimulationController const& simController)
    : _AlienWindow("Flow generator", "windows.flow generator", false)
    , _simController(simController)
{
}

void _FlowGeneratorWindow::processIntern()
{
    auto parameters = _simController->getSimulationParameters();
    auto origParameters = _simController->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    auto worldSize = _simController->getWorldSize();

    if (ImGui::BeginTabBar(
            "##Flow",
            ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (parameters.numFlowCenters < MAX_FLOW_CENTERS) {
            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                auto index = parameters.numFlowCenters;
                parameters.flowCenters[index] = createFlowCenter();
                origParameters.flowCenters[index] = createFlowCenter();
                ++parameters.numFlowCenters;
                ++origParameters.numFlowCenters;
                _simController->setSimulationParameters_async(parameters);
                _simController->setOriginalSimulationParameters(origParameters);
            }
            AlienImGui::Tooltip("Add center");
        }

        for (int tab = 0; tab < parameters.numFlowCenters; ++tab) {
            FlowCenter& flowCenter = parameters.flowCenters[tab];
            FlowCenter& origFlowCenter = origParameters.flowCenters[tab];
            bool open = true;
            char name[18] = {};
            bool* openPtr = &open;
            snprintf(name, IM_ARRAYSIZE(name), "Center %01d", tab + 1);
            if (ImGui::BeginTabItem(name, openPtr, ImGuiTabItemFlags_None)) {

                int dummy = 0;
                AlienImGui::Combo(AlienImGui::ComboParameters().name("Type").values({"Radial flow"}).textWidth(MaxContentTextWidth), dummy);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Position X")
                        .textWidth(MaxContentTextWidth)
                        .min(0)
                        .max(toFloat(worldSize.x))
                        .format("%.0f")
                        .defaultValue(origFlowCenter.posX),
                    flowCenter.posX);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Position Y")
                        .textWidth(MaxContentTextWidth)
                        .min(0)
                        .max(toFloat(worldSize.y))
                        .format("%.0f")
                        .defaultValue(origFlowCenter.posY),
                    flowCenter.posY);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Radius")
                        .textWidth(MaxContentTextWidth)
                        .min(0)
                        .max(std::min(toFloat(worldSize.x), toFloat(worldSize.y)))
                        .format("%.0f")
                        .defaultValue(origFlowCenter.radius),
                    flowCenter.radius);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Strength")
                        .textWidth(MaxContentTextWidth)
                        .min(0)
                        .max(0.5f)
                        .logarithmic(true)
                        .format("%.5f")
                        .defaultValue(origFlowCenter.strength),
                    flowCenter.strength);

                std::vector<std::string> orientations = {"Clockwise", "Counter clockwise"};
                int currentOrientation = flowCenter.orientation == Orientation::Clockwise ? 0 : 1;
                int origCurrentOrientation = origFlowCenter.orientation == Orientation::Clockwise ? 0 : 1;
                AlienImGui::Combo(
                    AlienImGui::ComboParameters()
                        .name("Orientation")
                        .textWidth(MaxContentTextWidth)
                        .defaultValue(origCurrentOrientation)
                        .values(orientations),
                    currentOrientation);
                flowCenter.orientation =
                    currentOrientation == 0 ? Orientation::Clockwise : Orientation::CounterClockwise;
                ImGui::EndTabItem();
            }
            if (!open) {
                for (int i = tab; i < parameters.numFlowCenters - 1; ++i) {
                    parameters.flowCenters[i] = parameters.flowCenters[i + 1];
                    origParameters.flowCenters[i] = origParameters.flowCenters[i + 1];
                }
                --parameters.numFlowCenters;
                --origParameters.numFlowCenters;
                _simController->setSimulationParameters_async(parameters);
                _simController->setOriginalSimulationParameters(origParameters);

            }
        }

        ImGui::EndTabBar();
    }
    if (parameters != lastParameters) {
        _simController->setSimulationParameters_async(parameters);
    }
}

FlowCenter _FlowGeneratorWindow::createFlowCenter() const
{
    FlowCenter result;
    auto worldSize = _simController->getWorldSize();
    result.posX = toFloat(worldSize.x / 2);
    result.posY = toFloat(worldSize.y / 2);
    auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    result.radius = maxRadius / 3;
    return result;
}
