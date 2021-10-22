#include "FlowGeneratorWindow.h"

#include "imgui.h"

#include "Base/Definitions.h"
#include "EngineImpl/SimulationController.h"
#include "StyleRepository.h"
#include "AlienImGui.h"

_FlowGeneratorWindow::_FlowGeneratorWindow(SimulationController const& simController)
    : _simController(simController)
{
}

void _FlowGeneratorWindow::process()
{
    if (!_on) {
        return;
    }
    auto flowFieldSettings = _simController->getFlowFieldSettings();
    auto origFlowFieldSettings = _simController->getOriginalFlowFieldSettings();
    auto lastFlowFieldSettings = flowFieldSettings;

    auto worldSize = _simController->getWorldSize();

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::Begin("Flow field", &_on, ImGuiWindowFlags_None);

    ImGui::Checkbox("##", &flowFieldSettings.active);
    ImGui::SameLine();
    
    const char* flowTypes[] = {"Radial flow"};
    int currentFlowTypes = 0;
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::Combo("##", &currentFlowTypes, flowTypes, IM_ARRAYSIZE(flowTypes));
    ImGui::PopItemWidth();

    ImGui::BeginDisabled(!flowFieldSettings.active);
    if (ImGui::BeginTabBar(
            "##Flow",
            ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (flowFieldSettings.numCenters < 2) {
            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                auto index = flowFieldSettings.numCenters;
                flowFieldSettings.centers[index] = createFlowCenter();
                _simController->setOriginalFlowFieldCenter(flowFieldSettings.centers[index], index);
                ++flowFieldSettings.numCenters;
            }
        }

        for (int tab = 0; tab < flowFieldSettings.numCenters; ++tab) {
            FlowCenter& flowCenter = flowFieldSettings.centers[tab];
            FlowCenter& origFlowCenter = origFlowFieldSettings.centers[tab];
            bool open = true;
            char name[16];
            bool* openPtr = flowFieldSettings.numCenters == 1 ? NULL : &open;
            snprintf(name, IM_ARRAYSIZE(name), "Center %01d", tab + 1);
            if (ImGui::BeginTabItem(name, openPtr, ImGuiTabItemFlags_None)) {

                AlienImGui::SliderFloat(
                    "Position X", flowCenter.posX, origFlowCenter.posX, 0.0, toFloat(worldSize.x), false, "%.0f");
                AlienImGui::SliderFloat(
                    "Position Y", flowCenter.posY, origFlowCenter.posY, 0.0, toFloat(worldSize.y), false, "%.0f");
                AlienImGui::SliderFloat(
                    "Radius",
                    flowCenter.radius,
                    origFlowCenter.radius,
                    0.0,
                    std::min(toFloat(worldSize.x), toFloat(worldSize.y)),
                    false,
                    "%.0f");
                AlienImGui::SliderFloat(
                    "Strength", flowCenter.strength, origFlowCenter.strength, 0.0, 0.5f, false, "%.3f");

                std::vector<std::string> orientations = {"Clockwise", "Counter clockwise"};
                int currentOrientation = flowCenter.orientation == Orientation::Clockwise ? 0 : 1;
                int origCurrentOrientation = origFlowCenter.orientation == Orientation::Clockwise ? 0 : 1;
                AlienImGui::Combo("Orientation", currentOrientation, origCurrentOrientation, orientations);
                flowCenter.orientation =
                    currentOrientation == 0 ? Orientation::Clockwise : Orientation::CounterClockwise;
                ImGui::EndTabItem();
            }
            if (!open) {
                for (int i = tab; i < flowFieldSettings.numCenters - 1; ++i) {
                    flowFieldSettings.centers[i] = flowFieldSettings.centers[i + 1];
                    _simController->setOriginalFlowFieldCenter(flowFieldSettings.centers[i], i);
                }
                --flowFieldSettings.numCenters;
            }
        }

        ImGui::EndTabBar();
    }
    ImGui::EndDisabled();
    ImGui::End();

    if (flowFieldSettings != lastFlowFieldSettings) {
        _simController->setFlowFieldSettings_async(flowFieldSettings);
    }
}

bool _FlowGeneratorWindow::isOn() const
{
    return _on;
}

void _FlowGeneratorWindow::setOn(bool value)
{
    _on = value;
}

FlowCenter _FlowGeneratorWindow::createFlowCenter()
{
    FlowCenter result;
    auto worldSize = _simController->getWorldSize();
    result.posX = toFloat(worldSize.x / 2);
    result.posY = toFloat(worldSize.y / 2);
    auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    result.radius = maxRadius / 3;

    return result;
}
