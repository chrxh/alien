#include "FlowGeneratorWindow.h"

#include "imgui.h"

#include "Base/Definitions.h"
#include "EngineImpl/SimulationController.h"
#include "StyleRepository.h"

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
    auto origFlowFieldSettings = flowFieldSettings;

    auto worldSize = _simController->getWorldSize();

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha);
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
                flowFieldSettings.radialFlowCenters[flowFieldSettings.numCenters] =
                    flowFieldSettings.radialFlowCenters[flowFieldSettings.numCenters - 1];
                ++flowFieldSettings.numCenters;
            }
        }

        for (int tab = 0; tab < flowFieldSettings.numCenters; ++tab) {
            RadialFlowCenterData& radialFlowData = flowFieldSettings.radialFlowCenters[tab];
            bool open = true;
            char name[16];
            bool* openPtr = flowFieldSettings.numCenters == 1 ? NULL : &open;
            snprintf(name, IM_ARRAYSIZE(name), "Center %01d", tab + 1);
            if (ImGui::BeginTabItem(name, openPtr, ImGuiTabItemFlags_None)) {
                if (ImGui::BeginTable("##", 2, ImGuiTableFlags_SizingStretchProp)) {

                    //pos x
                    ImGui::TableNextRow();

                    ImGui::TableSetColumnIndex(0);
                    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                    ImGui::SliderFloat(
                        "##posX", &radialFlowData.posX, 0.0, toFloat(worldSize.x), "%.0f", ImGuiSliderFlags_None);

                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("Position X");

                    //pos y
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                    ImGui::SliderFloat(
                        "##posY", &radialFlowData.posY, 0.0, toFloat(worldSize.y), "%.0f", ImGuiSliderFlags_None);

                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("Position Y");

                    //radius
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                    ImGui::SliderFloat(
                        "##radius",
                        &radialFlowData.radius,
                        0.0,
                        std::min(toFloat(worldSize.x), toFloat(worldSize.y)),
                        "%.0f",
                        ImGuiSliderFlags_None);

                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("Radius");

                    //strength
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);

                    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                    ImGui::SliderFloat(
                        "##strength", &radialFlowData.strength, 0.0, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);

                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("Strength");

                    //orientation
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    const char* orientations[] = {"Clockwise", "Counter clockwise"};
                    int currentOrientations = radialFlowData.orientation == Orientation::Clockwise ? 0 : 1;
                    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
                    ImGui::Combo("##", &currentOrientations, orientations, IM_ARRAYSIZE(orientations));
                    radialFlowData.orientation =
                        currentOrientations == 0 ? Orientation::Clockwise : Orientation::CounterClockwise;
                    ImGui::PopItemWidth();

                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("Orientation");

                    ImGui::EndTable();
                }
                ImGui::EndTabItem();
            }

            if (!open) {
                for (int i = tab; i < flowFieldSettings.numCenters - 1; ++i) {
                    flowFieldSettings.radialFlowCenters[i] = flowFieldSettings.radialFlowCenters[i + 1];
                }
                --flowFieldSettings.numCenters;
            }
        }

        ImGui::EndTabBar();
    }
    ImGui::EndDisabled();
    ImGui::End();

    if (flowFieldSettings != origFlowFieldSettings) {
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
