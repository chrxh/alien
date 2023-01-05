#include <imgui.h>

#include "AlienImGui.h"
#include "Base/Definitions.h"
#include "EngineInterface/SimulationController.h"

#include "RadiationSourcesWindow.h"
#include "GlobalSettings.h"
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
    auto origParameters = _simController->getOriginalSimulationParameters();

    auto worldSize = _simController->getWorldSize();

    if (ImGui::BeginTabBar("##ParticleSources", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (parameters.numParticleSources < MAX_PARTICLE_SOURCES) {

            //add source
            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                auto index = parameters.numParticleSources;
                parameters.particleSources[index] = createParticleSource();
                origParameters.particleSources[index] = createParticleSource();
                ++parameters.numParticleSources;
                ++origParameters.numParticleSources;
                _simController->setSimulationParameters_async(parameters);
                _simController->setOriginalSimulationParameters(origParameters);
            }
            AlienImGui::Tooltip("Add source");
        }

        for (int tab = 0; tab < parameters.numParticleSources; ++tab) {
            RadiationSource& flowCenter = parameters.particleSources[tab];
            RadiationSource& origFlowCenter = origParameters.particleSources[tab];
            bool open = true;
            char name[18] = {};
            bool* openPtr = &open;
            snprintf(name, IM_ARRAYSIZE(name), "Source %01d", tab + 1);
            if (ImGui::BeginTabItem(name, openPtr, ImGuiTabItemFlags_None)) {

                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Position X")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(toFloat(worldSize.x))
                        .format("%.0f")
                        .defaultValue(origFlowCenter.posX),
                    flowCenter.posX);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Position Y")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(toFloat(worldSize.y))
                        .format("%.0f")
                        .defaultValue(origFlowCenter.posY),
                    flowCenter.posY);

                ImGui::EndTabItem();
            }

            //del source
            if (!open) {
                for (int i = tab; i < parameters.numParticleSources - 1; ++i) {
                    parameters.particleSources[i] = parameters.particleSources[i + 1];
                    origParameters.particleSources[i] = origParameters.particleSources[i + 1];
                }
                --parameters.numParticleSources;
                --origParameters.numParticleSources;
                _simController->setSimulationParameters_async(parameters);
                _simController->setOriginalSimulationParameters(origParameters);
            }
        }

        ImGui::EndTabBar();
    }
    if (parameters != origParameters) {
        _simController->setSimulationParameters_async(parameters);
    }
}

RadiationSource _RadiationSourcesWindow::createParticleSource() const
{
    RadiationSource result;
    auto worldSize = _simController->getWorldSize();
    result.posX = toFloat(worldSize.x / 2);
    result.posY = toFloat(worldSize.y / 2);
    return result;
}
