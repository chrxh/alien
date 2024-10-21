#include "GpuSettingsDialog.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"
#include "EngineInterface/SimulationFacade.h"

#include "StyleRepository.h"
#include "AlienImGui.h"

namespace
{
    auto const RightColumnWidth = 110.0f;
}

void GpuSettingsDialog::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    GpuSettings gpuSettings;
    gpuSettings.numBlocks = GlobalSettings::get().getInt("settings.gpu.num blocks", gpuSettings.numBlocks);

    _simulationFacade->setGpuSettings_async(gpuSettings);
}

void GpuSettingsDialog::shutdownIntern()
{
    auto gpuSettings = _simulationFacade->getGpuSettings();
    GlobalSettings::get().setInt("settings.gpu.num blocks", gpuSettings.numBlocks);
}

GpuSettingsDialog::GpuSettingsDialog()
    : AlienDialog("CUDA settings")
{}

void GpuSettingsDialog::processIntern()
{
    auto gpuSettings = _simulationFacade->getGpuSettings();
    auto origGpuSettings = _simulationFacade->getOriginalGpuSettings();
    auto lastGpuSettings = gpuSettings;

    AlienImGui::InputInt(
        AlienImGui::InputIntParameters()
            .name("Blocks")
            .textWidth(RightColumnWidth)
            .defaultValue(origGpuSettings.numBlocks)
            .tooltip("This values specifies the number of CUDA thread blocks. If you are using a high-end graphics card, you can try to increase the number of "
                     "blocks."),
        gpuSettings.numBlocks);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienImGui::Separator();

    if (AlienImGui::Button("Adopt")) {
        close();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
        gpuSettings = _gpuSettings;
    }

    validationAndCorrection(gpuSettings);

    if (gpuSettings != lastGpuSettings) {
        _simulationFacade->setGpuSettings_async(gpuSettings);
    }
}

void GpuSettingsDialog::openIntern()
{
    _gpuSettings = _simulationFacade->getGpuSettings();
}

void GpuSettingsDialog::validationAndCorrection(GpuSettings& settings) const
{
    settings.numBlocks = std::min(1000000, std::max(16, settings.numBlocks));
}
