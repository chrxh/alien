#include "GpuSettingsDialog.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"
#include "EngineInterface/SimulationFacade.h"

#include "StyleRepository.h"
#include "AlienGui.h"

namespace
{
    auto const RightColumnWidth = 110.0f;
}

void GpuSettingsDialog::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    GpuSettings gpuSettings;
    gpuSettings.numBlocks = GlobalSettings::get().getValue("settings.gpu.num blocks", gpuSettings.numBlocks);

    _simulationFacade->setGpuSettings_async(gpuSettings);
}

void GpuSettingsDialog::shutdownIntern()
{
    auto gpuSettings = _simulationFacade->getGpuSettings();
    GlobalSettings::get().setValue("settings.gpu.num blocks", gpuSettings.numBlocks);
}

GpuSettingsDialog::GpuSettingsDialog()
    : AlienDialog("CUDA settings")
{}

void GpuSettingsDialog::processIntern()
{
    auto gpuSettings = _simulationFacade->getGpuSettings();
    auto origGpuSettings = _simulationFacade->getOriginalGpuSettings();
    auto lastGpuSettings = gpuSettings;

    AlienGui::InputInt(
        AlienGui::InputIntParameters()
            .name("Blocks")
            .textWidth(RightColumnWidth)
            .defaultValue(origGpuSettings.numBlocks)
            .tooltip("This values specifies the number of CUDA thread blocks. If you are using a high-end graphics card, you can try to increase the number of "
                     "blocks."),
        gpuSettings.numBlocks);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button("Adopt")) {
        close();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
        gpuSettings = _gpuSettings;
    }

    validateAndCorrect(gpuSettings);

    if (gpuSettings != lastGpuSettings) {
        _simulationFacade->setGpuSettings_async(gpuSettings);
    }
}

void GpuSettingsDialog::openIntern()
{
    _gpuSettings = _simulationFacade->getGpuSettings();
}

void GpuSettingsDialog::validateAndCorrect(GpuSettings& settings) const
{
    settings.numBlocks = std::min(1000000, std::max(16, settings.numBlocks));
}
