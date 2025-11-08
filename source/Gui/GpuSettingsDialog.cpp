#include "GpuSettingsDialog.h"

#include <imgui.h>

#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "StyleRepository.h"
#include <EngineInterface/SimulationFacade.h>

namespace
{
    auto const RightColumnWidth = 110.0f;
}

void GpuSettingsDialog::initIntern()
{

    CudaSettings gpuSettings;
    gpuSettings.numBlocks = GlobalSettings::get().getValue("settings.gpu.num blocks", gpuSettings.numBlocks);

    _SimulationFacade::get()->setGpuSettings_async(gpuSettings);
}

void GpuSettingsDialog::shutdownIntern()
{
    auto gpuSettings = _SimulationFacade::get()->getGpuSettings();
    GlobalSettings::get().setValue("settings.gpu.num blocks", gpuSettings.numBlocks);
}

GpuSettingsDialog::GpuSettingsDialog()
    : AlienDialog("CUDA settings")
{}

void GpuSettingsDialog::processIntern()
{
    auto gpuSettings = _SimulationFacade::get()->getGpuSettings();
    auto origGpuSettings = _SimulationFacade::get()->getOriginalGpuSettings();
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
        _SimulationFacade::get()->setGpuSettings_async(gpuSettings);
    }
}

void GpuSettingsDialog::openIntern()
{
    _gpuSettings = _SimulationFacade::get()->getGpuSettings();
}

void GpuSettingsDialog::validateAndCorrect(CudaSettings& settings) const
{
    settings.numBlocks = std::min(1000000, std::max(16, settings.numBlocks));
}
