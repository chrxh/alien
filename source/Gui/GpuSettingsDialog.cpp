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

_GpuSettingsDialog::_GpuSettingsDialog(SimulationFacade const& simulationFacade)
    : _AlienDialog("CUDA settings")
    , _simulationFacade(simulationFacade)
{
    GpuSettings gpuSettings;
    gpuSettings.numBlocks = GlobalSettings::get().getInt("settings.gpu.num blocks", gpuSettings.numBlocks);

    _simulationFacade->setGpuSettings_async(gpuSettings);
}

_GpuSettingsDialog::~_GpuSettingsDialog()
{
    auto gpuSettings = _simulationFacade->getGpuSettings();
    GlobalSettings::get().setInt("settings.gpu.num blocks", gpuSettings.numBlocks);
}

void _GpuSettingsDialog::processIntern()
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

void _GpuSettingsDialog::openIntern()
{
    _gpuSettings = _simulationFacade->getGpuSettings();
}

void _GpuSettingsDialog::validationAndCorrection(GpuSettings& settings) const
{
    settings.numBlocks = std::min(1000000, std::max(8, settings.numBlocks));
}
