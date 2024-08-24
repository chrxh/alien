#include "GpuSettingsDialog.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"
#include "EngineInterface/SimulationController.h"

#include "StyleRepository.h"
#include "AlienImGui.h"

namespace
{
    auto const RightColumnWidth = 110.0f;
}

_GpuSettingsDialog::_GpuSettingsDialog(SimulationController const& simController)
    : _AlienDialog("CUDA settings")
    , _simController(simController)
{
    GpuSettings gpuSettings;
    gpuSettings.numBlocks = GlobalSettings::getInstance().getInt("settings.gpu.num blocks", gpuSettings.numBlocks);

    _simController->setGpuSettings_async(gpuSettings);
}

_GpuSettingsDialog::~_GpuSettingsDialog()
{
    auto gpuSettings = _simController->getGpuSettings();
    GlobalSettings::getInstance().setInt("settings.gpu.num blocks", gpuSettings.numBlocks);
}

void _GpuSettingsDialog::processIntern()
{
    auto gpuSettings = _simController->getGpuSettings();
    auto origGpuSettings = _simController->getOriginalGpuSettings();
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
        _simController->setGpuSettings_async(gpuSettings);
    }
}

void _GpuSettingsDialog::openIntern()
{
    _gpuSettings = _simController->getGpuSettings();
}

void _GpuSettingsDialog::validationAndCorrection(GpuSettings& settings) const
{
    settings.numBlocks = std::min(1000000, std::max(8, settings.numBlocks));
}
