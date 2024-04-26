#include "GpuSettingsDialog.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"
#include "EngineInterface/SimulationController.h"

#include "StyleRepository.h"
#include "AlienImGui.h"

namespace
{
    auto const RightColumnWidth = 180.0f;
}

_GpuSettingsDialog::_GpuSettingsDialog(SimulationController const& simController)
    : _AlienDialog("CUDA settings")
    , _simController(simController)
{
    GpuSettings gpuSettings;
    gpuSettings.numBlocks = GlobalSettings::getInstance().getInt("settings.gpu.num blocks", gpuSettings.numBlocks);
    gpuSettings.numThreadsPerBlock = GlobalSettings::getInstance().getInt("settings.gpu.num threads per block", gpuSettings.numThreadsPerBlock);

    _simController->setGpuSettings_async(gpuSettings);
}

_GpuSettingsDialog::~_GpuSettingsDialog()
{
    auto gpuSettings = _simController->getGpuSettings();
    GlobalSettings::getInstance().setInt("settings.gpu.num blocks", gpuSettings.numBlocks);
    GlobalSettings::getInstance().setInt("settings.gpu.num threads per block", gpuSettings.numThreadsPerBlock);
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
                .tooltip(std::string("Number of CUDA thread blocks.")),
            gpuSettings.numBlocks);

        AlienImGui::InputInt(
            AlienImGui::InputIntParameters()
                .name("Threads per Block")
                .textWidth(RightColumnWidth)
                .defaultValue(origGpuSettings.numThreadsPerBlock)
                .tooltip(std::string("Number of CUDA threads per blocks.")),
            gpuSettings.numThreadsPerBlock);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        gpuSettings.numBlocks = std::max(gpuSettings.numBlocks, 1);
        gpuSettings.numThreadsPerBlock = std::max(gpuSettings.numThreadsPerBlock, 1);

        ImGui::Text("Total threads");
        ImGui::PushFont(StyleRepository::getInstance().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        ImGui::TextUnformatted(
            StringHelper::format(gpuSettings.numBlocks * gpuSettings.numThreadsPerBlock).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
        AlienImGui::Separator();

        if (AlienImGui::Button("OK")) {
            close();
        }
        ImGui::SetItemDefaultFocus();

        ImGui::SameLine();
        if (AlienImGui::Button("Cancel")) {
            close();
            gpuSettings = _gpuSettings;
        }

    if (gpuSettings != lastGpuSettings) {
        _simController->setGpuSettings_async(gpuSettings);
    }
}

void _GpuSettingsDialog::openIntern()
{
    _gpuSettings = _simController->getGpuSettings();
}
