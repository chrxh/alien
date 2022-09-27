#include "GpuSettingsDialog.h"

#include <imgui.h>

#include "Base/StringHelper.h"
#include "EngineInterface/SimulationController.h"
#include "StyleRepository.h"
#include "AlienImGui.h"
#include "GlobalSettings.h"

namespace
{
    auto const MaxContentTextWidth = 180.0f;
}

_GpuSettingsDialog::_GpuSettingsDialog(SimulationController const& simController)
    : _simController(simController)
{
    auto gpuSettings = GlobalSettings::getInstance().getGpuSettings();
    _simController->setGpuSettings_async(gpuSettings);
}

_GpuSettingsDialog::~_GpuSettingsDialog()
{
    auto gpuSettings = _simController->getGpuSettings();
    GlobalSettings::getInstance().setGpuSettings(gpuSettings);
}

void _GpuSettingsDialog::process()
{
    if (!_show) {
        return;
    }
    auto gpuSettings = _simController->getGpuSettings();
    auto origGpuSettings = _simController->getOriginalGpuSettings();
    auto lastGpuSettings = gpuSettings;

    ImGui::OpenPopup("CUDA settings");
    if (ImGui::BeginPopupModal("CUDA settings", NULL, ImGuiWindowFlags_None)) {

        AlienImGui::InputInt(
            AlienImGui::InputIntParameters()
                .name("Blocks")
                .textWidth(MaxContentTextWidth)
                .defaultValue(origGpuSettings.numBlocks)
                .tooltip(std::string("Number of CUDA thread blocks.")),
            gpuSettings.numBlocks);

        AlienImGui::InputInt(
            AlienImGui::InputIntParameters()
                .name("Threads per Block")
                .textWidth(MaxContentTextWidth)
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

        AlienImGui::Separator();

        if (AlienImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
            _show = false;
        }
        ImGui::SetItemDefaultFocus();

        ImGui::SameLine();
        if (AlienImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
            _show = false;
            gpuSettings = _gpuSettings;
        }

        ImGui::EndPopup();
    }
    if (gpuSettings != lastGpuSettings) {
        _simController->setGpuSettings_async(gpuSettings);
    }
}

void _GpuSettingsDialog::show()
{
    _show = true;
    _gpuSettings = _simController->getGpuSettings();
}
