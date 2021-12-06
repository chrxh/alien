#include "GpuSettingsDialog.h"

#include "imgui.h"

#include "Base/StringFormatter.h"
#include "EngineImpl/SimulationController.h"
#include "StyleRepository.h"
#include "AlienImGui.h"
#include "GlobalSettings.h"


_GpuSettingsDialog::_GpuSettingsDialog(
    StyleRepository const& styleRepository,
    SimulationController const& simController)
    : _styleRepository(styleRepository)
    , _simController(simController)
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

    ImGui::OpenPopup("GPU settings");
    if (ImGui::BeginPopupModal("GPU settings", NULL, ImGuiWindowFlags_None)) {

        AlienImGui::InputInt(
            "Blocks", gpuSettings.NUM_BLOCKS, origGpuSettings.NUM_BLOCKS, std::string("Number of GPU thread blocks."));

        AlienImGui::InputInt(
            "Threads per Block",
            gpuSettings.NUM_THREADS_PER_BLOCK,
            origGpuSettings.NUM_THREADS_PER_BLOCK,
            std::string("Number of GPU threads per blocks."));

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        gpuSettings.NUM_BLOCKS = std::max(gpuSettings.NUM_BLOCKS, 1);
        gpuSettings.NUM_THREADS_PER_BLOCK = std::max(gpuSettings.NUM_THREADS_PER_BLOCK, 1);

        ImGui::Text("Total threads");
        ImGui::PushFont(_styleRepository->getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        ImGui::TextUnformatted(
            StringFormatter::format(gpuSettings.NUM_BLOCKS * gpuSettings.NUM_THREADS_PER_BLOCK).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        AlienImGui::Separator();

        if (ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
            _show = false;
        }
        ImGui::SetItemDefaultFocus();

        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
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
