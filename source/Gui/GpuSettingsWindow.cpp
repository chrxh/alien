#include "GpuSettingsWindow.h"

#include "imgui.h"

#include "Base/StringFormatter.h"
#include "EngineImpl/SimulationController.h"
#include "StyleRepository.h"
#include "AlienImGui.h"
#include "GlobalSettings.h"


_GpuSettingsWindow::_GpuSettingsWindow(
    StyleRepository const& styleRepository,
    SimulationController const& simController,
    GlobalSettings const& globalSettings)
    : _styleRepository(styleRepository)
    , _simController(simController)
    , _globalSettings(globalSettings)
{
    auto gpuSettings = _globalSettings->getGpuSettings();
    _simController->setGpuSettings_async(gpuSettings);
}

_GpuSettingsWindow::~_GpuSettingsWindow()
{
    auto gpuSettings = _simController->getGpuSettings();
    _globalSettings->setGpuSettings(gpuSettings);
}

void _GpuSettingsWindow::process()
{
    if (!_on) {
        return;
    }
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None;
    auto gpuSettings = _simController->getGpuSettings();
    auto origGpuSettings = _simController->getOriginalGpuSettings();
    auto lastGpuSettings = gpuSettings;

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::Begin("GPU settings", &_on, windowFlags);

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
    ImGui::Text(StringFormatter::format(gpuSettings.NUM_BLOCKS * gpuSettings.NUM_THREADS_PER_BLOCK).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();

    ImGui::End();

    if (gpuSettings != lastGpuSettings) {
        _simController->setGpuSettings_async(gpuSettings);
    }
}

bool _GpuSettingsWindow::isOn() const
{
    return _on;
}

void _GpuSettingsWindow::setOn(bool value)
{
    _on = value;
}
