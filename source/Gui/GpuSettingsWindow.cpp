#include "GpuSettingsWindow.h"

#include "Base/StringFormatter.h"
#include "EngineImpl/SimulationController.h"
#include "StyleRepository.h"
#include "Widgets.h"

#include "imgui.h"

_GpuSettingsWindow::_GpuSettingsWindow(
    StyleRepository const& styleRepository,
    SimulationController const& simController)
    : _styleRepository(styleRepository)
    , _simController(simController)
{}

void _GpuSettingsWindow::process()
{
    if (!_on) {
        return;
    }
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None;
    auto gpuSettings = _simController->getGpuSettings();
    auto origGpuSettings = gpuSettings;

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha);
    ImGui::Begin("GPU settings", &_on, windowFlags);

    if (ImGui::BeginTable("##", 2, ImGuiTableFlags_SizingStretchProp)) {

        //blocks
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::InputInt("##width", &gpuSettings.NUM_BLOCKS);
        ImGui::PopItemWidth();

        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Blocks");
        ImGui::SameLine();
        Widgets::processHelpMarker("This is a more typical looking tree with selectable nodes.\n"
                                   "Click to select, CTRL+Click to toggle, click on arrows or double-click to open.");

        //threads per block
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::InputInt("##height", &gpuSettings.NUM_THREADS_PER_BLOCK);
        ImGui::PopItemWidth();

        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Threads per Block");
        ImGui::SameLine();
        Widgets::processHelpMarker("This is a more typical looking tree with selectable nodes.\n"
                                   "Click to select, CTRL+Click to toggle, click on arrows or double-click to open.");

        ImGui::TableNextRow();
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();

        //total threads
        gpuSettings.NUM_BLOCKS = std::max(gpuSettings.NUM_BLOCKS, 1);
        gpuSettings.NUM_THREADS_PER_BLOCK = std::max(gpuSettings.NUM_THREADS_PER_BLOCK, 1);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
        int totalThreads = gpuSettings.NUM_BLOCKS * gpuSettings.NUM_THREADS_PER_BLOCK;
        char text[100] = "test";
        strcpy_s(text, StringFormatter::format(gpuSettings.NUM_BLOCKS * gpuSettings.NUM_THREADS_PER_BLOCK).c_str());
        ImGui::InputText("##", text, IM_ARRAYSIZE(text), ImGuiInputTextFlags_ReadOnly);
        ImGui::PopItemWidth();

        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Total threads");

        ImGui::EndTable();
    }

    ImGui::End();

    if (gpuSettings != origGpuSettings) {
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
