#include "ColorizeDialog.h"

#include "imgui.h"

_ColorizeDialog::_ColorizeDialog(SimulationController const& simController)
    : _simController(simController)
{}

void _ColorizeDialog::process()
{
    if (!_show) {
        return;
    }
    auto name = "Colorize";
    ImGui::OpenPopup(name);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal(name, NULL, ImGuiWindowFlags_AlwaysAutoResize)) {

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Spacing();

        if (ImGui::Button("OK")) {
            onColorize();
            ImGui::CloseCurrentPopup();
            _show = false;
        }
        ImGui::SetItemDefaultFocus();
        ImGui::EndPopup();
    }
}

void _ColorizeDialog::show()
{
    _show = true;
}

void _ColorizeDialog::onColorize()
{
}
