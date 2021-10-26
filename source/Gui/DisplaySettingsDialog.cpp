#include "DisplaySettingsDialog.h"

#include "imgui.h"
#include "AlienImGui.h"

_DisplaySettingsDialog::_DisplaySettingsDialog(GLFWwindow* window)
    : _window(window)
{}

void _DisplaySettingsDialog::process()
{
    if (!_show) {
        return;
    }

    ImGui::OpenPopup("Display settings");
    if (ImGui::BeginPopupModal("Display settings", NULL, ImGuiWindowFlags_None)) {

        const char* displaySizes[] = {"Default", "1920 x 1080"};
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::Combo("##", &_currentDisplaySize, displaySizes, IM_ARRAYSIZE(displaySizes));
        ImGui::PopItemWidth();

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
        }

        ImGui::EndPopup();
    }
}

void _DisplaySettingsDialog::show()
{
    _show = true;
}
