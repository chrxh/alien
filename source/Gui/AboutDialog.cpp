#include "AboutDialog.h"

#include <imgui.h>

#include "Resources.h"
#include "AlienImGui.h"

_AboutDialog::_AboutDialog() {}

void _AboutDialog::process()
{
    if (!_show) {
        return;
    }
    auto name = "About";
    ImGui::OpenPopup(name);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal(name, NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text(
            "Artificial Life Environment, version %s\n\nis an open source project initiated by\nChristian Heinemann.",
            Const::ProgramVersion.c_str());

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Spacing();

        if (AlienImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
            _show = false;
        }
        ImGui::SetItemDefaultFocus();
        ImGui::EndPopup();
    }
}

void _AboutDialog::show()
{
    _show = true;
}
