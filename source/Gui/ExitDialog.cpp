#include <imgui.h>

#include "ExitDialog.h"
#include "AlienImGui.h"

_ExitDialog::_ExitDialog(bool& onExit)
    : AlienDialog("Exit")
    , _onExit(onExit)
{}

void _ExitDialog::processIntern()
{
    ImGui::Text("Do you really want to terminate the program?");

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();

    if (AlienImGui::Button("OK")) {
        _onExit = true;
        close();
    }
    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
    ImGui::SetItemDefaultFocus();
}
