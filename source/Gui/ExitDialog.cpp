#include <imgui.h>

#include "ExitDialog.h"
#include "AlienImGui.h"
#include "MainLoopController.h"

ExitDialog::ExitDialog()
    : AlienDialog("Exit")
{}

void ExitDialog::processIntern()
{
    ImGui::Text("Do you really want to terminate the program?");

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienImGui::Separator();

    if (AlienImGui::Button("OK")) {
        MainLoopController::get().scheduleShutdown();
        close();
    }
    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
    ImGui::SetItemDefaultFocus();
}
