#include <imgui.h>

#include "ExitDialog.h"
#include "AlienGui.h"
#include "MainLoopController.h"

ExitDialog::ExitDialog()
    : AlienDialog("Exit")
{}

void ExitDialog::processIntern()
{
    ImGui::TextWrapped("%s", "Do you really want to terminate the program?");

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button("OK")) {
        MainLoopController::get().scheduleClosing();
        close();
    }
    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
    ImGui::SetItemDefaultFocus();
}
