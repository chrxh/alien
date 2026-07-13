#include "ExitDialog.h"

#include <imgui.h>

#include "AlienGui.h"
#include "MainLoopController.h"

ExitDialog::ExitDialog()
    : AlienDialog(_("Exit"))
{}

void ExitDialog::processIntern()
{
    ImGui::TextWrapped("%s", _("Do you really want to terminate the program?"));

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button(_("OK"))) {
        MainLoopController::get().scheduleClosing();
        close();
    }
    ImGui::SameLine();
    if (AlienGui::Button(_("Cancel"))) {
        close();
    }
    ImGui::SetItemDefaultFocus();
}
