#include "AboutDialog.h"

#include <imgui.h>

#include "Base/Resources.h"
#include "AlienImGui.h"

_AboutDialog::_AboutDialog()
    : _AlienDialog("About")
{}

void _AboutDialog::processIntern()
{
    ImGui::Text("Artificial Life Environment, version %s\n\nis an open source project initiated by\nChristian Heinemann.", Const::ProgramVersion.c_str());

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();

    if (AlienImGui::Button("OK")) {
        close();
    }
    ImGui::SetItemDefaultFocus();
}
