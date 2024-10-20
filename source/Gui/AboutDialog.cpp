#include "AboutDialog.h"

#include <imgui.h>

#include "Base/Resources.h"
#include "AlienImGui.h"
#include "StyleRepository.h"

AboutDialog::AboutDialog()
    : AlienDialog("About")
{}

void AboutDialog::processIntern()
{
    ImGui::Text("Artificial Life Environment, version %s\n\nis an open source project initiated and maintained by\nChristian Heinemann.", Const::ProgramVersion.c_str());

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienImGui::Separator();

    if (AlienImGui::Button("OK")) {
        close();
    }
    ImGui::SetItemDefaultFocus();
}
