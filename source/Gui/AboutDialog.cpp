#include "AboutDialog.h"

#include <imgui.h>

#include <Base/Resources.h>

#include "AlienGui.h"
#include "StyleRepository.h"

AboutDialog::AboutDialog()
    : AlienDialog(_("About"))
{}

void AboutDialog::processIntern()
{
    ImGui::Text(
        _("Artificial Life Environment, version %s\n\nis an open source project initiated and maintained by\nChristian Heinemann."),
        Const::ProgramVersion.c_str());

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button(_("OK"))) {
        close();
    }
    ImGui::SetItemDefaultFocus();
}
