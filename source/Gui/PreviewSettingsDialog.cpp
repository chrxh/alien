#include "PreviewSettingsDialog.h"

#include <imgui.h>

#include "AlienGui.h"
#include "GenomeTabEditData.h"

PreviewSettingsDialog::PreviewSettingsDialog()
    : AlienDialog("Preview settings")
{}

void PreviewSettingsDialog::setEditData(GenomeTabEditData const& editData)
{
    _editData = editData;
}

void PreviewSettingsDialog::processIntern()
{
    // Convert boolean to switcher index: 0 = Node index, 1 = Cell function
    int displayMode = _editData->showNodeIndex ? 0 : 1;

    AlienGui::Switcher(
        AlienGui::SwitcherParameters()
            .name("Display mode")
            .textWidth(scale(120.0f))
            .values({"Node index", "Cell function"}),
        displayMode);

    // Convert switcher index back to boolean
    _editData->showNodeIndex = (displayMode == 0);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button("Close")) {
        close();
    }
}
