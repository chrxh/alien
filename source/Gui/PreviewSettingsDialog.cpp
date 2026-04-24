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
    AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Show node index").tooltip("Show node indices instead of cell function names"), _editData->showNodeIndex);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button("Close")) {
        close();
    }
}
