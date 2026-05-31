#include "PreviewSettingsDialog.h"

#include <imgui.h>

#include "AlienGui.h"
#include "GenomeWindowEditData.h"

PreviewSettingsDialog::PreviewSettingsDialog()
    : AlienDialog("Preview settings")
{}

void PreviewSettingsDialog::setEditData(GenomeWindowEditData const& editData)
{
    _editData = editData;
}

void PreviewSettingsDialog::processIntern()
{
    // Convert boolean to switcher index: 0 = Node index, 1 = Cell type
    int displayMode = _showNodeIndex ? 0 : 1;

    AlienGui::Switcher(AlienGui::SwitcherParameters().name("Display mode").textWidth(scale(120.0f)).values({"Node index", "Cell type"}), &displayMode);

    // Convert switcher index back to boolean
    _showNodeIndex = (displayMode == 0);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button("Adopt")) {
        close();
        _editData->showNodeIndex = _showNodeIndex;
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
}

void PreviewSettingsDialog::openIntern()
{
    _showNodeIndex = _editData->showNodeIndex;
}
