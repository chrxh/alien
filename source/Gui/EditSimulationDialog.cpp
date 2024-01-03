#include "EditSimulationDialog.h"

#include <imgui.h>

#include "Network/NetworkService.h"

#include "AlienImGui.h"
#include "BrowserWindow.h"
#include "StyleRepository.h"
#include "MessageDialog.h"

_EditSimulationDialog::_EditSimulationDialog(BrowserWindow const& browserWindow)
    : _AlienDialog("")
    , _browserWindow(browserWindow)
{}

void _EditSimulationDialog::openForLeaf(NetworkResourceTreeTO const& treeTO)
{
    changeTitle("Change name or description");
    _AlienDialog::open();
    _treeTO = treeTO;

    auto& rawTO = _treeTO->getLeaf().rawTO;
    _newName = rawTO->resourceName;
    _newDescription = rawTO->description;
}

void _EditSimulationDialog::openForFolder(NetworkResourceTreeTO const& treeTO, std::vector<NetworkResourceRawTO> const& rawTOs)
{
    changeTitle("Change folder name");
    _AlienDialog::open();
    _treeTO = treeTO;
    _rawTOs = rawTOs;
}

void _EditSimulationDialog::processIntern()
{
    if (_treeTO->isLeaf()) {
        processLeaf();
    }
}

void _EditSimulationDialog::processLeaf()
{
    auto& rawTO = _treeTO->getLeaf().rawTO;
    std::string resourceTypeString = rawTO->resourceType == NetworkResourceType_Simulation ? "simulation" : "genome";

    AlienImGui::InputText(AlienImGui::InputTextParameters().textWidth(0).hint("Name"), _newName);

    AlienImGui::Separator();

    ImGui::PushID("description");
    AlienImGui::InputTextMultiline(
        AlienImGui::InputTextMultilineParameters()
            .hint("Description (optional)")
            .textWidth(0)
            .height(ImGui::GetContentRegionAvail().y - scale(50.0f)),
        _newDescription);
    ImGui::PopID();

    AlienImGui::Separator();

    ImGui::BeginDisabled(rawTO->resourceName.empty());
    if (AlienImGui::Button("OK")) {
        if (!NetworkService::editResource(rawTO->id, _newName, _newDescription)) {
            showMessage("Error", "Failed to edit " + resourceTypeString + ".");
            return;
        }
        _browserWindow->onRefresh();
        close();
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
}
