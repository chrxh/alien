#include "EditSimulationDialog.h"

#include <imgui.h>

#include "Network/NetworkService.h"
#include "Network/NetworkResourceService.h"
#include "Network/ValidationService.h"

#include "AlienImGui.h"
#include "BrowserWindow.h"
#include "HelpStrings.h"
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

    _newName = NetworkResourceService::concatenateFolderName(treeTO->folderNames, false);
}

void _EditSimulationDialog::processIntern()
{
    if (_treeTO->isLeaf()) {
        processForLeaf();
    } else {
        processForFolder();
    }
}

void _EditSimulationDialog::processForLeaf()
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

    ImGui::BeginDisabled(_newName.empty());
    if (AlienImGui::Button("OK")) {
        if (ValidationService::isStringValidForDatabase(_newName) && ValidationService::isStringValidForDatabase(_newDescription)) {
            if (!NetworkService::editResource(rawTO->id, _newName, _newDescription)) {
                showMessage("Error", "Failed to edit " + resourceTypeString + ".");
            } else {
                _browserWindow->onRefresh();
            }
            close();
        } else {
            showMessage("Error", Const::NotAllowedCharacters);
        }
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
}

void _EditSimulationDialog::processForFolder()
{
    if (ImGui::BeginChild("", {0, -scale(50.0f)})) {
        AlienImGui::InputText(AlienImGui::InputTextParameters().textWidth(0).hint("Folder name"), _newName);
    }
    ImGui::EndChild();

    AlienImGui::Separator();

    ImGui::BeginDisabled(_newName.empty());
    if (AlienImGui::Button("OK")) {
        if (ValidationService::isStringValidForDatabase(_newName)) {
            for (auto const& rawTO : _rawTOs) {
                auto nameWithoutFolder = NetworkResourceService::removeFoldersFromName(rawTO->resourceName);
                auto newName = NetworkResourceService::concatenateFolderName({_newName, nameWithoutFolder}, false);
                if (!NetworkService::editResource(rawTO->id, newName, rawTO->description)) {
                    showMessage("Error", "Failed to change folder name.");
                    break;
                }
            }
            _browserWindow->onRefresh();
            close();
        } else {
            showMessage("Error", Const::NotAllowedCharacters);
        }
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
}
