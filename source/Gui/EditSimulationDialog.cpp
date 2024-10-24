#include "EditSimulationDialog.h"

#include <imgui.h>

#include "Network/NetworkService.h"
#include "Network/NetworkResourceService.h"
#include "Network/ValidationService.h"

#include "AlienImGui.h"
#include "BrowserWindow.h"
#include "DelayedExecutionController.h"
#include "HelpStrings.h"
#include "StyleRepository.h"
#include "GenericMessageDialog.h"
#include "OverlayMessageController.h"

void EditSimulationDialog::openForLeaf(NetworkResourceTreeTO const& treeTO)
{
    changeTitle("Change name or description");
    AlienDialog::open();
    _treeTO = treeTO;

    auto& rawTO = _treeTO->getLeaf().rawTO;
    _newName = rawTO->resourceName;
    _newDescription = rawTO->description;
}

void EditSimulationDialog::openForFolder(NetworkResourceTreeTO const& treeTO, std::vector<NetworkResourceRawTO> const& rawTOs)
{
    changeTitle("Change folder name");
    AlienDialog::open();
    _treeTO = treeTO;
    _rawTOs = rawTOs;
    
    _newName = NetworkResourceService::get().concatenateFolderName(treeTO->folderNames, false);
    _origFolderName = _newName;
}

EditSimulationDialog::EditSimulationDialog()
    : AlienDialog("")
{}

void EditSimulationDialog::processIntern()
{
    if (_treeTO->isLeaf()) {
        processForLeaf();
    } else {
        processForFolder();
    }
}

void EditSimulationDialog::processForLeaf()
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
            delayedExecution([rawTO = rawTO, resourceTypeString = resourceTypeString, this] {
                if (!NetworkService::get().editResource(rawTO->id, _newName, _newDescription)) {
                    showMessage("Error", "Failed to edit " + resourceTypeString + ".");
                }
                BrowserWindow::get().onRefresh();
            });
            printOverlayMessage("Applying changes ...");
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

void EditSimulationDialog::processForFolder()
{
    if (ImGui::BeginChild("##Folder", {0, -scale(50.0f)})) {
        AlienImGui::InputText(AlienImGui::InputTextParameters().textWidth(0).hint("Folder name"), _newName);
    }
    ImGui::EndChild();

    AlienImGui::Separator();

    ImGui::BeginDisabled(_newName.empty());
    if (AlienImGui::Button("OK")) {
        if (ValidationService::isStringValidForDatabase(_newName)) {
            delayedExecution([this] {
                for (auto const& rawTO : _rawTOs) {
                    auto nameWithoutOldFolder = rawTO->resourceName.substr(_origFolderName.size() + 1);
                    auto newName = NetworkResourceService::get().concatenateFolderName({_newName, nameWithoutOldFolder}, false);
                    if (!NetworkService::get().editResource(rawTO->id, newName, rawTO->description)) {
                        showMessage("Error", "Failed to change folder name.");
                        break;
                    }
                }
                BrowserWindow::get().onRefresh();
            });
            printOverlayMessage("Applying changes ...");
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
