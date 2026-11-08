#include "EditSimulationDialog.h"

#include <algorithm>

#include <boost/algorithm/string/trim.hpp>

#include <imgui.h>

#include <Base/StringHelper.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>
#include <Network/NetworkValidationService.h>

#include <Persister/Interface/SimulationLibraryService.h>

#include "AlienGui.h"
#include "BrowserData.h"
#include "BrowserWindow.h"
#include "DelayedExecutionController.h"
#include "GenericMessageDialog.h"
#include "HelpStrings.h"
#include "NetworkTransferController.h"
#include "OverlayController.h"
#include "SimulationLibraryController.h"
#include "StyleService.h"

namespace
{
    auto constexpr FooterHeight = 50.0f;
    auto constexpr LocalNameHint = "A name must not contain < > : \" \\ | ? * and must not begin or end with a dot or a space";
    auto constexpr EmptyFolderHint = "A name must not begin or end with a slash or contain two in a row";

    std::string getLocalNameHint(std::string const& name)
    {
        return name.starts_with('/') || name.ends_with('/') || name.contains("//") ? EmptyFolderHint : LocalNameHint;
    }
}

void EditSimulationDialog::openForLeaf(NetworkResourceTreeTO const& treeTO)
{
    changeTitle("Change name or description");
    AlienDialog::open();
    _treeTO = treeTO;
    _rawTOs.clear();
    _errorMessage.reset();

    auto& rawTO = _treeTO->getLeaf().rawTO;
    _newName = rawTO->resourceName;
    _newDescription = rawTO->description;

    // A description of this computer that is longer than the dialog can show must not be cut by saving it
    _descriptionEditable = true;
    if (isLocal()) {
        auto const& content = SimulationLibraryController::get().getContent();
        auto const& simulations = rawTO->resourceType == NetworkResourceType_Genome ? content.genomes : content.simulations;
        auto findResult =
            std::ranges::find_if(simulations, [&](SimulationFile const& simulation) { return StringHelper::toUtf8(simulation.path) == rawTO->id; });
        _descriptionEditable = findResult == simulations.end() || findResult->descriptionComplete;
    }
}

void EditSimulationDialog::openForFolder(NetworkResourceTreeTO const& treeTO, std::vector<NetworkResourceRawTO> const& rawTOs)
{
    changeTitle("Change folder name");
    AlienDialog::open();
    _treeTO = treeTO;
    _rawTOs = rawTOs;
    _errorMessage.reset();

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
    if (AlienGui::InputText(AlienGui::InputTextParameters().textWidth(0).hint("Name"), _newName)) {
        _errorMessage.reset();
    }
    auto newName = boost::algorithm::trim_copy(_newName);
    auto& service = SimulationLibraryService::get();
    auto isGenome = _treeTO->type == NetworkResourceType_Genome;
    auto isValidLocalName = isGenome ? service.isValidGenomeName(newName) : service.isValidSimulationName(newName);
    auto valid = isLocal() ? isValidLocalName : !_newName.empty();
    if (isLocal() && !valid && !newName.empty() && !_errorMessage.has_value()) {
        _errorMessage = !isGenome && service.isValidFolderName(newName) ? "Names of this form are reserved for autosaves" : getLocalNameHint(newName);
    }
    processErrorMessage();

    AlienGui::Separator();

    ImGui::PushID("description");
    ImGui::BeginGroup();
    ImGui::BeginDisabled(!_descriptionEditable);
    AlienGui::InputTextMultiline(
        AlienGui::InputTextMultilineParameters().hint("Desc (optional)").textWidth(0).height(ImGui::GetContentRegionAvail().y - scale(FooterHeight)),
        _newDescription);
    ImGui::EndDisabled();
    ImGui::EndGroup();
    ImGui::PopID();
    if (!_descriptionEditable) {
        AlienGui::Tooltip(AlienGui::TooltipParameters().text("The description is longer than this field can show. It can be changed in its text file."));
    }

    AlienGui::Separator();

    processButtons(valid);
}

void EditSimulationDialog::processForFolder()
{
    auto valid = false;
    if (ImGui::BeginChild("##Folder", {0, -scale(FooterHeight)})) {
        if (AlienGui::InputText(AlienGui::InputTextParameters().textWidth(0).hint("Folder name"), _newName)) {
            _errorMessage.reset();
        }
        auto newName = boost::algorithm::trim_copy(_newName);
        valid = isLocal() ? SimulationLibraryService::get().isValidFolderName(newName) : !_newName.empty();
        if (isLocal() && !valid && !newName.empty() && !_errorMessage.has_value()) {
            _errorMessage = getLocalNameHint(newName);
        }
        processErrorMessage();
    }
    ImGui::EndChild();

    AlienGui::Separator();

    processButtons(valid);
}

void EditSimulationDialog::processErrorMessage()
{
    if (!_errorMessage.has_value()) {
        return;
    }
    ImGui::PushStyleColor(ImGuiCol_Text, Const::WarningColor.Value);
    ImGui::PushTextWrapPos(0.0f);
    AlienGui::Text(*_errorMessage);
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

void EditSimulationDialog::processButtons(bool valid)
{
    ImGui::BeginDisabled(!valid);
    if (AlienGui::Button("OK")) {
        if (_treeTO->isLeaf()) {
            onApplyToLeaf();
        } else {
            onApplyToFolder();
        }
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
}

bool EditSimulationDialog::isLocal() const
{
    return isLocalWorkspace(BrowserWindow::get().getCurrentWorkspaceType());
}

void EditSimulationDialog::onApplyToLeaf()
{
    auto& rawTO = _treeTO->getLeaf().rawTO;
    if (isLocal()) {
        auto errorMessage = SimulationLibraryController::get().onEdit(
            StringHelper::toPath(rawTO->id), boost::algorithm::trim_copy(_newName), _descriptionEditable ? std::make_optional(_newDescription) : std::nullopt);
        if (errorMessage.has_value()) {
            _errorMessage = errorMessage;
            return;
        }
        close();
        return;
    }

    if (NetworkValidationService::get().isStringValidForDatabase(_newName) && NetworkValidationService::get().isStringValidForDatabase(_newDescription)) {
        EditNetworkResourceRequestData::Entry entry{.resourceId = rawTO->id, .newName = _newName, .newDescription = _newDescription};
        NetworkTransferController::get().onEdit(EditNetworkResourceRequestData{.entries = std::vector{entry}});
        close();
    } else {
        showMessage("Error", Const::NotAllowedCharacters);
    }
}

void EditSimulationDialog::onApplyToFolder()
{
    if (isLocal()) {
        auto& service = SimulationLibraryService::get();
        auto const& controller = SimulationLibraryController::get();
        auto newFolderName = boost::algorithm::trim_copy(_newName);

        // The files keep their names, which differ from the names shown for autosaves
        std::vector<SimulationLibraryController::Renaming> renamings;
        for (auto const& rawTO : _rawTOs) {
            auto simulationPath = StringHelper::toPath(rawTO->id);
            auto name = service.getSimulationName(controller.getMainFolder(simulationPath), simulationPath);
            if (name.has_value() && name->size() > _origFolderName.size()) {
                renamings.emplace_back(simulationPath, newFolderName + name->substr(_origFolderName.size()));
            }
        }
        if (auto errorMessage = SimulationLibraryController::get().onRename(renamings)) {
            _errorMessage = errorMessage;
            return;
        }
        close();
        return;
    }

    if (NetworkValidationService::get().isStringValidForDatabase(_newName)) {
        EditNetworkResourceRequestData requestData;
        for (auto const& rawTO : _rawTOs) {
            auto nameWithoutOldFolder = rawTO->resourceName.substr(_origFolderName.size() + 1);
            auto newName = NetworkResourceService::get().concatenateFolderName({_newName, nameWithoutOldFolder}, false);
            requestData.entries.emplace_back(rawTO->id, newName, rawTO->description);
        }
        NetworkTransferController::get().onEdit(requestData);
        close();
    } else {
        showMessage("Error", Const::NotAllowedCharacters);
    }
}
