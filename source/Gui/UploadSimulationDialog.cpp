#include "UploadSimulationDialog.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "PersisterInterface/SerializerService.h"
#include "Network/NetworkService.h"
#include "Network/NetworkValidationService.h"

#include "AlienGui.h"
#include "GenericMessageDialog.h"
#include "StyleRepository.h"
#include "BrowserWindow.h"
#include "EditorController.h"
#include "Viewport.h"
#include "GenomeEditorWindow.h"
#include "HelpStrings.h"
#include "LoginDialog.h"
#include "NetworkTransferController.h"

namespace
{
    auto constexpr FolderWidgetHeight = 50.0f;

    std::map<NetworkResourceType, std::string> const BrowserDataTypeToLowerString = {
        {NetworkResourceType_Simulation, "simulation"},
        {NetworkResourceType_Genome, "genome"}};
    std::map<NetworkResourceType, std::string> const BrowserDataTypeToUpperString = {
        {NetworkResourceType_Simulation, "Simulation"},
        {NetworkResourceType_Genome, "Genome"}};
}

void UploadSimulationDialog::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    auto& settings = GlobalSettings::get();
    _share = settings.getValue("dialogs.upload.share", _share);
}

void UploadSimulationDialog::shutdownIntern()
{
    auto& settings = GlobalSettings::get();
    settings.setValue("dialogs.upload.share", _share);
}


void UploadSimulationDialog::open(NetworkResourceType resourceType, std::string const& folder)
{
    if (NetworkService::get().getLoggedInUserName()) {
        changeTitle("Upload " + BrowserDataTypeToLowerString.at(resourceType));
        _resourceType = resourceType;
        _folder = folder;
        _resourceName = _resourceNameByFolder[_folder];
        _resourceDescription = _resourceDescriptionByFolder[_folder];
        AlienDialog::open();
    } else {
        LoginDialog::get().open();
    }
}

UploadSimulationDialog::UploadSimulationDialog()
    : AlienDialog("")
{}

void UploadSimulationDialog::processIntern()
{
    auto resourceTypeString = BrowserDataTypeToLowerString.at(_resourceType);
    if (ImGui::BeginChild("##header", ImVec2(0, scale(52.0f)), true, ImGuiWindowFlags_HorizontalScrollbar)) {
        AlienGui::Text("Data privacy policy");
        AlienGui::HelpMarker(
            "The " + resourceTypeString + " file, name and description are stored on the server. It cannot be guaranteed that the data will not be deleted.");

        AlienGui::Text("How to use or create folders?");
        AlienGui::HelpMarker(
            "If you want to upload the " + resourceTypeString
            + " to a folder, you can use the `/`-notation. The folder will be created automatically if it does not exist.\nFor instance, naming a simulation "
              "as `Biome/Water "
              "world/Initial/Variant 1` will create the nested folders `Biome`, `Water world` and `Initial`.");
    }
    ImGui::EndChild();

    if (!_folder.empty()) {
        if (ImGui::BeginChild("##folder info", ImVec2(0, scale(85.0f)), true, ImGuiWindowFlags_HorizontalScrollbar)) {
            AlienGui::Text("The following folder has been selected in the browser\nand will used for the upload:\n\n");
            AlienGui::BoldText(_folder);
        }
        ImGui::EndChild();
    }

    AlienGui::Separator();

    AlienGui::InputText(AlienGui::InputTextParameters().hint(BrowserDataTypeToUpperString.at(_resourceType)  + " name").textWidth(0), _resourceName);

    AlienGui::Separator();

    ImGui::PushID("description");
    AlienGui::InputTextMultiline(
        AlienGui::InputTextMultilineParameters()
            .hint("Description (optional)")
            .textWidth(0)
            .height(ImGui::GetContentRegionAvail().y - StyleRepository::get().scale(70.0f)),
        _resourceDescription);
    ImGui::PopID();

    AlienGui::ToggleButton(
        AlienGui::ToggleButtonParameters()
            .name("Make public")
            .tooltip(
                "If true, the " + resourceTypeString + " will be visible to all users. If false, the " + resourceTypeString
                + " will only be visible in the private workspace. This property can also be changed later if desired."),
        _share);

    AlienGui::Separator();

    ImGui::BeginDisabled(_resourceName.empty());
    if (AlienGui::Button("OK")) {
        if (NetworkValidationService::get().isStringValidForDatabase(_resourceName) && NetworkValidationService::get().isStringValidForDatabase(_resourceDescription)) {
            close();
            onUpload();
        } else {
            showMessage("Error", Const::NotAllowedCharacters);
        }
        _resourceNameByFolder[_folder] = _resourceName;
        _resourceDescriptionByFolder[_folder] = _resourceDescription;
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
}

void UploadSimulationDialog::onUpload()
{
    auto data = [&]() -> std::variant<UploadNetworkResourceRequestData::SimulationData, UploadNetworkResourceRequestData::GenomeData> {
        if (_resourceType == NetworkResourceType_Simulation) {
            return UploadNetworkResourceRequestData::SimulationData{.zoom = Viewport::get().getZoomFactor(), .center = Viewport::get().getCenterInWorldPos()};
        } else {
            return UploadNetworkResourceRequestData::GenomeData{.description = GenomeEditorWindow::get().getCurrentGenome()};
        }
    }();
    auto workspaceType = _share ? WorkspaceType_Public : WorkspaceType_Private;
    NetworkTransferController::get().onUpload(UploadNetworkResourceRequestData{
        .folderName = _folder,
        .resourceWithoutFolderName = _resourceName,
        .resourceDescription = _resourceDescription,
        .workspaceType = workspaceType,
        .downloadCache = BrowserWindow::get().getSimulationCache(),
        .data = data});
}
