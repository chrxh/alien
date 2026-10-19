#include "UploadSimulationDialog.h"

#include <algorithm>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Data/SimulationParameters.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>
#include <Network/NetworkValidationService.h>

#include <Persister/Interface/SerializerService.h>

#include "AlienGui.h"
#include "BrowserWindow.h"
#include "EditorController.h"
#include "GenericMessageDialog.h"
#include "GenomeEditorWindow.h"
#include "HelpStrings.h"
#include "LoginDialog.h"
#include "NetworkTransferController.h"
#include "StyleService.h"
#include "Viewport.h"

namespace
{
    auto constexpr HelpMarkerWidth = 24.0f;
    auto constexpr BottomSpace = 90.0f;
    auto constexpr MinDescriptionHeight = 50.0f;

    std::map<NetworkResourceType, std::string> const BrowserDataTypeToLowerString = {
        {NetworkResourceType_Simulation, "simulation"},
        {NetworkResourceType_Genome, "genome"}};
    std::map<NetworkResourceType, std::string> const BrowserDataTypeToUpperString = {
        {NetworkResourceType_Simulation, "Simulation"},
        {NetworkResourceType_Genome, "Genome"}};
}

namespace
{
    std::string getSuggestedResourceName(NetworkResourceType resourceType, std::string const& folder)
    {
        if (resourceType == NetworkResourceType_Genome) {
            return GenomeEditorWindow::get().getCurrentGenome()._name;
        }
        auto projectName = std::string(_SimulationFacade::get()->getSimulationParameters().projectName.value);
        if (projectName == SimulationParameters().projectName.value) {
            projectName.clear();
        }
        auto userName = NetworkService::get().getLoggedInUserName();
        if (!folder.empty() || !userName.has_value()) {
            return projectName;
        }
        return *userName + "/" + projectName;
    }
}

void UploadSimulationDialog::open(NetworkResourceType resourceType, std::string const& folder, std::string const& resourceName)
{
    if (NetworkService::get().getLoggedInUserName()) {
        changeTitle("Upload " + BrowserDataTypeToLowerString.at(resourceType));
        _resourceType = resourceType;
        _folder = folder;
        _resourceName = resourceName.empty() ? getSuggestedResourceName(resourceType, folder) : resourceName;
        _resourceDescription = _resourceDescriptionByFolder[_folder];
        if (_resourceType == NetworkResourceType_Simulation) {
            _preview.createForSimulation();
        } else {
            _preview.createForGenome(GenomeEditorWindow::get().getCurrentPreviewDescs());
        }
        AlienDialog::open();
    } else {
        LoginDialog::get().open();
    }
}

UploadSimulationDialog::UploadSimulationDialog()
    : AlienDialog("", {640.0f, 810.0f})
{}

void UploadSimulationDialog::processIntern()
{
    auto resourceTypeString = BrowserDataTypeToLowerString.at(_resourceType);

    _preview.process();

    AlienGui::Separator();

    if (!_folder.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
        AlienGui::Text(ICON_FA_FOLDER);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        AlienGui::Text(AlienGui::TextParameters().text(_folder).style(AlienGui::TextStyle::Bold));
        AlienGui::Tooltip(
            AlienGui::TooltipParameters().text("This folder is selected in the browser. The " + resourceTypeString + " will be uploaded into it."));
    }

    AlienGui::InputText(
        AlienGui::InputTextParameters()
            .hint(BrowserDataTypeToUpperString.at(_resourceType) + " name")
            .textWidth(HelpMarkerWidth)
            .tooltip(
                "Use the `/`-notation to upload the " + resourceTypeString
                + " to a folder. Missing folders are created automatically.\nFor instance, the name `Biome/Water world/Initial/Variant 1` creates the nested "
                  "folders `Biome`, `Water world` and `Initial`."),
        _resourceName);

    ImGui::PushID("description");
    AlienGui::InputTextMultiline(
        AlienGui::InputTextMultilineParameters()
            .hint(
                _resourceType == NetworkResourceType_Simulation ? "What happens in it? What should people watch for? (optional)"
                                                                : "What does the creature do? What should people watch for? (optional)")
            .textWidth(0)
            .height(std::max(MinDescriptionHeight, scaleInverse(ImGui::GetContentRegionAvail().y) - BottomSpace)),
        _resourceDescription);
    ImGui::PopID();

    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
    AlienGui::Text(ICON_FA_INFO_CIRCLE);
    ImGui::SameLine();
    AlienGui::Text(
        "Shared " + resourceTypeString + "s appear in the Community workspace and on the ALIEN Discord.\nExperiments and work in progress are welcome.");
    ImGui::PopStyleColor();

    AlienGui::Separator();

    ImGui::BeginDisabled(NetworkResourceService::get().removeFoldersFromName(_resourceName).empty());
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters()
                                   .buttonText(ICON_FA_UPLOAD "  Share with the community")
                                   .highlighted(true)
                                   .frame(true)
                                   .transparentBackground(false)
                                   .tooltip("Everybody can see and download the " + resourceTypeString + " in the Community workspace of the browser."))) {
        onUpload(WorkspaceType_Public);
    }
    ImGui::SetItemDefaultFocus();
    ImGui::SameLine();
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters()
                                   .buttonText(ICON_FA_LOCK "  Keep private")
                                   .transparentBackground(false)
                                   .tooltip(
                                       "Only you can see the " + resourceTypeString
                                       + ", in the Private workspace of the browser. You can still share it later there with Change visibility."))) {
        onUpload(WorkspaceType_Private);
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }

    ImGui::SameLine();
    auto privacyText = std::string("Privacy");
    auto privacyWidth = ImGui::CalcTextSize(privacyText.c_str()).x + ImGui::GetStyle().ItemSpacing.x + ImGui::CalcTextSize(ICON_FA_QUESTION_CIRCLE).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - privacyWidth));
    AlienGui::Text(AlienGui::TextParameters().text(privacyText).style(AlienGui::TextStyle::Decent));
    AlienGui::HelpMarker(
        "The " + resourceTypeString + " file, name and description are stored on the server. It cannot be guaranteed that the data will not be deleted.");
}

void UploadSimulationDialog::onUpload(WorkspaceType workspaceType)
{
    _resourceDescriptionByFolder[_folder] = _resourceDescription;
    if (!NetworkValidationService::get().isStringValidForDatabase(_resourceName)
        || !NetworkValidationService::get().isStringValidForDatabase(_resourceDescription)) {
        showMessage("Error", Const::NotAllowedCharacters);
        return;
    }
    close();

    auto data = [&]() -> std::variant<UploadNetworkResourceRequestData::SimulationData, UploadNetworkResourceRequestData::CreatureData> {
        if (_resourceType == NetworkResourceType_Simulation) {
            return UploadNetworkResourceRequestData::SimulationData{
                .zoom = Viewport::get().getZoomFactor(), .center = Viewport::get().getCenterInWorldPos(), .jpg = _preview.getJpg()};
        } else {
            return UploadNetworkResourceRequestData::CreatureData{.description = GenomeEditorWindow::get().getCurrentGenome(), .jpg = _preview.getJpg()};
        }
    }();
    NetworkTransferController::get().onUpload(UploadNetworkResourceRequestData{
        .folderName = _folder,
        .resourceWithoutFolderName = _resourceName,
        .resourceDescription = _resourceDescription,
        .workspaceType = workspaceType,
        .downloadCache = BrowserWindow::get().getSimulationCache(),
        .data = data});
}
