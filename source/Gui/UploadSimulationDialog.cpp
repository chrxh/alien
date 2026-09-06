#include "UploadSimulationDialog.h"

#include <algorithm>

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <imgui.h>

#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>

#include <Network/NetworkService.h>
#include <Network/NetworkValidationService.h>

#include <PersisterInterface/SerializerService.h>

#include "AlienGui.h"
#include "BrowserWindow.h"
#include "EditorController.h"
#include "GenericMessageDialog.h"
#include "GenomeEditorWindow.h"
#include "HelpStrings.h"
#include "LoginDialog.h"
#include "NetworkTransferController.h"
#include "OpenGLHelper.h"
#include "PictureGuiService.h"
#include "StyleRepository.h"
#include "Viewport.h"

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

void UploadSimulationDialog::initIntern()
{

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
        createPreview();
        AlienDialog::open();
    } else {
        LoginDialog::get().open();
    }
}

UploadSimulationDialog::UploadSimulationDialog()
    : AlienDialog("", {450.0f, 700.0f})
{}

void UploadSimulationDialog::createPreview()
{
    if (_previewTexture.has_value()) {
        glDeleteTextures(1, &_previewTexture->textureId);
        _previewTexture.reset();
    }
    _previewJpg.reset();

    if (_resourceType != NetworkResourceType_Simulation) {
        return;
    }
    _previewJpg = PictureGuiService::get().createSimulationPreviewJpg();
    if (!_previewJpg.has_value()) {
        return;
    }
    try {
        _previewTexture = OpenGLHelper::loadTextureFromMemory(*_previewJpg);
    } catch (std::exception const&) {
        log(Priority::Important, "upload dialog: preview picture could not be decoded");
    }
}

void UploadSimulationDialog::processPreview()
{
    if (!_previewTexture.has_value()) {
        return;
    }

    // Reserving the scrollbar width independently of its visibility avoids a feedback loop between the picture height and the scrollbar
    auto const& style = ImGui::GetStyle();
    auto availableWidth = ImGui::GetWindowWidth() - style.WindowPadding.x * 2 - style.ScrollbarSize;
    auto width = std::min(availableWidth, scale(toFloat(_previewTexture->width)));
    auto height = width * toFloat(_previewTexture->height) / toFloat(_previewTexture->width);
    ImGui::Image((ImTextureID)(intptr_t)_previewTexture->textureId, {width, height});
}

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
            AlienGui::Text(AlienGui::TextParameters().text(_folder).style(AlienGui::TextStyle::Bold));
        }
        ImGui::EndChild();
    }

    processPreview();

    AlienGui::Separator();

    AlienGui::InputText(AlienGui::InputTextParameters().hint(BrowserDataTypeToUpperString.at(_resourceType) + " name").textWidth(0), _resourceName);

    AlienGui::Separator();

    ImGui::PushID("description");
    AlienGui::InputTextMultiline(
        AlienGui::InputTextMultilineParameters()
            .hint("Desc (optional)")
            .textWidth(0)
            .height(ImGui::GetContentRegionAvail().y - StyleRepository::get().scale(70.0f)),
        _resourceDescription);
    ImGui::PopID();

    AlienGui::ToggleButton(
        AlienGui::ToggleButtonParameters()
            .name("Share with the community")
            .tooltip(
                "If true, the " + resourceTypeString + " will be visible to all users in the Community workspace. If false, the " + resourceTypeString
                + " will only be visible in your own workspace. This property can also be changed later if desired."),
        _share);

    AlienGui::Separator();

    ImGui::BeginDisabled(_resourceName.empty());
    if (AlienGui::Button("OK")) {
        if (NetworkValidationService::get().isStringValidForDatabase(_resourceName)
            && NetworkValidationService::get().isStringValidForDatabase(_resourceDescription)) {
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
    auto data = [&]() -> std::variant<UploadNetworkResourceRequestData::SimulationData, UploadNetworkResourceRequestData::CreatureData> {
        if (_resourceType == NetworkResourceType_Simulation) {
            return UploadNetworkResourceRequestData::SimulationData{
                .zoom = Viewport::get().getZoomFactor(), .center = Viewport::get().getCenterInWorldPos(), .jpg = _previewJpg};
        } else {
            return UploadNetworkResourceRequestData::CreatureData{.description = GenomeEditorWindow::get().getCurrentGenome()};
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
