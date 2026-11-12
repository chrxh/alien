#include "UploadSimulationDialog.h"

#include <algorithm>
#include <optional>
#include <vector>

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
    auto constexpr PreviewWidth = 450.0f;
    auto constexpr ColumnSpacing = 24.0f;
    auto constexpr FooterHeight = 50.0f;
    auto constexpr MinColumnHeight = 100.0f;
    auto constexpr MinDescriptionHeight = 50.0f;
    auto constexpr SectionSpacing = 10.0f;
    auto constexpr VisibilitySectionHeight = 140.0f;
    auto constexpr VisibilityButtonHeight = 30.0f;
    auto constexpr VisibilityRounding = 8.0f;
    auto constexpr VisibilityInset = 2.0f;

    std::map<NetworkResourceType, std::string> const BrowserDataTypeToLowerString = {
        {NetworkResourceType_Simulation, "simulation"},
        {NetworkResourceType_Genome, "genome"}};
    std::map<NetworkResourceType, std::string> const BrowserDataTypeToUpperString = {
        {NetworkResourceType_Simulation, "Simulation"},
        {NetworkResourceType_Genome, "Genome"}};
}

namespace
{
    std::string getSuggestedNameWithoutFolder(NetworkResourceType resourceType)
    {
        if (resourceType == NetworkResourceType_Genome) {
            return GenomeEditorWindow::get().getCurrentGenome()._name;
        }
        auto projectName = std::string(_SimulationFacade::get()->getSimulationParameters().projectName.value);
        if (projectName == SimulationParameters().projectName.value) {
            return {};
        }
        return projectName;
    }

    std::string getSuggestedResourceName(NetworkResourceType resourceType, std::string const& folder)
    {
        auto name = getSuggestedNameWithoutFolder(resourceType);
        auto userName = NetworkService::get().getLoggedInUserName();
        if (!folder.empty() || !userName.has_value()) {
            return name;
        }
        return *userName + "/" + name;
    }
}

void UploadSimulationDialog::open(NetworkResourceType resourceType, std::string const& folder, std::string const& resourceName)
{
    if (NetworkService::get().getLoggedInUserName()) {
        changeTitle("Upload " + BrowserDataTypeToLowerString.at(resourceType));
        _resourceType = resourceType;
        _folder = folder;
        _resourceName = resourceName.empty() ? getSuggestedResourceName(resourceType, folder) : resourceName;
        _resourceDescription = _resourceDescriptionByTypeAndFolder[{_resourceType, _folder}];
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
    : AlienDialog("", {960.0f, 420.0f})
{}

void UploadSimulationDialog::processIntern()
{
    auto columnHeight = std::max(scale(MinColumnHeight), ImGui::GetContentRegionAvail().y - scale(FooterHeight));
    if (ImGui::BeginChild("##picture", {scale(PreviewWidth), columnHeight}, 0, ImGuiWindowFlags_NoScrollbar)) {
        _preview.process(
            ImGui::GetContentRegionAvail().x,
            _resourceType == NetworkResourceType_Simulation ? ICON_FA_CAMERA "  Current view" : ICON_FA_CAMERA "  Current genome");
    }
    ImGui::EndChild();

    ImGui::SameLine(0, scale(ColumnSpacing));
    if (ImGui::BeginChild("##form", {0, columnHeight})) {
        processForm();
    }
    ImGui::EndChild();

    AlienGui::Separator();
    processFooter();
}

namespace
{
    void processLabel(std::string const& text, std::optional<std::string> const& annotation = std::nullopt)
    {
        auto labelAscent = ImGui::GetFont()->Ascent;
        AlienGui::Text(text);
        if (annotation.has_value()) {
            ImGui::SameLine();
            ImGui::PushFont(StyleService::get().getTinyFont());
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + labelAscent - ImGui::GetFont()->Ascent);
            AlienGui::Text(AlienGui::TextParameters().text(*annotation).style(AlienGui::TextStyle::Decent));
            ImGui::PopFont();
        }
    }

    void addSectionSpacing()
    {
        ImGui::Dummy({0, scale(SectionSpacing)});
    }
}

void UploadSimulationDialog::processForm()
{
    auto resourceTypeString = BrowserDataTypeToLowerString.at(_resourceType);

    processLabel("Name");
    processNameInput();

    addSectionSpacing();
    processLabel("Description", "optional");
    ImGui::PushID("description");
    AlienGui::InputTextMultiline(
        AlienGui::InputTextMultilineParameters()
            .hint(
                _resourceType == NetworkResourceType_Simulation ? "What happens in it? What should people watch for?"
                                                                : "What does the creature do? What should people watch for?")
            .textWidth(0)
            .height(std::max(MinDescriptionHeight, scaleInverse(ImGui::GetContentRegionAvail().y) - VisibilitySectionHeight)),
        _resourceDescription);
    ImGui::PopID();

    addSectionSpacing();
    processLabel("Visibility");
    processVisibilitySelection();

    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    AlienGui::Text(
        _workspaceType == WorkspaceType_Public ? "Everyone can see and download the " + resourceTypeString
                + " in the Community workspace. It is also announced on the ALIEN Discord. Work in progress is welcome."
                                               : "Only you can see the " + resourceTypeString
                + ", in the Private workspace of the browser. You can still share it later there with Change visibility.");
    ImGui::PopTextWrapPos();
}

void UploadSimulationDialog::processNameInput()
{
    auto resourceTypeString = BrowserDataTypeToLowerString.at(_resourceType);
    auto nameParameters = AlienGui::InputTextParameters().hint(BrowserDataTypeToUpperString.at(_resourceType) + " name, use / to create folders").textWidth(0);
    auto nameTooltip = AlienGui::TooltipParameters().text(
        "Use the `/`-notation to upload the " + resourceTypeString
        + " to a folder. Missing folders are created automatically.\nFor instance, the name `Biome/Water world/Initial/Variant 1` creates the nested "
          "folders `Biome`, `Water world` and `Initial`.");
    if (_folder.empty()) {
        AlienGui::InputText(nameParameters, _resourceName);
        AlienGui::Tooltip(nameTooltip);
        return;
    }

    auto const& style = ImGui::GetStyle();
    auto framePos = ImGui::GetCursorScreenPos();
    auto frameEnd = ImVec2{framePos.x + ImGui::GetContentRegionAvail().x, framePos.y + ImGui::GetFrameHeight()};
    ImGui::GetWindowDrawList()->AddRectFilled(framePos, frameEnd, ImGui::GetColorU32(ImGuiCol_FrameBg), style.FrameRounding);

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + style.FramePadding.x);
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
    AlienGui::Text(ICON_FA_FOLDER "  " + _folder + " /");
    ImGui::PopStyleColor();
    AlienGui::Tooltip(AlienGui::TooltipParameters().text("This folder is selected in the browser. The " + resourceTypeString + " will be uploaded into it."));

    ImGui::SameLine(0, 0);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0, 0, 0, 0));
    AlienGui::InputText(nameParameters, _resourceName);
    ImGui::PopStyleColor(3);
    AlienGui::Tooltip(nameTooltip);
}

void UploadSimulationDialog::processVisibilitySelection()
{
    struct VisibilityOption
    {
        WorkspaceType workspaceType;
        std::string label;
    };
    std::vector<VisibilityOption> const options = {{WorkspaceType_Public, ICON_FA_GLOBE "  Community"}, {WorkspaceType_Private, ICON_FA_LOCK "  Private"}};

    auto drawList = ImGui::GetWindowDrawList();
    auto pos = ImGui::GetCursorScreenPos();
    auto width = ImGui::GetContentRegionAvail().x;
    auto height = scale(VisibilityButtonHeight);
    auto rounding = scale(VisibilityRounding);
    auto inset = scale(VisibilityInset);
    drawList->AddRectFilled(pos, {pos.x + width, pos.y + height}, ImGui::GetColorU32(ImGuiCol_FrameBg), rounding);

    auto optionWidth = width / toFloat(options.size());
    for (auto const& option : options) {
        if (&option != &options.front()) {
            ImGui::SameLine(0, 0);
        }
        auto optionPos = ImGui::GetCursorScreenPos();
        ImGui::PushID(toInt(option.workspaceType));
        if (ImGui::InvisibleButton("##visibility", {optionWidth, height})) {
            _workspaceType = option.workspaceType;
        }
        auto hovered = ImGui::IsItemHovered();
        ImGui::PopID();

        auto selected = _workspaceType == option.workspaceType;
        auto optionMin = ImVec2{optionPos.x + inset, optionPos.y + inset};
        auto optionMax = ImVec2{optionPos.x + optionWidth - inset, optionPos.y + height - inset};
        if (selected) {
            drawList->AddRectFilled(optionMin, optionMax, Const::AccentDeepColor, rounding - inset);
            drawList->AddRect(optionMin, optionMax, Const::AccentLineColor, rounding - inset);
        } else if (hovered) {
            drawList->AddRectFilled(optionMin, optionMax, ImGui::GetColorU32(ImGuiCol_FrameBgHovered), rounding - inset);
        }
        auto textSize = ImGui::CalcTextSize(option.label.c_str());
        drawList->AddText(
            {optionPos.x + (optionWidth - textSize.x) / 2, optionPos.y + (height - textSize.y) / 2},
            selected ? Const::TextStrongColor : Const::TextDimColor,
            option.label.c_str());
    }
}

void UploadSimulationDialog::processFooter()
{
    ImGui::BeginDisabled(NetworkResourceService::get().removeFoldersFromName(_resourceName).empty());
    auto buttonText =
        _workspaceType == WorkspaceType_Public ? std::string(ICON_FA_UPLOAD "  Share with the community") : std::string(ICON_FA_LOCK "  Upload privately");
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(buttonText).highlighted(true).frame(true).transparentBackground(false))) {
        onUpload(_workspaceType);
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }

    AlienGui::PrivacyPolicyLink(
        "The " + BrowserDataTypeToLowerString.at(_resourceType)
        + " file, name and description are stored on the server. It cannot be guaranteed that the data will not be deleted.");
}

void UploadSimulationDialog::onUpload(WorkspaceType workspaceType)
{
    _resourceDescriptionByTypeAndFolder[{_resourceType, _folder}] = _resourceDescription;
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
