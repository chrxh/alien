#include "BrowserDetailsWidget.h"

#include <algorithm>
#include <filesystem>
#include <numeric>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Resources.h>
#include <Base/StringHelper.h>
#include <Base/WebLinkHelper.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>

#include <Persister/Interface/SimulationLibrary.h>

#include "AlienGui.h"
#include "BrowserData.h"
#include "BrowserHelper.h"
#include "BrowserPictureCache.h"
#include "GenericFileDialog.h"
#include "HelpStrings.h"
#include "SimulationLibraryController.h"
#include "StyleService.h"

namespace
{
    auto constexpr FactsLabelWidth = 96.0f;
    auto constexpr SectionSpacing = 8.0f;
    auto constexpr ChipHeight = 24.0f;
    auto constexpr ChipPaddingX = 7.0f;
    auto constexpr ChipSpacing = 6.0f;
    auto constexpr ChipEmojiSize = 18.0f;
    auto constexpr FolderButtonWidth = 30.0f;
    auto constexpr SettingsTextWidth = 108.0f;  // Fits the longest label of the settings
    auto constexpr VisibilityHeight = 26.0f;
    auto constexpr MaxNumFileInfos = 200;
    auto constexpr TimestampLengthWithoutSeconds = 16;  // e.g. 2026-11-13 14:02

    void addSectionSpacing()
    {
        ImGui::Dummy({0, scale(SectionSpacing)});
    }

    void processLabel(std::string const& text)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
        AlienGui::Text(text);
        ImGui::PopStyleColor();
    }

    void processFact(std::string const& label, std::string const& value)
    {
        processLabel(label);
        ImGui::SameLine(scale(FactsLabelWidth));
        AlienGui::Text(AlienGui::TextParameters().text(value).truncate(true));
    }

    void processWrappedText(std::string const& text)
    {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(text.c_str());
        ImGui::PopTextWrapPos();
    }

    void processStatus(std::string const& icon, std::string const& text)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
        AlienGui::Text(icon);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        processWrappedText(text);
    }

    // An autosave is shown by its time step, which a file that is only named in a similar way is not
    bool isCreatedByAutosave(NetworkResourceRawTO const& rawTO)
    {
        auto stem = StringHelper::toUtf8(StringHelper::toPath(rawTO->id).stem());
        return stem.starts_with(Const::AutosavePrefix) && NetworkResourceService::get().removeFoldersFromName(rawTO->resourceName) != stem;
    }

    // The simulation that ALIEN saves when it is closed and loads when it is started
    bool isSavedOnExit(NetworkResourceRawTO const& rawTO)
    {
        auto simulationPath = StringHelper::toPath(rawTO->id);
        if (!boost::algorithm::iequals(StringHelper::toUtf8(simulationPath.filename()), StringHelper::toUtf8(Const::AutosaveFileWithoutPath))) {
            return false;
        }
        std::error_code error;
        return std::filesystem::equivalent(simulationPath, std::filesystem::current_path() / Const::AutosaveFile, error) && !error;
    }

    std::string getTimestampWithoutSeconds(NetworkResourceRawTO const& rawTO)
    {
        return rawTO->timestamp.substr(0, TimestampLengthWithoutSeconds);
    }

    std::string getByline(NetworkResourceRawTO const& rawTO)
    {
        if (rawTO->workspaceType == WorkspaceType_Local) {
            return "Saved " + getTimestampWithoutSeconds(rawTO);
        }
        if (rawTO->workspaceType == WorkspaceType_Autosave) {
            return (isCreatedByAutosave(rawTO) ? "Autosaved " : "Saved ") + getTimestampWithoutSeconds(rawTO);
        }
        return "by " + rawTO->userName + ", " + getTimestampWithoutSeconds(rawTO);
    }

    bool processIconButton(std::string const& icon, std::string const& tooltip)
    {
        ImGui::SameLine();
        return AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(icon).transparentBackground(false).tooltip(tooltip));
    }

    bool processDeleteButton()
    {
        ImGui::SameLine();
        return AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_TRASH "  Delete").transparentBackground(false));
    }

    struct WorkspaceInfo
    {
        std::string icon;
        std::string name;
        std::string text;
    };

    WorkspaceInfo getWorkspaceInfo(WorkspaceId const& workspaceId)
    {
        if (workspaceId.resourceType == NetworkResourceType_Genome && workspaceId.workspaceType == WorkspaceType_Local) {
            return {ICON_FA_DESKTOP, "My genomes", Const::BrowserLocalGenomesWorkspaceTooltip};
        }
        switch (workspaceId.workspaceType) {
        case WorkspaceType_Local:
            return {ICON_FA_DESKTOP, "My simulations", Const::BrowserLocalWorkspaceTooltip};
        case WorkspaceType_Autosave:
            return {ICON_FA_HISTORY, "Autosaves", Const::BrowserAutosaveWorkspaceTooltip};
        case WorkspaceType_AlienProject:
            return {ICON_FA_STAR, "Featured", Const::BrowserFeaturedWorkspaceTooltip};
        case WorkspaceType_Public:
            return {ICON_FA_GLOBE, "Community", Const::BrowserCommunityWorkspaceTooltip};
        case WorkspaceType_Liked:
            return {ICON_FA_HEART, "Liked", Const::BrowserLikedWorkspaceTooltip};
        default:
            return {ICON_FA_LOCK, "My uploads", Const::BrowserPrivateWorkspaceTooltip};
        }
    }
}

BrowserDetailsWidget _BrowserDetailsWidget::create(BrowserData const& data, BrowserPictureCache const& pictureCache, Actions const& actions)
{
    return BrowserDetailsWidget(new _BrowserDetailsWidget(data, pictureCache, actions));
}

_BrowserDetailsWidget::_BrowserDetailsWidget(BrowserData const& data, BrowserPictureCache const& pictureCache, Actions const& actions)
    : _data(data)
    , _pictureCache(pictureCache)
    , _actions(actions)
{}

void _BrowserDetailsWidget::process(float width, float height)
{
    ImGui::PushID("Details");
    ImGui::PushStyleColor(ImGuiCol_ChildBg, (ImU32)Const::BrowserPaneColor);
    if (ImGui::BeginChild(
            "##details", {width, height}, ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {

        // The actions replace the selection, so the selected object is kept alive here
        auto treeTO = _data->getSelectedTreeTO();
        if (treeTO == nullptr) {
            processWorkspace();
        } else if (treeTO->isLeaf()) {
            processResource(treeTO);
        } else {
            processFolder(treeTO);
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopID();
}

void _BrowserDetailsWidget::processResource(NetworkResourceTreeTO const& treeTO)
{
    auto rawTO = treeTO->getLeaf().rawTO;
    auto& resourceService = NetworkResourceService::get();

    // A scrollbar would change the width of the picture and with it the height of the content
    if (ImGui::BeginChild("##content", {0, -ImGui::GetFrameHeightWithSpacing()}, 0, ImGuiWindowFlags_NoScrollbar)) {
        _pictureCache->request({rawTO});
        BrowserHelper::Picture(_pictureCache->find(rawTO), ImGui::GetContentRegionAvail().x);
        ImGui::Spacing();

        auto folderNames = resourceService.getFolderNames(rawTO->resourceName);
        if (!folderNames.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserLeafTextColor);
            AlienGui::Text(AlienGui::TextParameters().text(resourceService.concatenateFolderName(folderNames, true)).truncate(true));
            ImGui::PopStyleColor();
        }
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserResourceTextColor);
        AlienGui::Text(
            AlienGui::TextParameters().text(resourceService.removeFoldersFromName(rawTO->resourceName)).style(AlienGui::TextStyle::Bold).truncate(true));
        ImGui::PopStyleColor();
        AlienGui::Text(AlienGui::TextParameters().text(getByline(rawTO)).style(AlienGui::TextStyle::Decent).truncate(true));

        if (!rawTO->description.empty()) {
            addSectionSpacing();
            processWrappedText(rawTO->description);
        }

        addSectionSpacing();
        processFacts(rawTO);

        if (!isLocalWorkspace(rawTO->workspaceType)) {
            addSectionSpacing();
            processReactions(treeTO);
            processVisibility(treeTO);
        } else if (rawTO->workspaceType == WorkspaceType_Local) {
            addSectionSpacing();
            if (auto upload = _data->findUpload(rawTO)) {
                if (upload->workspaceType == WorkspaceType_Private) {
                    processStatus(ICON_FA_LOCK, "Uploaded only for you as " + upload->resourceName);
                } else {
                    processStatus(ICON_FA_GLOBE, "Shared with the community as " + upload->resourceName);
                }
            } else {
                processStatus(ICON_FA_DESKTOP, "Only on this computer");
            }
        } else {
            auto& controller = SimulationLibraryController::get();
            if (isSavedOnExit(rawTO)) {
                addSectionSpacing();
                processStatus(ICON_FA_HISTORY, "Saved again whenever ALIEN is closed");
            } else if (
                isCreatedByAutosave(rawTO) && controller.getSaveMode() == SimulationLibraryController::SaveMode_Limited
                && controller.isAutosaveFolder(StringHelper::toPath(rawTO->id).parent_path())) {
                addSectionSpacing();
                processStatus(ICON_FA_HISTORY, "Deleted when " + std::to_string(controller.getNumberOfFiles()) + " newer autosaves exist in its folder");
            }
        }
    }
    ImGui::EndChild();

    processResourceButtons(treeTO);
}

void _BrowserDetailsWidget::processFacts(NetworkResourceRawTO const& rawTO)
{
    if (isLocalWorkspace(rawTO->workspaceType)) {
        if (rawTO->resourceType == NetworkResourceType_Simulation) {
            if (auto const& fileInfo = getFileInfo(rawTO)) {
                processFact("Time step", StringHelper::format(fileInfo->timestep));
                processFact("World size", std::to_string(fileInfo->worldSize.x) + " x " + std::to_string(fileInfo->worldSize.y));
            }
        }
        processFact("File size", StringHelper::formatFileSize(rawTO->contentSize));
        return;
    }

    processFact("Downloads", StringHelper::format(static_cast<uint64_t>(std::max(0, rawTO->numDownloads))));
    if (rawTO->resourceType == NetworkResourceType_Simulation) {
        processFact("World size", std::to_string(rawTO->width) + " x " + std::to_string(rawTO->height));
        processFact("Objects", StringHelper::format(static_cast<uint64_t>(std::max(0, rawTO->particles / 1000))) + " K");
    } else {
        processFact("Cells", StringHelper::format(static_cast<uint64_t>(std::max(0, rawTO->particles))));
    }
    processFact("File size", StringHelper::formatFileSize(rawTO->contentSize));
    processFact("Version", rawTO->version);
}

void _BrowserDetailsWidget::processReactions(NetworkResourceTreeTO const& treeTO)
{
    auto rawTO = treeTO->getLeaf().rawTO;
    processLabel("Reactions");

    auto drawList = ImGui::GetWindowDrawList();
    auto height = scale(ChipHeight);
    auto paddingX = scale(ChipPaddingX);
    auto emojiSize = scale(ChipEmojiSize);
    auto rightEdge = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;

    // Chips continue in the same line as long as they fit
    auto isFirstChip = true;
    auto placeChip = [&](float width) {
        if (!isFirstChip) {
            ImGui::SameLine(0, scale(ChipSpacing));
            if (ImGui::GetCursorScreenPos().x + width > rightEdge) {
                ImGui::NewLine();
            }
        }
        isFirstChip = false;
    };

    std::optional<int> toggledEmojiType;
    for (auto const& [emojiType, numLikes] : rawTO->numLikesByEmojiType) {
        auto hasEmoji = emojiType >= 0 && emojiType < toInt(_data->getEmojis().size());
        auto count = std::to_string(numLikes);
        auto countSize = ImGui::CalcTextSize(count.c_str());
        auto width = paddingX * 2 + countSize.x + (hasEmoji ? emojiSize + paddingX / 2 : 0);
        placeChip(width);

        auto pos = ImGui::GetCursorScreenPos();
        ImGui::PushID(emojiType);
        if (ImGui::InvisibleButton("##reaction", {width, height})) {
            toggledEmojiType = emojiType;
        }
        auto hovered = ImGui::IsItemHovered();
        ImGui::PopID();
        AlienGui::Tooltip(AlienGui::TooltipParameters()
                              .textFunc([this, rawTO, emojiType = emojiType] { return _data->getUserNamesToEmojiType(rawTO, emojiType); })
                              .delay(false));

        auto isOwnReaction = _data->isOwnReaction(rawTO->id, emojiType);
        auto chipMax = ImVec2{pos.x + width, pos.y + height};
        if (isOwnReaction) {
            drawList->AddRectFilled(pos, chipMax, Const::AccentDeepColor, height / 2);
            drawList->AddRect(pos, chipMax, Const::AccentLineColor, height / 2);
        } else {
            drawList->AddRectFilled(pos, chipMax, hovered ? Const::ActionButtonHoveredColor : Const::RaisedColor, height / 2);
        }
        auto textPosX = pos.x + paddingX;
        if (hasEmoji) {
            auto emojiPos = ImVec2{textPosX, pos.y + (height - emojiSize) / 2};
            drawList->AddImage(_data->getEmojis().at(emojiType).textureId, emojiPos, {emojiPos.x + emojiSize, emojiPos.y + emojiSize});
            textPosX += emojiSize + paddingX / 2;
        }
        drawList->AddText({textPosX, pos.y + (height - countSize.y) / 2}, isOwnReaction ? Const::TextStrongColor : Const::TextBaseColor, count.c_str());
    }

    auto plusSize = ImGui::CalcTextSize(ICON_FA_PLUS);
    auto plusWidth = paddingX * 2 + plusSize.x;
    placeChip(plusWidth);
    auto pos = ImGui::GetCursorScreenPos();
    auto plusClicked = ImGui::InvisibleButton("##addReaction", {plusWidth, height});
    auto plusHovered = ImGui::IsItemHovered();
    AlienGui::Tooltip(AlienGui::TooltipParameters().text("Choose a reaction").delay(false));
    drawList->AddRectFilled(pos, {pos.x + plusWidth, pos.y + height}, plusHovered ? Const::ActionButtonHoveredColor : Const::RaisedColor, height / 2);
    drawList->AddText({pos.x + paddingX, pos.y + (height - plusSize.y) / 2}, Const::TextDimColor, ICON_FA_PLUS);

    if (toggledEmojiType.has_value()) {
        _data->onToggleLike(treeTO, *toggledEmojiType);
    }
    if (plusClicked) {
        _data->setActivateEmojiPopup(true);
        _data->setEmojiPopupTO(treeTO);
    }
}

void _BrowserDetailsWidget::processVisibility(NetworkResourceTreeTO const& treeTO)
{
    auto rawTO = treeTO->getLeaf().rawTO;
    if (!NetworkService::get().isLoggedIn() || !_data->isOwner(rawTO)) {
        return;
    }
    addSectionSpacing();
    processLabel("Visibility");

    // A featured upload is visible to the community as well
    auto visibility = rawTO->workspaceType == WorkspaceType_Private ? 1 : 0;
    if (AlienGui::SegmentedControl(
            AlienGui::SegmentedControlParameters().values({ICON_FA_GLOBE "  Community", ICON_FA_LOCK "  Only me"}).height(VisibilityHeight), visibility)) {
        _actions.changeVisibility(treeTO);
    }
}

void _BrowserDetailsWidget::processResourceButtons(NetworkResourceTreeTO const& treeTO)
{
    auto rawTO = treeTO->getLeaf().rawTO;

    if (AlienGui::ActionButton(
            AlienGui::ActionButtonParameters().buttonText(ICON_FA_FOLDER_OPEN "  Open").highlighted(true).frame(true).transparentBackground(false))) {
        _data->onOpenResource(treeTO->getLeaf());
    }

    auto isLocal = isLocalWorkspace(rawTO->workspaceType);
    auto canChange = _data->isOwner(rawTO) && (isLocal || NetworkService::get().isLoggedIn());
    if (canChange && rawTO->workspaceType != WorkspaceType_Autosave) {
        ImGui::SameLine();
        if (AlienGui::ActionButton(
                AlienGui::ActionButtonParameters().buttonText(ICON_FA_EDIT "  Edit").transparentBackground(false).tooltip("Change name or description"))) {
            _actions.edit(treeTO);
        }
    }
    if (rawTO->workspaceType == WorkspaceType_Local) {
        auto resourceTypeString = std::string(rawTO->resourceType == NetworkResourceType_Simulation ? "simulation" : "genome");
        ImGui::SameLine();
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters()
                                       .buttonText(ICON_FA_UPLOAD "  Share")
                                       .transparentBackground(false)
                                       .tooltip("Upload this " + resourceTypeString + ". You choose who can see it."))) {
            _actions.share(rawTO);
        }
    }
    if (!canChange) {
        return;
    }
    if (!isLocal) {
        auto replacement = std::string(rawTO->resourceType == NetworkResourceType_Simulation ? "the current simulation" : "the genome of the genome editor");
        ImGui::SameLine();
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters()
                                       .buttonText(ICON_FA_EXCHANGE_ALT "  Replace")
                                       .transparentBackground(false)
                                       .tooltip("Replace the content of this upload with " + replacement + "."))) {
            _actions.replace(rawTO);
        }
    }
    if (processDeleteButton()) {
        _actions.remove(treeTO);
    }
}

void _BrowserDetailsWidget::processFolder(NetworkResourceTreeTO const& treeTO)
{
    auto const& folder = treeTO->getFolder();
    auto isLocal = isLocalWorkspace(_data->getCurrentWorkspaceId().workspaceType);
    auto isSimulation = treeTO->type == NetworkResourceType_Simulation;

    if (ImGui::BeginChild("##content", {0, -ImGui::GetFrameHeightWithSpacing()}, 0, ImGuiWindowFlags_NoScrollbar)) {
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
        AlienGui::Text(ICON_FA_FOLDER_OPEN);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        AlienGui::Text(AlienGui::TextParameters()
                           .text(NetworkResourceService::get().concatenateFolderName(treeTO->folderNames, false))
                           .style(AlienGui::TextStyle::Bold)
                           .truncate(true));

        addSectionSpacing();
        processFact(isSimulation ? "Simulations" : "Genomes", StringHelper::format(static_cast<uint64_t>(std::max(0, folder.numLeafs))));
        if (!isLocal) {
            processFact("Reactions", StringHelper::format(static_cast<uint64_t>(std::max(0, folder.numReactions))));
        }
    }
    ImGui::EndChild();

    // The liked resources of a folder are only a part of it, so a folder is managed in the workspace it belongs to
    auto isLiked = _data->getCurrentWorkspaceId().workspaceType == WorkspaceType_Liked;
    if (isLiked || !_data->isOwner(treeTO) || (!isLocal && !NetworkService::get().isLoggedIn())) {
        return;
    }
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_EDIT "  Rename").transparentBackground(false))) {
        _actions.edit(treeTO);
    }
    if (!isLocal && _data->getCurrentWorkspaceId().workspaceType != WorkspaceType_AlienProject) {
        if (processIconButton(
                ICON_FA_SHARE_ALT,
                "Change visibility: Community " ICON_FA_LONG_ARROW_ALT_RIGHT " Only me and Only me " ICON_FA_LONG_ARROW_ALT_RIGHT " Community")) {
            _actions.changeVisibility(treeTO);
        }
    }
    if (processDeleteButton()) {
        _actions.remove(treeTO);
    }
}

void _BrowserDetailsWidget::processWorkspace()
{
    auto workspaceId = _data->getCurrentWorkspaceId();
    auto const& workspace = _data->getCurrentWorkspace();
    auto isSimulation = workspaceId.resourceType == NetworkResourceType_Simulation;
    auto info = getWorkspaceInfo(workspaceId);

    // Scrolls if the window is too low for the content
    if (ImGui::BeginChild("##workspace")) {
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDimColor.Value);
        AlienGui::Text(info.icon);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        AlienGui::Text(AlienGui::TextParameters().text(info.name).style(AlienGui::TextStyle::Bold));

        addSectionSpacing();
        processWrappedText(info.text);

        if (workspaceId.workspaceType != WorkspaceType_Private || NetworkService::get().isLoggedIn()) {
            addSectionSpacing();
            processFact(isSimulation ? "Simulations" : "Genomes", StringHelper::format(static_cast<uint64_t>(workspace.rawTOs.size())));
        }
        if (isLocalWorkspace(workspaceId.workspaceType)) {
            auto totalSize =
                std::accumulate(workspace.rawTOs.begin(), workspace.rawTOs.end(), uint64_t(0), [](uint64_t sum, NetworkResourceRawTO const& rawTO) {
                    return sum + rawTO->contentSize;
                });
            processFact("Size", StringHelper::formatFileSize(totalSize));

            addSectionSpacing();
            processFolders();
            if (workspaceId.workspaceType == WorkspaceType_Autosave) {
                addSectionSpacing();
                processAutosaveSettings();
            }
        }
    }
    ImGui::EndChild();
}

void _BrowserDetailsWidget::processFolders()
{
    auto& controller = SimulationLibraryController::get();
    auto const& style = ImGui::GetStyle();
    auto isAutosave = _data->getCurrentWorkspaceId().workspaceType == WorkspaceType_Autosave;
    auto folder = isAutosave ? controller.getAutosaveMainFolder() : controller.getLibraryFolder();
    auto folderText = StringHelper::toUtf8(folder);

    // The end of a long path is shown, because it tells the folders apart
    auto pos = ImGui::GetCursorScreenPos();
    auto fieldWidth = ImGui::GetContentRegionAvail().x - scale(SettingsTextWidth) - scale(FolderButtonWidth) - style.FramePadding.x;
    auto size = ImVec2{std::max(1.0f, fieldWidth), ImGui::GetFrameHeight()};
    ImGui::Dummy(size);
    AlienGui::Tooltip(AlienGui::TooltipParameters().text(folderText));
    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(pos, {pos.x + size.x, pos.y + size.y}, ImGui::GetColorU32(ImGuiCol_FrameBg), style.FrameRounding);
    auto textWidth = ImGui::CalcTextSize(folderText.c_str()).x;
    auto innerWidth = size.x - style.FramePadding.x * 2;
    drawList->PushClipRect({pos.x + style.FramePadding.x, pos.y}, {pos.x + size.x - style.FramePadding.x, pos.y + size.y}, true);
    drawList->AddText(
        {pos.x + style.FramePadding.x + std::min(0.0f, innerWidth - textWidth), pos.y + style.FramePadding.y}, Const::TextBaseColor, folderText.c_str());
    drawList->PopClipRect();

    ImGui::SameLine();
    AlienGui::MoveTickLeft();
    if (AlienGui::Button(ICON_FA_FOLDER_OPEN, FolderButtonWidth)) {
        GenericFileDialog::get().showOpenFileDialog("Select directory", "", folderText, [isAutosave](std::filesystem::path const& path) {
            if (isAutosave) {
                SimulationLibraryController::get().setAutosaveMainFolder(path);
            } else {
                SimulationLibraryController::get().setLibraryFolder(path);
            }
        });
    }
    AlienGui::Tooltip(AlienGui::TooltipParameters().text(
        isAutosave ? "Choose another folder for the autosaves. The files stay where they are."
                   : "Choose another folder for My simulations and My genomes. The files stay where they are."));
    ImGui::SameLine();
    AlienGui::Text("Folder");

    if (isAutosave) {
        processAutosaveFolder();
    }

    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_EXTERNAL_LINK_ALT "  Show in file manager"))) {
        std::error_code error;
        std::filesystem::create_directories(folder, error);
        WebLinkHelper::openInFileManager(folder);
    }
}

void _BrowserDetailsWidget::processAutosaveFolder()
{
    auto& controller = SimulationLibraryController::get();

    // What is typed is kept here, because only valid folders reach the controller. It is read again whenever the field appears.
    auto frameCount = ImGui::GetFrameCount();
    if (frameCount != _lastAutosaveFolderFrame + 1) {
        _autosaveFolderName = controller.getAutosaveFolderName();
    }
    _lastAutosaveFolderFrame = frameCount;

    ImGui::BeginGroup();
    if (AlienGui::InputText(AlienGui::InputTextParameters().name("Subfolder").textWidth(SettingsTextWidth), _autosaveFolderName)) {
        controller.setAutosaveFolderName(_autosaveFolderName);
    }
    ImGui::EndGroup();
    AlienGui::Tooltip(AlienGui::TooltipParameters().text(
        "The subfolder in which the autosaves are stored, for instance one for each simulation run. Use / for nested subfolders. Without a "
        "subfolder, the autosaves are stored in the folder itself."));

    if (boost::algorithm::trim_copy(_autosaveFolderName) != controller.getAutosaveFolderName()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Const::WarningColor.Value);
        processWrappedText("The subfolder is not valid. A name must not contain < > : \" \\ | ? * and must not begin or end with a dot.");
        ImGui::PopStyleColor();
    }
}

void _BrowserDetailsWidget::processAutosaveSettings()
{
    auto& controller = SimulationLibraryController::get();

    ImGui::BeginGroup();
    auto autosaveEnabled = controller.isAutosaveEnabled();
    if (AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Enable autosave").textWidth(SettingsTextWidth), &autosaveEnabled)) {
        controller.setAutosaveEnabled(autosaveEnabled);
    }
    ImGui::EndGroup();
    AlienGui::Tooltip(AlienGui::TooltipParameters().text("Saves the simulation automatically at the interval below, also while it is paused."));

    ImGui::BeginDisabled(!autosaveEnabled);
    auto autosaveInterval = controller.getAutosaveInterval();
    // A held plus or minus button changes the value without reporting it
    if (AlienGui::InputInt(AlienGui::InputIntParameters().name("Interval (min)").textWidth(SettingsTextWidth), autosaveInterval)
        || autosaveInterval != controller.getAutosaveInterval()) {
        controller.setAutosaveInterval(autosaveInterval);
    }
    ImGui::EndDisabled();

    auto saveMode = controller.getSaveMode();
    if (AlienGui::Switcher(
            AlienGui::SwitcherParameters().name("Mode").values({"Limited save files", "Unlimited save files"}).textWidth(SettingsTextWidth), &saveMode)) {
        controller.setSaveMode(saveMode);
    }

    ImGui::BeginDisabled(saveMode == SimulationLibraryController::SaveMode_Unlimited);
    auto numberOfFiles = controller.getNumberOfFiles();
    if (AlienGui::InputInt(AlienGui::InputIntParameters().name("Number of files").textWidth(SettingsTextWidth), numberOfFiles)
        || numberOfFiles != controller.getNumberOfFiles()) {
        controller.setNumberOfFiles(numberOfFiles);
    }
    ImGui::EndDisabled();

    ImGui::Spacing();
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters()
                                   .buttonText(ICON_FA_PLUS "  Autosave now")
                                   .transparentBackground(false)
                                   .tooltip("Save the simulation as an autosave right now"))) {
        controller.onAutosave();
    }
}

std::optional<SimulationFileInfo> const& _BrowserDetailsWidget::getFileInfo(NetworkResourceRawTO const& rawTO)
{
    // A file may have changed whenever the library has been read again
    if (auto libraryRevision = SimulationLibraryController::get().getRevision(); libraryRevision != _libraryRevision) {
        _libraryRevision = libraryRevision;
        _fileInfoByKey.clear();
    }

    // A simulation that is saved again keeps its path
    auto key = rawTO->id + "|" + rawTO->timestamp;
    auto findResult = _fileInfoByKey.find(key);
    if (findResult == _fileInfoByKey.end()) {
        if (toInt(_fileInfoByKey.size()) >= MaxNumFileInfos) {
            _fileInfoByKey.clear();
        }
        SimulationFileInfo fileInfo;
        auto hasFileInfo = SerializerService::get().deserializeSimulationInfoFromFile(fileInfo, StringHelper::toPath(rawTO->id));
        findResult = _fileInfoByKey.emplace(key, hasFileInfo ? std::make_optional(fileInfo) : std::nullopt).first;
    }
    return findResult->second;
}
