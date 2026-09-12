#include "BrowserWindow.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <ranges>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>
#include <Base/Resources.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>

#include <PersisterInterface/PersisterFacade.h>

#include "AlienGui.h"
#include "BrowserController.h"
#include "BrowserGalleryWidget.h"
#include "BrowserHelper.h"
#include "BrowserLoginBannerWidget.h"
#include "BrowserLoginHintWidget.h"
#include "BrowserTableWidget.h"
#include "BrowserUserListWidget.h"
#include "EditSimulationDialog.h"
#include "GenericMessageDialog.h"
#include "HelpStrings.h"
#include "LoginDialog.h"
#include "NetworkTransferController.h"
#include "ReplaceSimulationDialog.h"
#include "StyleService.h"
#include "UploadSimulationDialog.h"

namespace
{
    auto constexpr UserTableWidth = 300.0f;
    auto constexpr BrowserBottomSpace = 41.0f;

    auto constexpr FilterWidth = 230.0f;
    auto constexpr MinFilterWidth = 100.0f;

    auto constexpr EmojiPopupScale = 0.66f;  // Relative to the resolution of the emoji images
    auto constexpr EmojiPopupWidth = 255.0f;
    auto constexpr EmojiPopupHeight = 300.0f;
    auto constexpr EmojiPopupSingleRowHeight = 75.0f;
}

BrowserWindow::BrowserWindow()
    : AlienWindow("Browser", "windows.browser", true, true, {172.0f, 78.0f}, {1252.0f, 825.0f})
{}

namespace
{
    std::unordered_map<NetworkResourceType, std::string> const networkResourceTypeToString = {
        {NetworkResourceType_Simulation, std::string("simulations")},
        {NetworkResourceType_Genome, std::string("genomes")}};
    std::unordered_map<WorkspaceType, std::string> const workspaceTypeToString = {
        {WorkspaceType_Public, std::string("public")},
        {WorkspaceType_AlienProject, std::string("alien-project")},
        {WorkspaceType_Private, std::string("private")}};
}

void BrowserWindow::initIntern()
{
    _data = BrowserController::get().getData();
    _galleryWidget = _BrowserGalleryWidget::create(_data);
    _tableWidget = _BrowserTableWidget::create(_data);
    _userListWidget = _BrowserUserListWidget::create(_data);
    _loginHintWidget = _BrowserLoginHintWidget::create();
    _loginBannerWidget = _BrowserLoginBannerWidget::create();

    auto& settings = GlobalSettings::get();
    _galleryView = settings.getValue("windows.browser.gallery view", _galleryView);
    _data->currentWorkspace.resourceType = settings.getValue("windows.browser.resource type", _data->currentWorkspace.resourceType);
    _data->currentWorkspace.workspaceType = settings.getValue("windows.browser.workspace type", _data->currentWorkspace.workspaceType);
    _userTableWidth = settings.getValue("windows.browser.user table width", scale(UserTableWidth)) * WindowController::get().getContentScaleCorrection();

    auto firstStart = settings.getValue("windows.browser.first start", true);
    BrowserController::get().refresh(firstStart);

    for (auto& [workspaceId, workspace] : _data->workspaces) {
        auto initialCollapsedSimulationFolders =
            NetworkResourceService::get().convertFolderNamesToSettings(NetworkResourceService::get().getFolderNames(workspace.rawTOs));
        auto collapsedSimulationFolders = settings.getValue(
            "windows.browser.collapsed folders." + networkResourceTypeToString.at(workspaceId.resourceType) + "."
                + workspaceTypeToString.at(workspaceId.workspaceType),
            initialCollapsedSimulationFolders);
        workspace.collapsedFolderNames = NetworkResourceService::get().convertSettingsToFolderNames(collapsedSimulationFolders);
        _data->createTreeTOs(workspace);
    }

    _data->lastSessionData.load(_data->getAllRawTOs());

    EditSimulationDialog::get().setup();
}

void BrowserWindow::shutdownIntern()
{
    auto& settings = GlobalSettings::get();
    settings.setValue("windows.browser.gallery view", _galleryView);
    settings.setValue("windows.browser.resource type", _data->currentWorkspace.resourceType);
    settings.setValue("windows.browser.workspace type", _data->currentWorkspace.workspaceType);
    settings.setValue("windows.browser.first start", false);
    settings.setValue("windows.browser.user table width", _userTableWidth);
    for (auto const& [workspaceId, workspace] : _data->workspaces) {
        settings.setValue(
            "windows.browser.collapsed folders." + networkResourceTypeToString.at(workspaceId.resourceType) + "."
                + workspaceTypeToString.at(workspaceId.workspaceType),
            NetworkResourceService::get().convertFolderNamesToSettings(workspace.collapsedFolderNames));
    }
    _galleryWidget->shutdown();
    _data->lastSessionData.save();
}

void BrowserWindow::onRefresh()
{
    BrowserController::get().refresh(true);
}

void BrowserWindow::onPreviewPictureChanged(std::string const& resourceId)
{
    _galleryWidget->invalidatePicture(resourceId);
}

WorkspaceType BrowserWindow::getCurrentWorkspaceType() const
{
    return _data->currentWorkspace.workspaceType;
}

DownloadCache& BrowserWindow::getSimulationCache()
{
    return _data->downloadCache;
}

void BrowserWindow::processIntern()
{
    processToolbar();
    _loginBannerWidget->process();

    auto startPos = ImGui::GetCursorScreenPos();

    if (ImGui::BeginChild("##workspaceAndUserList", {0, -scale(5.0f)}, 0, ImGuiWindowFlags_NoScrollbar)) {
        processWorkspace();

        ImGui::SameLine();
        AlienGui::MovableVerticalSeparator(AlienGui::MovableVerticalSeparatorParameters().additive(false).bottomSpace(BrowserBottomSpace), _userTableWidth);

        ImGui::SameLine();
        processUserList();

        processStatusBar();
    }
    ImGui::EndChild();

    processRefreshingScreen({startPos.x, startPos.y});

    processEmojiWindow();
}

void BrowserWindow::processBackground()
{
    _galleryWidget->processPendingRequests();
}

void BrowserWindow::processActivated()
{
    onRefresh();
}

namespace
{
    std::string getAccountChipText()
    {
        auto userName = NetworkService::get().getLoggedInUserName();
        return userName.has_value() ? ICON_FA_USER "  " + *userName + "  " ICON_FA_CARET_DOWN : std::string(ICON_FA_SIGN_IN_ALT "  Log in or register");
    }
}

void BrowserWindow::processToolbar()
{
    std::string resourceTypeString = _data->currentWorkspace.resourceType == NetworkResourceType_Simulation ? "simulation" : "genome";
    auto isOwnerForSelectedItem = _data->isOwner(_data->selectedTreeTO);

    std::vector<AlienGui::ToolbarItem> items{
        AlienGui::ToolbarItem::createButton(
            AlienGui::ToolbarItemParameters().icon(ICON_FA_SYNC).name("Refresh").disabled(BrowserController::get().isRefreshing()).action([&] {
                onRefresh();
            })),
        AlienGui::ToolbarItem::createSeparator(),
        AlienGui::ToolbarItem::createButton(
            AlienGui::ToolbarItemParameters()
                .icon(ICON_FA_UPLOAD)
                .name("Upload " + resourceTypeString)
                .tooltip(
                    "Upload your current " + resourceTypeString
                    + " to the server and made visible in the browser. You can choose whether you want to share it with the community or whether it should "
                      "only be visible in your own workspace.\nIf you have already selected a folder, your "
                    + resourceTypeString + " will be uploaded there.")
                .action([&] {
                    std::string prefix = [&] {
                        if (_data->selectedTreeTO == nullptr || _data->selectedTreeTO->isLeaf()) {
                            return std::string();
                        }
                        return NetworkResourceService::get().concatenateFolderName(_data->selectedTreeTO->folderNames, true);
                    }();
                    UploadSimulationDialog::get().open(_data->currentWorkspace.resourceType, prefix);
                })),
        AlienGui::ToolbarItem::createButton(
            AlienGui::ToolbarItemParameters().icon(ICON_FA_EDIT).name("Change name or description").disabled(!isOwnerForSelectedItem).action([&] {
                onEditResource(_data->selectedTreeTO);
            })),
        AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                .icon(ICON_FA_EXCHANGE_ALT)
                                                .name("Replace " + resourceTypeString)
                                                .tooltip(
                                                    "Replace the selected " + resourceTypeString
                                                    + " with the one that is currently open. The name, description and reactions will be preserved.")
                                                .disabled(!isOwnerForSelectedItem || !_data->selectedTreeTO->isLeaf())
                                                .action([&] { onReplaceResource(_data->selectedTreeTO->getLeaf()); })),
        AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                .icon(ICON_FA_SHARE_ALT)
                                                .name("Change visibility")
                                                .tooltip("Change visibility: Community " ICON_FA_LONG_ARROW_ALT_RIGHT
                                                         " my workspace and my workspace " ICON_FA_LONG_ARROW_ALT_RIGHT " Community")
                                                .disabled(!isOwnerForSelectedItem)
                                                .action([&] { onMoveResource(_data->selectedTreeTO); })),
        AlienGui::ToolbarItem::createButton(
            AlienGui::ToolbarItemParameters().icon(ICON_FA_TRASH).name("Delete selected " + resourceTypeString).disabled(!isOwnerForSelectedItem).action([&] {
                onDeleteResource(_data->selectedTreeTO);
            })),
        AlienGui::ToolbarItem::createSeparator(),
        AlienGui::ToolbarItem::createButton(
            AlienGui::ToolbarItemParameters().icon(ICON_FA_EXPAND_ARROWS_ALT).name("Expand all folders").disabled(_galleryView).action([&] {
                onExpandFolders();
            })),
        AlienGui::ToolbarItem::createButton(
            AlienGui::ToolbarItemParameters().icon(ICON_FA_COMPRESS_ARROWS_ALT).name("Collapse all folders").disabled(_galleryView).action([&] {
                onCollapseFolders();
            })),
        AlienGui::ToolbarItem::createSeparator(),
        AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                .icon(ICON_FA_TH)
                                                .name("Gallery view")
                                                .tooltip("Show the " + resourceTypeString + "s as tiles with preview pictures.")
                                                .selected(_galleryView)
                                                .action([&] { _galleryView = true; })),
        AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                .icon(ICON_FA_LIST)
                                                .name("Table view")
                                                .tooltip("Show the " + resourceTypeString + "s as a sortable table with folders.")
                                                .selected(!_galleryView)
                                                .action([&] { _galleryView = false; }))};

#ifdef _WIN32
    items.emplace_back(AlienGui::ToolbarItem::createSeparator());
    items.emplace_back(AlienGui::ToolbarItem::createButton(
        AlienGui::ToolbarItemParameters().icon(ICON_FA_COMMENTS).name("Open ALIEN Discord server").action([&] { openWeblink(Const::DiscordURL); })));
#endif

    auto toolbarParameters = AlienGui::ToolbarParameters().id("Browser");

    if (!_loginBannerWidget->isVisible()) {
        toolbarParameters.trailing([this] { processAccountChip(); }).trailingWidth(calcAccountChipWidth()).trailingAsButton(true);
    }

    AlienGui::Toolbar(toolbarParameters, items);
}

void BrowserWindow::processWorkspace()
{
    auto sizeAvailable = ImGui::GetContentRegionAvail();
    if (ImGui::BeginChild(
            "##1",
            ImVec2(sizeAvailable.x - _userTableWidth, sizeAvailable.y - scale(BrowserBottomSpace)),
            false,
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        if (ImGui::BeginTabBar("##Type", ImGuiTabBarFlags_FittingPolicyResizeDown)) {
            if (ImGui::BeginTabItem("Simulations", nullptr, ImGuiTabItemFlags_None)) {
                if (_data->currentWorkspace.resourceType != NetworkResourceType_Simulation) {
                    _data->currentWorkspace.resourceType = NetworkResourceType_Simulation;
                    _data->selectedTreeTO = nullptr;
                }
                processResourceView();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Genomes", nullptr, ImGuiTabItemFlags_None)) {
                if (_data->currentWorkspace.resourceType != NetworkResourceType_Genome) {
                    _data->currentWorkspace.resourceType = NetworkResourceType_Genome;
                    _data->selectedTreeTO = nullptr;
                }
                processResourceView();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::EndChild();
}

void BrowserWindow::processResourceView()
{
    processWorkspaceSelection();

    if (_galleryView) {
        ImGui::SameLine();
        AlienGui::VerticalSeparator();
        ImGui::SameLine();
        _galleryWidget->processSorting();
    }

    if (isLoginRequired()) {
        auto viewPos = ImGui::GetCursorScreenPos();
        auto viewSize = ImGui::GetContentRegionAvail();
        viewSize.y -= scale(BrowserHelper::WorkspaceBottomSpace);

        if (_galleryView) {
            _galleryWidget->processPlaceholderTiles();
        } else {
            _tableWidget->process();
        }
        _loginHintWidget->process({viewPos.x, viewPos.y}, {viewSize.x, viewSize.y});
    } else if (_galleryView) {
        _galleryWidget->process();
    } else {
        _tableWidget->process();
    }

    processFooter();
}

bool BrowserWindow::isLoginRequired() const
{
    return _data->currentWorkspace.workspaceType == WorkspaceType_Private && !NetworkService::get().getLoggedInUserName();
}

void BrowserWindow::processWorkspaceSelection()
{
    processWorkspaceButton(WorkspaceType_AlienProject, ICON_FA_STAR, "Featured", Const::BrowserFeaturedWorkspaceTooltip);
    ImGui::SameLine();
    processWorkspaceButton(WorkspaceType_Public, ICON_FA_GLOBE, "Community", Const::BrowserCommunityWorkspaceTooltip);
    ImGui::SameLine();
    processWorkspaceButton(WorkspaceType_Private, ICON_FA_LOCK, "Private", Const::BrowserPrivateWorkspaceTooltip);
}

void BrowserWindow::processWorkspaceButton(WorkspaceType workspaceType, std::string const& icon, std::string const& name, std::string const& tooltip)
{
    auto isSelected = _data->currentWorkspace.workspaceType == workspaceType;

    auto label = icon + "  " + name;
    if (workspaceType != WorkspaceType_Private || NetworkService::get().getLoggedInUserName()) {
        label += "  (" + std::to_string(_data->workspaces.at(WorkspaceId{_data->currentWorkspace.resourceType, workspaceType}).rawTOs.size()) + ")";
    }
    label += "##workspace" + std::to_string(workspaceType);

    auto selected = isSelected;
    if (AlienGui::SelectableButton(AlienGui::SelectableButtonParameters().name(label).tooltip(tooltip), selected) && !isSelected) {
        _data->currentWorkspace.workspaceType = workspaceType;
        _data->selectedTreeTO = nullptr;
        _galleryWidget->resetPage();
    }
}

float BrowserWindow::calcAccountChipWidth() const
{
    return scaleInverse(ImGui::CalcTextSize(getAccountChipText().c_str()).x + ImGui::GetStyle().FramePadding.x * 2);
}

void BrowserWindow::processAccountChip()
{
    auto userName = NetworkService::get().getLoggedInUserName();

    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters()
                                   .buttonText(getAccountChipText())
                                   .highlighted(!userName.has_value())
                                   .frame(!userName.has_value())
                                   .transparentBackground(false)
                                   .tooltip(userName.has_value() ? std::string("Show the account menu") : Const::BrowserLoginChipTooltip))) {
        if (userName.has_value()) {
            ImGui::OpenPopup("##account");
        } else {
            LoginDialog::get().open();
        }
    }

    if (ImGui::BeginPopup("##account")) {
        if (ImGui::Selectable(ICON_FA_SIGN_OUT_ALT "  Log out")) {
            NetworkService::get().logout();
            onRefresh();
        }
        ImGui::EndPopup();
    }
}

void BrowserWindow::processFooter()
{
    auto filterWidth = std::clamp(scaleInverse(ImGui::GetContentRegionAvail().x), MinFilterWidth, FilterWidth);
    if (AlienGui::InputFilter(AlienGui::InputFilterParameters().width(filterWidth), _data->filter)) {
        _galleryWidget->resetPage();
        _data->createTreeTOsForAllWorkspaces();
    }

    if (_galleryView) {
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - _galleryWidget->getPagerWidth());
        _galleryWidget->processPaging();
    }
}

void BrowserWindow::processUserList()
{
    auto sizeAvailable = ImGui::GetContentRegionAvail();
    if (ImGui::BeginChild("##2", ImVec2(sizeAvailable.x, sizeAvailable.y - scale(BrowserBottomSpace)), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        _userListWidget->process();
    }
    ImGui::EndChild();
}

void BrowserWindow::processStatusBar()
{
    std::vector<std::string> statusItems;

    std::unordered_set<NetworkResourceRawTO> simulations;
    std::unordered_set<NetworkResourceRawTO> genomes;
    for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
        auto const& simWorkspace = _data->workspaces.at(WorkspaceId{NetworkResourceType_Simulation, workspaceType});
        auto const& genomeWorkspace = _data->workspaces.at(WorkspaceId{NetworkResourceType_Genome, workspaceType});
        simulations.insert(simWorkspace.rawTOs.begin(), simWorkspace.rawTOs.end());
        genomes.insert(genomeWorkspace.rawTOs.begin(), genomeWorkspace.rawTOs.end());
    }

    auto numSimulations = toInt(simulations.size());
    auto numGenomes = toInt(genomes.size());

    statusItems.emplace_back("Server: " + NetworkService::get().getServerAddress());

    if (auto userName = NetworkService::get().getLoggedInUserName()) {
        statusItems.emplace_back("Logged in as " + *userName);
    } else {
        statusItems.emplace_back("Not logged in");
    }
    statusItems.emplace_back(std::to_string(numSimulations) + " simulations found");
    statusItems.emplace_back(std::to_string(numGenomes) + " genomes found");
    statusItems.emplace_back(std::to_string(_data->userTOs.size()) + " simulators found");

    if (!NetworkService::get().getLoggedInUserName()) {
        statusItems.emplace_back("In order to share and upvote simulations you need to log in.");
    }

    AlienGui::StatusBar(statusItems);
}

void BrowserWindow::processEmojiWindow()
{
    if (_data->activateEmojiPopup) {
        ImGui::OpenPopup("emoji");
        _data->activateEmojiPopup = false;
    }
    if (ImGui::BeginPopup("emoji")) {
        ImGui::Text("Choose a reaction");
        ImGui::Spacing();
        ImGui::Spacing();
        if (_showAllEmojis) {
            if (ImGui::BeginChild("##reactionchild", ImVec2(scale(EmojiPopupWidth), scale(EmojiPopupHeight)), false)) {
                int offset = 0;
                for (int i = 0; i < _BrowserData::NumEmojiBlocks; ++i) {
                    for (int j = 0; j < _BrowserData::NumEmojisPerBlock[i]; ++j) {
                        if (j % _BrowserData::NumEmojisPerRow != 0) {
                            ImGui::SameLine();
                        }
                        processEmojiButton(offset + j);
                    }
                    AlienGui::Separator();
                    offset += _BrowserData::NumEmojisPerBlock[i];
                }
            }
            ImGui::EndChild();
        } else {
            if (ImGui::BeginChild("##reactionchild", ImVec2(scale(EmojiPopupWidth), scale(EmojiPopupSingleRowHeight)), false)) {
                for (int i = 0; i < _BrowserData::NumEmojisPerRow; ++i) {
                    if (i % _BrowserData::NumEmojisPerRow != 0) {
                        ImGui::SameLine();
                    }
                    processEmojiButton(i);
                }
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + scale(8.0f));

                if (AlienGui::Button("More", ImGui::GetContentRegionAvail().x)) {
                    _showAllEmojis = true;
                }
            }
            ImGui::EndChild();
        }
        ImGui::EndPopup();
    } else {
        _showAllEmojis = false;
    }
}

void BrowserWindow::processEmojiButton(int emojiType)
{
    auto const& emoji = _data->emojis.at(emojiType);
    ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(Const::ToolbarButtonBackgroundColor));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, static_cast<ImVec4>(Const::ToolbarButtonHoveredColor));
    auto cursorPos = ImGui::GetCursorScreenPos();
    auto emojiWidth = scale(toFloat(emoji.width) * EmojiPopupScale);
    auto emojiHeight = scale(toFloat(emoji.height) * EmojiPopupScale);
    auto leaf = _data->emojiPopupTO->getLeaf();
    ImGui::PushID(emojiType);
    if (ImGui::ImageButton("emoji_popup", (ImTextureID)(intptr_t)emoji.textureId, ImVec2(emojiWidth, emojiHeight), ImVec2(0, 0), ImVec2(1.0f, 1.0f))) {
        _data->onToggleLike(_data->emojiPopupTO, toInt(emojiType));
        ImGui::CloseCurrentPopup();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopID();

    bool isLiked = _data->ownEmojiTypeBySimId.contains(leaf.rawTO->id) && _data->ownEmojiTypeBySimId.at(leaf.rawTO->id) == emojiType;
    if (isLiked) {
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        auto& style = ImGui::GetStyle();
        drawList->AddRect(
            ImVec2(cursorPos.x, cursorPos.y),
            ImVec2(cursorPos.x + emojiWidth + style.FramePadding.x * 2, cursorPos.y + emojiHeight + style.FramePadding.y * 2),
            (ImU32)ImColor::HSV(0, 0, 1, 0.5f),
            1.0f);
    }
}

void BrowserWindow::processRefreshingScreen(RealVector2D const& startPos)
{
    if (BrowserController::get().isRefreshing()) {
        auto size = ImGui::GetItemRectSize();
        auto afterTablePos = ImGui::GetCursorScreenPos();

        ImGui::SetCursorScreenPos({startPos.x, startPos.y});
        if (ImGui::BeginChild("##overlay", {size.x, size.y}, 0, ImGuiWindowFlags_NoScrollbar)) {

            AlienGui::DisabledField();
            AlienGui::Spinner(AlienGui::SpinnerParameters().pos({startPos.x + size.x / 2, startPos.y + size.y / 2}));
        }
        ImGui::EndChild();
        ImGui::SetCursorScreenPos(afterTablePos);
    }
}

void BrowserWindow::onEditResource(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        EditSimulationDialog::get().openForLeaf(treeTO);
    } else {
        auto rawTOs = NetworkResourceService::get().getMatchingRawTOs(treeTO, _data->getCurrentWorkspace().rawTOs);
        EditSimulationDialog::get().openForFolder(treeTO, rawTOs);
    }
}

void BrowserWindow::onReplaceResource(BrowserLeaf const& leaf)
{
    ReplaceSimulationDialog::get().open(_data->currentWorkspace.resourceType, leaf);
}

void BrowserWindow::onMoveResource(NetworkResourceTreeTO const& treeTO)
{
    auto& source = _data->getCurrentWorkspace();
    auto rawTOs = NetworkResourceService::get().getMatchingRawTOs(treeTO, source.rawTOs);

    std::vector<NetworkResourceRawTO> movedRawTOs;
    for (auto const& rawTO : rawTOs) {
        auto& publicWorkspace = _data->workspaces.at(WorkspaceId{_data->currentWorkspace.resourceType, WorkspaceType_Public});
        switch (rawTO->workspaceType) {
        case WorkspaceType_Private: {
            rawTO->workspaceType = WorkspaceType_Public;
            publicWorkspace.rawTOs.emplace_back(rawTO);
            movedRawTOs.emplace_back(rawTO);
        } break;
        case WorkspaceType_Public: {
            rawTO->workspaceType = WorkspaceType_Private;
            auto findResult = std::ranges::find_if(publicWorkspace.rawTOs, [&](NetworkResourceRawTO const& otherRawTO) { return otherRawTO->id == rawTO->id; });
            if (findResult != publicWorkspace.rawTOs.end()) {
                publicWorkspace.rawTOs.erase(findResult);
            }
            movedRawTOs.emplace_back(rawTO);
        } break;
        default:
            break;
        }
    }
    if (movedRawTOs.empty()) {
        return;
    }
    _data->createTreeTOsForCurrentResourceType();

    // Apply changes to server
    MoveNetworkResourceRequestData requestData;
    for (auto const& rawTO : movedRawTOs) {
        requestData.entries.emplace_back(rawTO->id, rawTO->workspaceType);
    }
    NetworkTransferController::get().onMove(requestData);
}

void BrowserWindow::onDeleteResource(NetworkResourceTreeTO const& treeTO)
{
    auto rawTOs = NetworkResourceService::get().getMatchingRawTOs(treeTO, _data->getCurrentWorkspace().rawTOs);

    auto message = treeTO->isLeaf() ? "Do you really want to delete the selected item?" : "Do you really want to delete the selected folder?";
    GenericMessageDialog::get().yesNo("Delete", message, [rawTOs = rawTOs, this]() {
        // Remove resources form workspace
        for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
            auto& workspace = _data->workspaces.at(WorkspaceId{_data->currentWorkspace.resourceType, workspaceType});
            for (auto const& rawTO : rawTOs) {
                auto findResult = std::ranges::find_if(workspace.rawTOs, [&](NetworkResourceRawTO const& otherRawTO) { return otherRawTO->id == rawTO->id; });
                if (findResult != workspace.rawTOs.end()) {
                    workspace.rawTOs.erase(findResult);
                }
            }
            _data->createTreeTOs(workspace);
        }

        // Apply changes to server
        DeleteNetworkResourceRequestData requestData;
        for (auto const& rawTO : rawTOs) {
            requestData.entries.emplace_back(rawTO->id);
        }
        NetworkTransferController::get().onDelete(requestData);
    });
}

void BrowserWindow::onExpandFolders()
{
    auto& workspace = _data->getCurrentWorkspace();
    workspace.collapsedFolderNames.clear();
    _data->createTreeTOs(workspace);
}

void BrowserWindow::onCollapseFolders()
{
    auto& workspace = _data->getCurrentWorkspace();
    workspace.collapsedFolderNames = NetworkResourceService::get().getFolderNames(workspace.rawTOs, 1);
    _data->createTreeTOs(workspace);
}

void BrowserWindow::openWeblink(std::string const& link)
{
#ifdef _WIN32
    ShellExecute(NULL, "open", link.c_str(), NULL, NULL, SW_SHOWNORMAL);
#endif
}
