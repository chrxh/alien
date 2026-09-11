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
#include <PersisterInterface/TaskProcessor.h>

#include "AlienGui.h"
#include "BrowserGalleryWidget.h"
#include "BrowserHelper.h"
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
    auto constexpr RefreshInterval = 20;  // In minutes

    auto constexpr UserTableWidth = 300.0f;
    auto constexpr BrowserBottomSpace = 41.0f;

    auto constexpr WorkspaceSwitcherWidth = 200.0f;
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
    _refreshProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());

    _data = _BrowserData::create();
    _galleryWidget = _BrowserGalleryWidget::create(_data);
    _tableWidget = _BrowserTableWidget::create(_data);
    _userListWidget = _BrowserUserListWidget::create(_data);
    _loginHintWidget = _BrowserLoginHintWidget::create();

    auto& settings = GlobalSettings::get();
    _galleryView = settings.getValue("windows.browser.gallery view", _galleryView);
    _data->currentWorkspace.resourceType = settings.getValue("windows.browser.resource type", _data->currentWorkspace.resourceType);
    _data->currentWorkspace.workspaceType = settings.getValue("windows.browser.workspace type", _data->currentWorkspace.workspaceType);
    _userTableWidth = settings.getValue("windows.browser.user table width", scale(UserTableWidth)) * WindowController::get().getContentScaleCorrection();

    auto firstStart = settings.getValue("windows.browser.first start", true);
    refreshIntern(firstStart);

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
    refreshIntern(true);
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

void BrowserWindow::refreshIntern(bool withRetry)
{
    _refreshProcessor->executeTask(
        [&](auto const& senderId) {
            return _PersisterFacade::get()->scheduleGetNetworkResources(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = withRetry}, GetNetworkResourcesRequestData());
        },
        [&](auto const& requestId) {
            auto data = _PersisterFacade::get()->fetchGetNetworkResourcesData(requestId);
            _data->userTOs = data.userTOs;
            _data->ownEmojiTypeBySimId = data.emojiTypeByResourceId;

            for (auto& [workspaceId, workspace] : _data->workspaces) {
                workspace.rawTOs.clear();
                auto userName = NetworkService::get().getLoggedInUserName().value_or("");
                for (auto const& rawTO : data.resourceTOs) {
                    if (rawTO->resourceType == workspaceId.resourceType) {
                        if ((workspaceId.workspaceType == WorkspaceType_Private && rawTO->userName == userName)
                            || ((workspaceId.workspaceType == WorkspaceType_Public || workspaceId.workspaceType == WorkspaceType_AlienProject)
                                && rawTO->workspaceType == workspaceId.workspaceType)) {
                            workspace.rawTOs.emplace_back(rawTO);
                        }
                    }
                }
                _data->createTreeTOs(workspace);
            }
            _data->sortUserList();
        },
        [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
}

void BrowserWindow::processIntern()
{
    processToolbar();

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
    auto now = std::chrono::steady_clock::now();
    if (!_lastRefreshTime) {
        _lastRefreshTime = now;
    }
    if (std::chrono::duration_cast<std::chrono::minutes>(now - *_lastRefreshTime).count() >= RefreshInterval) {
        _lastRefreshTime = now;
        refreshIntern(false);
    }

    processPendingRequestIds();
}

void BrowserWindow::processActivated()
{
    onRefresh();
}

void BrowserWindow::processToolbar()
{
    std::string resourceTypeString = _data->currentWorkspace.resourceType == NetworkResourceType_Simulation ? "simulation" : "genome";
    auto isOwnerForSelectedItem = _data->isOwner(_data->selectedTreeTO);

    std::vector<AlienGui::ToolbarItem> items{
        AlienGui::ToolbarItem::createButton(
            AlienGui::ToolbarItemParameters().icon(ICON_FA_SYNC).name("Refresh").disabled(_refreshProcessor->pendingTasks()).action([&] { onRefresh(); })),
        AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                .icon(ICON_FA_SIGN_IN_ALT)
                                                .name("Login or register")
                                                .disabled(NetworkService::get().getLoggedInUserName().has_value())
                                                .action([&] { LoginDialog::get().open(); })),
        AlienGui::ToolbarItem::createButton(
            AlienGui::ToolbarItemParameters().icon(ICON_FA_SIGN_OUT_ALT).name("Logout").disabled(!NetworkService::get().getLoggedInUserName()).action([&] {
                NetworkService::get().logout();
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

    AlienGui::Toolbar(AlienGui::ToolbarParameters().id("Browser"), items);
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
        processFilter();
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
}

bool BrowserWindow::isLoginRequired() const
{
    return _data->currentWorkspace.workspaceType == WorkspaceType_Private && !NetworkService::get().getLoggedInUserName();
}

void BrowserWindow::processWorkspaceSelection()
{
    auto userName = NetworkService::get().getLoggedInUserName();
    auto privateWorkspaceString = userName.has_value() ? std::string("Private workspace") : std::string("Private workspace (login required)");
    auto workspaceType_reordered = 2 - _data->currentWorkspace.workspaceType;  // Change the order for display
    if (AlienGui::Switcher(
            AlienGui::SwitcherParameters()
                .width(WorkspaceSwitcherWidth)
                .textWidth(0.0f)
                .tooltip(Const::BrowserWorkspaceTooltip)
                .values({privateWorkspaceString, std::string("Featured"), std::string("Community")}),
            &workspaceType_reordered)) {
        _data->selectedTreeTO = nullptr;
        _galleryWidget->resetPage();
    }
    _data->currentWorkspace.workspaceType = 2 - workspaceType_reordered;
}

void BrowserWindow::processFilter()
{
    ImGui::Spacing();

    auto filterParameters = AlienGui::InputFilterParameters();
    if (_galleryView) {
        auto availableWidth = ImGui::GetContentRegionAvail().x - _galleryWidget->getPagerWidth() - ImGui::GetStyle().ItemSpacing.x;
        filterParameters.width(std::max(MinFilterWidth, scaleInverse(availableWidth)));
    }
    if (AlienGui::InputFilter(filterParameters, _data->filter)) {
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
    if (_refreshProcessor->pendingTasks()) {
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

void BrowserWindow::processPendingRequestIds()
{
    _refreshProcessor->process();
    _data->processPendingRequests();
    _galleryWidget->processPendingRequests();
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
