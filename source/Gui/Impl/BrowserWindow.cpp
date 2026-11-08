#include "BrowserWindow.h"

#include <algorithm>
#include <ranges>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>
#include <Base/Resources.h>
#include <Base/StringHelper.h>
#include <Base/WebLinkHelper.h>

#include <Network/NetworkResourceService.h>
#include <Network/NetworkService.h>

#include <Persister/Interface/PersisterFacade.h>

#include "AlienGui.h"
#include "BrowserController.h"
#include "BrowserDetailsWidget.h"
#include "BrowserGalleryWidget.h"
#include "BrowserHelper.h"
#include "BrowserLoginBannerWidget.h"
#include "BrowserLoginHintWidget.h"
#include "BrowserPictureCache.h"
#include "BrowserTableWidget.h"
#include "BrowserUserListWidget.h"
#include "DelayedExecutionController.h"
#include "EditSimulationDialog.h"
#include "FileTransferController.h"
#include "GenericMessageDialog.h"
#include "GenomeEditorWindow.h"
#include "HelpStrings.h"
#include "LoginDialog.h"
#include "NetworkTransferController.h"
#include "SaveSimulationDialog.h"
#include "SimulationLibraryController.h"
#include "StyleService.h"

namespace
{
    auto constexpr UserTableWidth = 290.0f;
    auto constexpr MinUserTableWidth = 120.0f;
    auto constexpr MinWorkspaceWidth = 640.0f;
    auto constexpr BrowserBottomSpace = 41.0f;

    auto constexpr WorkspaceSelectionWidth = 196.0f;
    auto constexpr DetailsWidth = 320.0f;
    auto constexpr MinResourceViewWidth = 120.0f;
    auto constexpr WorkspaceItemHeight = 28.0f;
    auto constexpr WorkspaceItemRounding = 6.0f;
    auto constexpr WorkspaceItemIconOffset = 10.0f;
    auto constexpr WorkspaceItemTextOffset = 36.0f;
    auto constexpr WorkspaceGroupSpacing = 12.0f;

    auto constexpr FilterWidth = 230.0f;
    auto constexpr MinFilterWidth = 100.0f;

    auto constexpr EmojiPopupScale = 0.66f;  // Relative to the resolution of the emoji images
    auto constexpr EmojiPopupWidth = 255.0f;
    auto constexpr EmojiPopupHeight = 300.0f;
    auto constexpr EmojiPopupSingleRowHeight = 75.0f;
}

BrowserWindow::BrowserWindow()
    : AlienWindow("Browser", "windows.browser", true, true, {172.0f, 78.0f}, {1440.0f, 860.0f})
{}

namespace
{
    std::unordered_map<NetworkResourceType, std::string> const networkResourceTypeToString = {
        {NetworkResourceType_Simulation, std::string("simulations")},
        {NetworkResourceType_Genome, std::string("genomes")}};
    std::unordered_map<WorkspaceType, std::string> const workspaceTypeToString = {
        {WorkspaceType_Public, std::string("public")},
        {WorkspaceType_AlienProject, std::string("alien-project")},
        {WorkspaceType_Private, std::string("private")},
        {WorkspaceType_Local, std::string("local")},
        {WorkspaceType_Autosave, std::string("autosave")},
        {WorkspaceType_Liked, std::string("liked")}};
}

void BrowserWindow::initIntern()
{
    _data = BrowserController::get().getData();
    _pictureCache = _BrowserPictureCache::create();
    _galleryWidget = _BrowserGalleryWidget::create(_data, _pictureCache);
    _tableWidget = _BrowserTableWidget::create(_data);
    _detailsWidget = _BrowserDetailsWidget::create(
        _data,
        _pictureCache,
        _BrowserDetailsWidget::Actions{
            .edit = [this](NetworkResourceTreeTO const& treeTO) { onEditResource(treeTO); },
            .changeVisibility = [this](NetworkResourceTreeTO const& treeTO) { onChangeVisibility(treeTO); },
            .remove = [this](NetworkResourceTreeTO const& treeTO) { onDeleteResource(treeTO); },
            .share = [this](NetworkResourceRawTO const& rawTO) { onShareSimulation(rawTO); },
            .replace = [](NetworkResourceRawTO const& rawTO) { SaveSimulationDialog::get().openForReplacing(rawTO); }});
    _userListWidget = _BrowserUserListWidget::create(_data);
    _loginHintWidget = _BrowserLoginHintWidget::create();
    _loginBannerWidget = _BrowserLoginBannerWidget::create();

    auto& settings = GlobalSettings::get();
    _galleryView = settings.getValue("windows.browser.gallery view", _galleryView);
    auto currentWorkspaceId = _data->getCurrentWorkspaceId();
    currentWorkspaceId.resourceType = settings.getValue("windows.browser.resource type", currentWorkspaceId.resourceType);

    // Earlier versions read the workspace type and know only the workspaces of the server, so the workspaces of this computer have a key of
    // their own
    auto serverWorkspaceType = settings.getValue("windows.browser.workspace type", currentWorkspaceId.workspaceType);
    if (serverWorkspaceType >= 0 && serverWorkspaceType < WorkspaceType_Count) {
        _lastServerWorkspaceType = serverWorkspaceType;
    }
    currentWorkspaceId.workspaceType = settings.getValue("windows.browser.workspace", _lastServerWorkspaceType);
    if (!_data->getWorkspaces().contains(currentWorkspaceId)) {
        currentWorkspaceId = WorkspaceId{NetworkResourceType_Simulation, WorkspaceType_AlienProject};
    }
    _data->setCurrentWorkspaceId(currentWorkspaceId);
    _userTableWidth = settings.getValue("windows.browser.simulators width", scale(UserTableWidth)) * WindowController::get().getContentScaleCorrection();

    auto firstStart = settings.getValue("windows.browser.first start", true);
    BrowserController::get().refresh(firstStart);

    for (auto const& [workspaceId, workspace] : _data->getWorkspaces()) {
        auto initialCollapsedSimulationFolders =
            NetworkResourceService::get().convertFolderNamesToSettings(NetworkResourceService::get().getFolderNames(workspace.rawTOs));
        auto collapsedSimulationFolders = settings.getValue(
            "windows.browser.collapsed folders." + networkResourceTypeToString.at(workspaceId.resourceType) + "."
                + workspaceTypeToString.at(workspaceId.workspaceType),
            initialCollapsedSimulationFolders);
        _data->setCollapsedFolderNames(workspaceId, NetworkResourceService::get().convertSettingsToFolderNames(collapsedSimulationFolders));
    }

    _data->loadLastSessionData();

    EditSimulationDialog::get().setup();
}

void BrowserWindow::shutdownIntern()
{
    auto& settings = GlobalSettings::get();
    settings.setValue("windows.browser.gallery view", _galleryView);
    settings.setValue("windows.browser.resource type", _data->getCurrentWorkspaceId().resourceType);
    settings.setValue("windows.browser.workspace type", _lastServerWorkspaceType);
    settings.setValue("windows.browser.workspace", _data->getCurrentWorkspaceId().workspaceType);
    settings.setValue("windows.browser.first start", false);
    settings.setValue("windows.browser.simulators width", _userTableWidth);
    for (auto const& [workspaceId, workspace] : _data->getWorkspaces()) {
        settings.setValue(
            "windows.browser.collapsed folders." + networkResourceTypeToString.at(workspaceId.resourceType) + "."
                + workspaceTypeToString.at(workspaceId.workspaceType),
            NetworkResourceService::get().convertFolderNamesToSettings(workspace.collapsedFolderNames));
    }
    _galleryWidget->shutdown();
    _data->saveLastSessionData();
}

void BrowserWindow::onRefresh()
{
    BrowserController::get().refresh(true);
}

void BrowserWindow::onPreviewPictureChanged(std::string const& resourceId)
{
    _pictureCache->invalidate(resourceId);
}

void BrowserWindow::showResource(NetworkResourceType resourceType, WorkspaceType workspaceType, std::string const& resourceId)
{
    _data->setCurrentWorkspaceId(WorkspaceId{.resourceType = resourceType, .workspaceType = workspaceType});
    _data->setSelectedTreeTO(nullptr);
    _galleryWidget->resetPage();
    _resourceIdToShow = resourceId;
    setOn(true);
}

WorkspaceType BrowserWindow::getCurrentWorkspaceType() const
{
    return _data->getCurrentWorkspaceId().workspaceType;
}

DownloadCache const& BrowserWindow::getSimulationCache() const
{
    return _data->getDownloadCache();
}

void BrowserWindow::processIntern()
{
    updateLocalWorkspaces();
    if (auto workspaceType = _data->getCurrentWorkspaceId().workspaceType; isServerWorkspace(workspaceType)) {
        _lastServerWorkspaceType = workspaceType;
    }

    if (_resourceIdToShow.has_value() && !BrowserController::get().isRefreshing()) {
        revealResource(*_resourceIdToShow);
        _resourceIdToShow.reset();
    }

    processToolbar();
    _loginBannerWidget->process();

    auto startPos = ImGui::GetCursorScreenPos();

    if (ImGui::BeginChild("##workspaceAndUserList", {0, -scale(5.0f)}, 0, ImGuiWindowFlags_NoScrollbar)) {
        auto maxUserTableWidth = std::max(scale(MinUserTableWidth), ImGui::GetContentRegionAvail().x - scale(MinWorkspaceWidth));
        _userTableWidth = std::clamp(_userTableWidth, scale(MinUserTableWidth), maxUserTableWidth);

        processWorkspace();

        ImGui::SameLine();
        AlienGui::MovableVerticalSeparator(AlienGui::MovableVerticalSeparatorParameters().additive(false).bottomSpace(BrowserBottomSpace), _userTableWidth);

        ImGui::SameLine();
        processUserList();

        processStatusBar();
    }
    ImGui::EndChild();

    // The simulations of this computer do not wait for the server
    if (!isLocalWorkspace(_data->getCurrentWorkspaceId().workspaceType)) {
        processRefreshingScreen({startPos.x, startPos.y});
    }

    processEmojiWindow();

    _data->setScrollToSelection(false);
}

void BrowserWindow::processBackground()
{
    _pictureCache->processPendingRequests();
}

void BrowserWindow::processActivated()
{
    onRefresh();
    SimulationLibraryController::get().refresh();
}

namespace
{
    std::string getAccountChipText()
    {
        auto userName = NetworkService::get().getLoggedInUserName();
        return userName.has_value() ? ICON_FA_USER "  " + *userName + "  " ICON_FA_CARET_DOWN : std::string(ICON_FA_SIGN_IN_ALT "  Log in");
    }
}

void BrowserWindow::processToolbar()
{
    auto isSimulation = _data->getCurrentWorkspaceId().resourceType == NetworkResourceType_Simulation;
    std::string resourceTypeString = isSimulation ? "simulation" : "genome";

    auto saveParameters = isSimulation
        ? AlienGui::ToolbarItemParameters().icon(ICON_FA_SAVE).name("Save simulation").tooltip("Save the simulation on this computer or upload it.")
        : AlienGui::ToolbarItemParameters()
              .icon(ICON_FA_SAVE)
              .name("Save genome")
              .tooltip("Save the genome of the genome editor on this computer or upload it.");
    auto openParameters = isSimulation
        ? AlienGui::ToolbarItemParameters()
              .icon(ICON_FA_FOLDER_OPEN)
              .name("Open simulation")
              .tooltip("Open a simulation file from any folder of this computer.")
              .action([] { FileTransferController::get().onOpenSimulationDialog(); })
        : AlienGui::ToolbarItemParameters()
              .icon(ICON_FA_FOLDER_OPEN)
              .name("Open genome")
              .tooltip("Open a genome file from any folder of this computer.")
              .action(
                  [] { FileTransferController::get().onOpenGenomeDialog([](GenomeDesc const& genome) { GenomeEditorWindow::get().openTab(genome, true); }); });

    std::vector<AlienGui::ToolbarItem> items{
        AlienGui::ToolbarItem::createButton(
            AlienGui::ToolbarItemParameters().icon(ICON_FA_SYNC).name("Refresh").disabled(BrowserController::get().isRefreshing()).action([&] {
                onRefresh();
                SimulationLibraryController::get().refresh();
            })),
        AlienGui::ToolbarItem::createSeparator(),
        AlienGui::ToolbarItem::createButton(openParameters),
        AlienGui::ToolbarItem::createButton(saveParameters.action([&] { onSave(); })),
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
                                                .action([&] { _galleryView = false; })),
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
        AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters().icon(ICON_FA_COMMENTS).name("Open ALIEN Discord server").action([] {
            WebLinkHelper::openInBrowser(Const::DiscordURL);
        }))};

    AlienGui::Toolbar(
        AlienGui::ToolbarParameters().id("Browser").trailing([this] { processAccountChip(); }).trailingWidth(calcAccountChipWidth()).trailingAsButton(true),
        items);
}

void BrowserWindow::processWorkspace()
{
    auto sizeAvailable = ImGui::GetContentRegionAvail();
    if (ImGui::BeginChild(
            "##1",
            ImVec2(sizeAvailable.x - _userTableWidth, sizeAvailable.y - scale(BrowserBottomSpace)),
            false,
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        auto workspaceId = _data->getCurrentWorkspaceId();
        AlienGui::TabBar(
            AlienGui::TabBarParameters().id("##Type").style(AlienGui::TabBarStyle::Group),
            {AlienGui::TabItemParameters().name("Simulations"), AlienGui::TabItemParameters().name("Genomes")},
            workspaceId.resourceType);

        // Genomes have no autosaves
        if (workspaceId.resourceType == NetworkResourceType_Genome && workspaceId.workspaceType == WorkspaceType_Autosave) {
            workspaceId.workspaceType = WorkspaceType_Local;
        }
        _data->setCurrentWorkspaceId(workspaceId);

        auto size = ImGui::GetContentRegionAvail();
        auto selectionWidth = scale(WorkspaceSelectionWidth);
        auto detailsWidth = scale(DetailsWidth);
        auto resourceViewWidth = std::max(scale(MinResourceViewWidth), size.x - selectionWidth - detailsWidth - ImGui::GetStyle().ItemSpacing.x * 2);

        processWorkspaceSelection(selectionWidth, size.y);

        ImGui::SameLine();
        if (ImGui::BeginChild("##resources", {resourceViewWidth, size.y}, 0, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
            processResourceView();
        }
        ImGui::EndChild();

        ImGui::SameLine();
        _detailsWidget->process(detailsWidth, size.y);
    }
    ImGui::EndChild();
}

void BrowserWindow::processWorkspaceSelection(float width, float height)
{
    ImGui::PushStyleColor(ImGuiCol_ChildBg, (ImU32)Const::BrowserPaneColor);
    if (ImGui::BeginChild("##workspaces", {width, height}, ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoScrollbar)) {
        AlienGui::Group(AlienGui::GroupParameters().text("This computer"));
        if (_data->getCurrentWorkspaceId().resourceType == NetworkResourceType_Simulation) {
            processWorkspaceItem(WorkspaceType_Local, ICON_FA_DESKTOP, "My simulations", Const::BrowserLocalWorkspaceTooltip);
            processWorkspaceItem(WorkspaceType_Autosave, ICON_FA_HISTORY, "Autosaves", Const::BrowserAutosaveWorkspaceTooltip);
        } else {
            processWorkspaceItem(WorkspaceType_Local, ICON_FA_DESKTOP, "My genomes", Const::BrowserLocalGenomesWorkspaceTooltip);
        }
        ImGui::Dummy({0, scale(WorkspaceGroupSpacing)});
        AlienGui::Group(AlienGui::GroupParameters().text("Server"));
        processWorkspaceItem(WorkspaceType_AlienProject, ICON_FA_STAR, "Featured", Const::BrowserFeaturedWorkspaceTooltip);
        processWorkspaceItem(WorkspaceType_Public, ICON_FA_GLOBE, "Community", Const::BrowserCommunityWorkspaceTooltip);
        processWorkspaceItem(WorkspaceType_Private, ICON_FA_LOCK, "My uploads", Const::BrowserPrivateWorkspaceTooltip);
        processWorkspaceItem(WorkspaceType_Liked, ICON_FA_HEART, "Liked", Const::BrowserLikedWorkspaceTooltip);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void BrowserWindow::processWorkspaceItem(WorkspaceType workspaceType, std::string const& icon, std::string const& name, std::string const& tooltip)
{
    auto currentWorkspaceId = _data->getCurrentWorkspaceId();
    auto isSelected = currentWorkspaceId.workspaceType == workspaceType;

    auto pos = ImGui::GetCursorScreenPos();
    auto size = ImVec2{ImGui::GetContentRegionAvail().x, scale(WorkspaceItemHeight)};
    ImGui::PushID(workspaceType);
    auto clicked = ImGui::InvisibleButton("##workspace", size);
    auto hovered = ImGui::IsItemHovered();
    ImGui::PopID();
    AlienGui::Tooltip(AlienGui::TooltipParameters().text(tooltip));

    auto drawList = ImGui::GetWindowDrawList();
    auto itemMax = ImVec2{pos.x + size.x, pos.y + size.y};
    auto rounding = scale(WorkspaceItemRounding);
    if (isSelected) {
        drawList->AddRectFilled(pos, itemMax, Const::AccentDeepColor, rounding);
        drawList->AddRect(pos, itemMax, Const::AccentLineColor, rounding);
    } else if (hovered) {
        drawList->AddRectFilled(pos, itemMax, Const::ActionButtonHoveredColor, rounding);
    }

    auto textPosY = pos.y + (size.y - ImGui::GetTextLineHeight()) / 2;
    drawList->AddText({pos.x + scale(WorkspaceItemIconOffset), textPosY}, isSelected ? Const::TextBaseColor : Const::TextDimColor, icon.c_str());
    drawList->AddText({pos.x + scale(WorkspaceItemTextOffset), textPosY}, isSelected ? Const::TextStrongColor : Const::TextBaseColor, name.c_str());
    if (workspaceType != WorkspaceType_Private || NetworkService::get().getLoggedInUserName()) {
        auto count = std::to_string(_data->getWorkspace(WorkspaceId{currentWorkspaceId.resourceType, workspaceType}).rawTOs.size());
        auto countWidth = ImGui::CalcTextSize(count.c_str()).x;
        drawList->AddText(
            {itemMax.x - scale(WorkspaceItemIconOffset) - countWidth, textPosY}, isSelected ? Const::TextDimColor : Const::TextFaintColor, count.c_str());
    }

    if (clicked) {
        selectWorkspace(workspaceType);
    }
}

void BrowserWindow::processResourceView()
{
    processHeader();

    if (isLoginRequired()) {
        auto viewPos = ImGui::GetCursorScreenPos();
        auto viewSize = ImGui::GetContentRegionAvail();
        if (_galleryView) {
            viewSize.y -= BrowserHelper::calcFooterHeight();
        }

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

void BrowserWindow::processHeader()
{
    auto sortingWidth = _galleryView ? _galleryWidget->getSortingWidth() + ImGui::GetStyle().ItemSpacing.x : 0.0f;

    auto filterWidth = std::clamp(scaleInverse(ImGui::GetContentRegionAvail().x - sortingWidth), MinFilterWidth, FilterWidth);
    auto filter = _data->getFilter();
    if (AlienGui::InputFilter(AlienGui::InputFilterParameters().width(filterWidth), filter)) {
        _galleryWidget->resetPage();
        _data->setFilter(filter);
    }

    if (_galleryView) {
        ImGui::SameLine();
        auto offset = ImGui::GetContentRegionAvail().x - _galleryWidget->getSortingWidth();
        if (offset > 0) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);
        }
        _galleryWidget->processSorting();
    }
}

void BrowserWindow::processFooter()
{
    if (!_galleryView) {
        return;
    }
    auto availableWidth = ImGui::GetContentRegionAvail().x;
    auto pagerWidth = _galleryWidget->getPagerWidth();
    if (availableWidth > pagerWidth) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + availableWidth - pagerWidth);
    }
    _galleryWidget->processPaging();
}

bool BrowserWindow::isLoginRequired() const
{
    return _data->getCurrentWorkspaceId().workspaceType == WorkspaceType_Private && !NetworkService::get().getLoggedInUserName();
}

float BrowserWindow::calcAccountChipWidth() const
{
    return scaleInverse(ImGui::CalcTextSize(getAccountChipText().c_str()).x + ImGui::GetStyle().FramePadding.x * 2);
}

void BrowserWindow::processAccountChip()
{
    auto userName = NetworkService::get().getLoggedInUserName();
    auto invitesToLogin = !userName.has_value() && !_loginBannerWidget->isVisible();

    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters()
                                   .buttonText(getAccountChipText())
                                   .highlighted(invitesToLogin)
                                   .frame(invitesToLogin)
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
        auto const& simWorkspace = _data->getWorkspace(WorkspaceId{NetworkResourceType_Simulation, workspaceType});
        auto const& genomeWorkspace = _data->getWorkspace(WorkspaceId{NetworkResourceType_Genome, workspaceType});
        simulations.insert(simWorkspace.rawTOs.begin(), simWorkspace.rawTOs.end());
        genomes.insert(genomeWorkspace.rawTOs.begin(), genomeWorkspace.rawTOs.end());
    }
    auto const& controller = SimulationLibraryController::get();
    auto resourceType = _data->getCurrentWorkspaceId().resourceType;
    auto numLocalResources = _data->getWorkspace(WorkspaceId{resourceType, WorkspaceType_Local}).rawTOs.size();

    statusItems.emplace_back("Server: " + NetworkService::get().getServerAddress());

    if (auto userName = NetworkService::get().getLoggedInUserName()) {
        statusItems.emplace_back("Logged in as " + *userName);
    } else {
        statusItems.emplace_back("Not logged in");
    }
    statusItems.emplace_back(std::to_string(simulations.size()) + " simulations and " + std::to_string(genomes.size()) + " genomes on the server");
    statusItems.emplace_back(
        std::to_string(numLocalResources) + (resourceType == NetworkResourceType_Simulation ? " simulations in My simulations" : " genomes in My genomes"));
    statusItems.emplace_back(std::to_string(_data->getUserTOs().size()) + " simulators found");
    if (controller.isAutosaveEnabled()) {
        statusItems.emplace_back("Next autosave in " + StringHelper::format(controller.getDurationUntilNextAutosave()));
    }

    if (!NetworkService::get().getLoggedInUserName()) {
        statusItems.emplace_back("In order to share simulations you need to log in.");
    }

    auto parameters = AlienGui::StatusBarParameters().infoItems(statusItems);
    if (controller.getContent().truncated) {
        parameters.warningItem("Not all files of this computer are shown")
            .warningTooltip("The folder contains more folders, deeper folders or more files than the browser shows.");
    }
    AlienGui::StatusBar(parameters);
}

void BrowserWindow::processEmojiWindow()
{
    if (_data->isActivateEmojiPopup()) {
        ImGui::OpenPopup("emoji");
        _data->setActivateEmojiPopup(false);
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
    auto const& emoji = _data->getEmojis().at(emojiType);
    ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(Const::ToolbarButtonBackgroundColor));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, static_cast<ImVec4>(Const::ToolbarButtonHoveredColor));
    auto cursorPos = ImGui::GetCursorScreenPos();
    auto emojiWidth = scale(toFloat(emoji.width) * EmojiPopupScale);
    auto emojiHeight = scale(toFloat(emoji.height) * EmojiPopupScale);
    auto leaf = _data->getEmojiPopupTO()->getLeaf();
    ImGui::PushID(emojiType);
    if (ImGui::ImageButton("emoji_popup", emoji.textureId, ImVec2(emojiWidth, emojiHeight), ImVec2(0, 0), ImVec2(1.0f, 1.0f))) {
        _data->onToggleLike(_data->getEmojiPopupTO(), toInt(emojiType));
        ImGui::CloseCurrentPopup();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopID();

    bool isLiked = _data->isOwnReaction(leaf.rawTO->id, emojiType);
    if (isLiked) {
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        auto& style = ImGui::GetStyle();
        drawList->AddRect(
            ImVec2(cursorPos.x, cursorPos.y),
            ImVec2(cursorPos.x + emojiWidth + style.FramePadding.x * 2, cursorPos.y + emojiHeight + style.FramePadding.y * 2),
            (ImU32)Const::BrowserOwnReactionFrameColor,
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

namespace
{
    NetworkResourceRawTO createLocalRawTO(SimulationFile const& simulation, NetworkResourceType resourceType)
    {
        auto result = std::make_shared<_NetworkResourceRawTO>();
        result->id = StringHelper::toUtf8(simulation.path);
        result->timestamp = simulation.timestamp;
        result->resourceName = simulation.name;
        result->contentSize = simulation.fileSize;
        result->description = simulation.description;
        result->version = Const::ProgramVersion;
        result->workspaceType = simulation.kind == SimulationFileKind::Named ? WorkspaceType_Local : WorkspaceType_Autosave;
        result->resourceType = resourceType;

        // An autosave is shown by its time step instead of its file name
        if (simulation.kind != SimulationFileKind::Named && simulation.timestep.has_value()) {
            auto folderNames = NetworkResourceService::get().getFolderNames(simulation.name);
            result->resourceName = NetworkResourceService::get().concatenateFolderName(folderNames, true) + StringHelper::format(*simulation.timestep);
        }
        return result;
    }

    std::vector<std::filesystem::path> getSimulationPaths(std::vector<NetworkResourceRawTO> const& rawTOs)
    {
        std::vector<std::filesystem::path> result;
        for (auto const& rawTO : rawTOs) {
            result.emplace_back(StringHelper::toPath(rawTO->id));
        }
        return result;
    }
}

void BrowserWindow::updateLocalWorkspaces()
{
    auto const& controller = SimulationLibraryController::get();
    if (_libraryRevision == controller.getRevision()) {
        return;
    }
    _libraryRevision = controller.getRevision();

    std::vector<NetworkResourceRawTO> simulations;
    std::vector<NetworkResourceRawTO> autosaves;
    for (auto const& simulation : controller.getContent().simulations) {
        auto& rawTOs = simulation.kind == SimulationFileKind::Named ? simulations : autosaves;
        rawTOs.emplace_back(createLocalRawTO(simulation, NetworkResourceType_Simulation));
    }
    std::vector<NetworkResourceRawTO> genomes;
    for (auto const& genome : controller.getContent().genomes) {
        genomes.emplace_back(createLocalRawTO(genome, NetworkResourceType_Genome));
    }
    _data->setRawTOs(WorkspaceId{NetworkResourceType_Simulation, WorkspaceType_Local}, simulations);
    _data->setRawTOs(WorkspaceId{NetworkResourceType_Simulation, WorkspaceType_Autosave}, autosaves);
    _data->setRawTOs(WorkspaceId{NetworkResourceType_Genome, WorkspaceType_Local}, genomes);

    auto allFiles = simulations;
    allFiles.insert(allFiles.end(), autosaves.begin(), autosaves.end());
    allFiles.insert(allFiles.end(), genomes.begin(), genomes.end());
    _pictureCache->retainLocalPictures(allFiles);
}

void BrowserWindow::selectWorkspace(WorkspaceType workspaceType)
{
    auto workspaceId = _data->getCurrentWorkspaceId();
    workspaceId.workspaceType = workspaceType;
    _data->setCurrentWorkspaceId(workspaceId);

    // Without a selection the workspace itself is described, also if it has been shown before
    _data->setSelectedTreeTO(nullptr);
    _galleryWidget->resetPage();
}

void BrowserWindow::onSave()
{
    auto workspaceId = _data->getCurrentWorkspaceId();
    auto selectedTreeTO = _data->getSelectedTreeTO();

    // The liked resources come from the workspaces of the server, which a selected one tells
    auto targetWorkspaceType = workspaceId.workspaceType;
    if (targetWorkspaceType == WorkspaceType_Liked && selectedTreeTO != nullptr && selectedTreeTO->isLeaf()) {
        targetWorkspaceType = selectedTreeTO->getLeaf().rawTO->workspaceType;
    }
    auto target = [&] {
        switch (targetWorkspaceType) {
        case WorkspaceType_AlienProject:
        case WorkspaceType_Public:
            return SaveTarget::Community;
        case WorkspaceType_Private:
            return SaveTarget::Private;
        default:
            return SaveTarget::ThisComputer;
        }
    }();

    // A selected resource suggests its name, which replaces it if it is an own one, and a selected folder suggests itself
    std::string suggestedName;
    auto suggestsName = workspaceId.workspaceType == WorkspaceType_Liked ? selectedTreeTO != nullptr && selectedTreeTO->isLeaf()
                                                                         : selectedTreeTO != nullptr && workspaceId.workspaceType != WorkspaceType_Autosave;
    if (suggestsName) {
        suggestedName = selectedTreeTO->isLeaf() ? selectedTreeTO->getLeaf().rawTO->resourceName
                                                 : NetworkResourceService::get().concatenateFolderName(selectedTreeTO->folderNames, true);
    }
    SaveSimulationDialog::get().open(workspaceId.resourceType, target, suggestedName);
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

void BrowserWindow::onChangeVisibility(NetworkResourceTreeTO const& treeTO)
{
    auto message = [&]() -> std::string {
        if (!treeTO->isLeaf()) {
            return "Do you really want to change the visibility of everything in the selected folder? Private items are shared with the community, and "
                   "shared items become private. Featured items stay as they are.";
        }
        auto workspaceType = treeTO->getLeaf().rawTO->workspaceType;
        if (workspaceType == WorkspaceType_Private) {
            return "Do you really want to share the selected item with the community? Everyone will be able to see and download it.";
        }
        if (workspaceType == WorkspaceType_AlienProject) {
            return "Do you really want to make the selected item private? It will no longer be visible to others and leaves Featured for good. Sharing "
                   "it again puts it into Community.";
        }
        return "Do you really want to make the selected item private? It will no longer be visible to others.";
    }();
    GenericMessageDialog::get().yesNo("Change visibility", message, [treeTO = treeTO, this]() { onMoveResource(treeTO); });
}

void BrowserWindow::onMoveResource(NetworkResourceTreeTO const& treeTO)
{
    auto const& source = _data->getCurrentWorkspace();
    auto rawTOs = NetworkResourceService::get().getMatchingRawTOs(treeTO, source.rawTOs);

    auto movedRawTOs = _data->changeVisibility(rawTOs, treeTO->isLeaf());
    if (movedRawTOs.empty()) {
        return;
    }

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

    if (isLocalWorkspace(_data->getCurrentWorkspaceId().workspaceType)) {
        auto resourceTypeString = std::string(treeTO->type == NetworkResourceType_Simulation ? "simulation" : "genome");
        auto message = treeTO->isLeaf()
            ? "Do you really want to delete the selected " + resourceTypeString + "? Its file, its picture and its description are removed from this computer."
            : "Do you really want to delete the selected folder with its " + StringHelper::formatCount(rawTOs.size(), resourceTypeString)
                + "? Their files, pictures and descriptions are removed from this computer.";
        GenericMessageDialog::get().yesNo("Delete", message, [simulationPaths = getSimulationPaths(rawTOs)]() {
            // The message dialog closes after this function, which would also close a message about a failure
            delayedExecution([simulationPaths] {
                if (auto errorMessage = SimulationLibraryController::get().onDelete(simulationPaths)) {
                    GenericMessageDialog::get().information("Delete", *errorMessage);
                }
            });
        });
        return;
    }

    auto message = treeTO->isLeaf() ? "Do you really want to delete the selected item?" : "Do you really want to delete the selected folder?";
    GenericMessageDialog::get().yesNo("Delete", message, [rawTOs = rawTOs, this]() {
        _data->removeRawTOs(rawTOs);

        // Apply changes to server
        DeleteNetworkResourceRequestData requestData;
        for (auto const& rawTO : rawTOs) {
            requestData.entries.emplace_back(rawTO->id);
        }
        NetworkTransferController::get().onDelete(requestData);
    });
}

void BrowserWindow::onShareSimulation(NetworkResourceRawTO const& rawTO)
{
    SaveSimulationDialog::get().openForSharing(rawTO->resourceType, StringHelper::toPath(rawTO->id), rawTO->resourceName, rawTO->description);
}

void BrowserWindow::onExpandFolders()
{
    _data->setCollapsedFolderNames(_data->getCurrentWorkspaceId(), {});
}

void BrowserWindow::onCollapseFolders()
{
    _data->setCollapsedFolderNames(_data->getCurrentWorkspaceId(), NetworkResourceService::get().getFolderNames(_data->getCurrentWorkspace().rawTOs, 1));
}

void BrowserWindow::revealResource(std::string const& resourceId)
{
    auto const& workspace = _data->getCurrentWorkspace();
    auto findResult = std::ranges::find_if(workspace.rawTOs, [&](NetworkResourceRawTO const& rawTO) { return rawTO->id == resourceId; });
    if (findResult == workspace.rawTOs.end()) {
        return;
    }
    auto rawTO = *findResult;

    auto collapsedFolderNames = workspace.collapsedFolderNames;
    auto folderNames = NetworkResourceService::get().getFolderNames(rawTO->resourceName);
    for (auto depth : std::views::iota(1, toInt(folderNames.size()) + 1)) {
        collapsedFolderNames.erase(std::vector(folderNames.begin(), folderNames.begin() + depth));
    }
    _data->setCollapsedFolderNames(_data->getCurrentWorkspaceId(), collapsedFolderNames);

    _data->setSelectedTreeTO(BrowserHelper::createLeafTreeTO(rawTO));
    _data->setScrollToSelection(true);
}
