#include "BrowserWindow.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <ranges>

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/regex.hpp>

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/GlobalSettings.h"
#include "Base/LoggingService.h"
#include "Base/Resources.h"
#include "Base/StringHelper.h"
#include "Base/VersionParserService.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"
#include "PersisterInterface/SerializerService.h"
#include "EngineInterface/SimulationFacade.h"
#include "Network/NetworkResourceService.h"
#include "Network/NetworkService.h"
#include "Network/NetworkResourceParserService.h"
#include "Network/NetworkResourceTreeTO.h"
#include "PersisterInterface/TaskProcessor.h"

#include "AlienImGui.h"
#include "StyleRepository.h"
#include "StatisticsWindow.h"
#include "Viewport.h"
#include "GenericMessageDialog.h"
#include "LoginDialog.h"
#include "UploadSimulationDialog.h"
#include "DelayedExecutionController.h"
#include "EditorController.h"
#include "EditSimulationDialog.h"
#include "OpenGLHelper.h"
#include "OverlayController.h"
#include "GenomeEditorWindow.h"
#include "HelpStrings.h"
#include "LoginController.h"
#include "NetworkTransferController.h"

namespace
{
    auto constexpr RefreshInterval = 20;  //in minutes

    auto constexpr UserTableWidth = 300.0f;
    auto constexpr BrowserBottomSpace = 41.0f;
    auto constexpr WorkspaceBottomSpace = 34.0f;
    auto constexpr RowHeight = 25.0f;

    auto constexpr NumEmojiBlocks = 4;
    int const NumEmojisPerBlock[] = {19, 14, 10, 6};
    auto constexpr NumEmojisPerRow = 5;
}

BrowserWindow::BrowserWindow()
    : AlienWindow("Browser", "windows.browser", true)
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

void BrowserWindow::initIntern(SimulationFacade simulationFacade, PersisterFacade persisterFacade)
{
    _simulationFacade = simulationFacade;
    _persisterFacade = persisterFacade;
    _downloadCache = std::make_shared<_DownloadCache>();

    _refreshProcessor = _TaskProcessor::createTaskProcessor(_persisterFacade);
    _emojiUserNameProcessor = _TaskProcessor::createTaskProcessor(_persisterFacade);
    _reactionProcessor = _TaskProcessor::createTaskProcessor(_persisterFacade);

    auto& settings = GlobalSettings::get();
    _currentWorkspace.resourceType = settings.getValue("windows.browser.resource type", _currentWorkspace.resourceType);
    _currentWorkspace.workspaceType = settings.getValue("windows.browser.workspace type", _currentWorkspace.workspaceType);
    _userTableWidth = settings.getValue("windows.browser.user table width", scale(UserTableWidth));

    int numEmojis = 0;
    for (int i = 0; i < NumEmojiBlocks; ++i) {
        numEmojis += NumEmojisPerBlock[i];
    }
    for (int i = 1; i <= numEmojis; ++i) {
        auto reactionName = "emoji" + std::to_string(i) + ".png";
        _emojis.emplace_back(OpenGLHelper::loadTexture(Const::ResourcePath / std::filesystem::path(reactionName)));
    }
    for (NetworkResourceType resourceType = 0; resourceType < NetworkResourceType_Count; ++resourceType) {
        for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
            _workspaces.emplace(WorkspaceId{resourceType, workspaceType}, Workspace());
        }
    }

    auto firstStart = GlobalSettings::get().getValue("windows.browser.first start", true);
    refreshIntern(firstStart);

    for (auto& [workspaceId, workspace] : _workspaces) {
        auto initialCollapsedSimulationFolders = NetworkResourceService::get().convertFolderNamesToSettings(NetworkResourceService::get().getFolderNames(workspace.rawTOs));
        auto collapsedSimulationFolders = GlobalSettings::get().getValue(
            "windows.browser.collapsed folders." + networkResourceTypeToString.at(workspaceId.resourceType) + "."
                + workspaceTypeToString.at(workspaceId.workspaceType),
            initialCollapsedSimulationFolders);
        workspace.collapsedFolderNames = NetworkResourceService::get().convertSettingsToFolderNames(collapsedSimulationFolders);
        createTreeTOs(workspace);
    }

    _lastSessionData.load(getAllRawTOs());

    EditSimulationDialog::get().setup();
}

void BrowserWindow::shutdownIntern()
{
    auto& settings = GlobalSettings::get();
    settings.setValue("windows.browser.resource type", _currentWorkspace.resourceType);
    settings.setValue("windows.browser.workspace type", _currentWorkspace.workspaceType);
    settings.setValue("windows.browser.first start", false);
    settings.setValue("windows.browser.user table width", _userTableWidth);
    for (auto const& [workspaceId, workspace] : _workspaces) {
        settings.setValue(
            "windows.browser.collapsed folders." + networkResourceTypeToString.at(workspaceId.resourceType) + "."
                + workspaceTypeToString.at(workspaceId.workspaceType),
            NetworkResourceService::get().convertFolderNamesToSettings(workspace.collapsedFolderNames));
    }
    _lastSessionData.save();
}

void BrowserWindow::onRefresh()
{
    refreshIntern(true);
}

WorkspaceType BrowserWindow::getCurrentWorkspaceType() const
{
    return _currentWorkspace.workspaceType;
}

DownloadCache& BrowserWindow::getSimulationCache()
{
    return _downloadCache;
}

void BrowserWindow::refreshIntern(bool withRetry)
{
    _refreshProcessor->executeTask(
        [&](auto const& senderId) {
            return _persisterFacade->scheduleGetNetworkResources(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = withRetry}, GetNetworkResourcesRequestData());
        },
        [&](auto const& requestId) {
            auto data = _persisterFacade->fetchGetNetworkResourcesData(requestId);
            _userTOs = data.userTOs;
            _ownEmojiTypeBySimId = data.emojiTypeByResourceId;

            for (auto& [workspaceId, workspace] : _workspaces) {
                workspace.rawTOs.clear();
                auto userName = NetworkService::get().getLoggedInUserName().value_or("");
                for (auto const& rawTO : data.resourceTOs) {
                    if (rawTO->resourceType == workspaceId.resourceType) {
                        //public user items should also be visible in private workspace
                        if ((workspaceId.workspaceType == WorkspaceType_Private && rawTO->userName == userName
                             && (rawTO->workspaceType == WorkspaceType_Private || rawTO->workspaceType == WorkspaceType_Public))
                            || ((workspaceId.workspaceType == WorkspaceType_Public || workspaceId.workspaceType == WorkspaceType_AlienProject)
                                && rawTO->workspaceType == workspaceId.workspaceType)) {
                            workspace.rawTOs.emplace_back(rawTO);
                        }
                    }
                }
                createTreeTOs(workspace);
            }
            sortUserList();
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
        processVerticalMovableSeparator();

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

void BrowserWindow::processToolbar()
{
    std::string resourceTypeString = _currentWorkspace.resourceType == NetworkResourceType_Simulation ? "simulation" : "genome";
    auto isOwnerForSelectedItem = isOwner(_selectedTreeTO);

    //refresh button
    ImGui::BeginDisabled(_refreshProcessor->pendingTasks());
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_SYNC))) {
        onRefresh();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Refresh");

    //login button
    ImGui::SameLine();
    ImGui::BeginDisabled(NetworkService::get().getLoggedInUserName().has_value());
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_SIGN_IN_ALT))) {
        LoginDialog::get().open();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Login or register");

    //logout button
    ImGui::SameLine();
    ImGui::BeginDisabled(!NetworkService::get().getLoggedInUserName());
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_SIGN_OUT_ALT))) {
        NetworkService::get().logout();
        onRefresh();
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Logout");

    //separator
    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    //upload button
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_UPLOAD))) {
        std::string prefix = [&] {
            if (_selectedTreeTO == nullptr || _selectedTreeTO->isLeaf()) {
                return std::string();
            }
            return NetworkResourceService::get().concatenateFolderName(_selectedTreeTO->folderNames, true);
        }();
        UploadSimulationDialog::get().open(_currentWorkspace.resourceType, prefix);
    }
    AlienImGui::Tooltip(
        "Upload your current " + resourceTypeString
        + " to the server and made visible in the browser. You can choose whether you want to share it with other users or whether it should only be visible "
          "in your private workspace.\nIf you have already selected a folder, your " + resourceTypeString + " will be uploaded there.");

    //edit button
    ImGui::SameLine();
    ImGui::BeginDisabled(!isOwnerForSelectedItem);
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_EDIT))) {
        onEditResource(_selectedTreeTO);
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Change name or description");

    //replace button
    ImGui::SameLine();
    ImGui::BeginDisabled(!isOwnerForSelectedItem || !_selectedTreeTO->isLeaf());
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_EXCHANGE_ALT))) {
        onReplaceResource(_selectedTreeTO->getLeaf());
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Replace the selected " + resourceTypeString + " with the one that is currently open. The name, description and reactions will be preserved.");

    //move to other workspace button
    ImGui::SameLine();
    ImGui::BeginDisabled(!isOwnerForSelectedItem);
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_SHARE_ALT))) {
        onMoveResource(_selectedTreeTO);
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Change visibility: public " ICON_FA_LONG_ARROW_ALT_RIGHT " private and private " ICON_FA_LONG_ARROW_ALT_RIGHT " public");

    //delete button
    ImGui::SameLine();
    ImGui::BeginDisabled(!isOwnerForSelectedItem);
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_TRASH))) {
        onDeleteResource(_selectedTreeTO);
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Delete selected " + resourceTypeString);

    //separator
    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    //expand button
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_EXPAND_ARROWS_ALT))) {
        onExpandFolders();
    }
    AlienImGui::Tooltip("Expand all folders");

    //collapse button
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_COMPRESS_ARROWS_ALT))) {
        onCollapseFolders();
    }
    AlienImGui::Tooltip("Collapse all folders");

#ifdef _WIN32
    //separator
    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    //Discord button
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_COMMENTS))) {
        openWeblink(Const::DiscordURL);
    }
    AlienImGui::Tooltip("Open ALIEN Discord server");
#endif

    AlienImGui::Separator();
}

void BrowserWindow::processWorkspace()
{
    auto sizeAvailable = ImGui::GetContentRegionAvail();
    if (ImGui::BeginChild(
            "##1",
            ImVec2(sizeAvailable.x - scale(_userTableWidth), sizeAvailable.y - scale(BrowserBottomSpace)),
            false,
            ImGuiWindowFlags_HorizontalScrollbar)) {
        if (ImGui::BeginTabBar("##Type", ImGuiTabBarFlags_FittingPolicyResizeDown)) {
            if (ImGui::BeginTabItem("Simulations", nullptr, ImGuiTabItemFlags_None)) {
                if (_currentWorkspace.resourceType != NetworkResourceType_Simulation) {
                    _currentWorkspace.resourceType = NetworkResourceType_Simulation;
                    _selectedTreeTO = nullptr;
                }
                processSimulationList();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Genomes", nullptr, ImGuiTabItemFlags_None)) {
                if (_currentWorkspace.resourceType != NetworkResourceType_Genome) {
                    _currentWorkspace.resourceType = NetworkResourceType_Genome;
                    _selectedTreeTO = nullptr;
                }
                processGenomeList();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        processWorkspaceSelectionAndFilter();
    }
    ImGui::EndChild();
}

void BrowserWindow::processWorkspaceSelectionAndFilter()
{
    ImGui::Spacing();
    if (ImGui::BeginTable("##", 2, 0, ImVec2(-1, 0))) {
        ImGui::TableSetupColumn("##workspaceType");
        ImGui::TableSetupColumn("##textFilter");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        auto userName = NetworkService::get().getLoggedInUserName();
        auto privateWorkspaceString = userName.has_value() ? *userName + "'s private workspace" : "Private workspace (need to login)";
        auto workspaceType_reordered = 2 - _currentWorkspace.workspaceType;  //change the order for display
        if (AlienImGui::Switcher(
                AlienImGui::SwitcherParameters()
                    .textWidth(48.0f)
                    .tooltip(Const::BrowserWorkspaceTooltip)
                    .values({privateWorkspaceString, std::string("alien-project's workspace"), std::string("Public workspace")}),
                workspaceType_reordered)) {
            _selectedTreeTO = nullptr;
        }
        _currentWorkspace.workspaceType = 2 - workspaceType_reordered;
        ImGui::SameLine();
        AlienImGui::VerticalSeparator();

        ImGui::TableSetColumnIndex(1);
        if (AlienImGui::InputFilter(AlienImGui::InputFilterParameters(), _filter)) {
            for (NetworkResourceType resourceType = 0; resourceType < NetworkResourceType_Count; ++resourceType) {
                for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
                    createTreeTOs(_workspaces.at(WorkspaceId{resourceType, workspaceType}));
                }
            }
        }

        ImGui::EndTable();
    }
}

void BrowserWindow::processVerticalMovableSeparator()
{
    auto sizeAvailable = ImGui::GetContentRegionAvail();
    ImGui::Button("##MovableSeparator", ImVec2(scale(5.0f), sizeAvailable.y - scale(BrowserBottomSpace)));
    if (ImGui::IsItemActive()) {
        _userTableWidth -= ImGui::GetIO().MouseDelta.x;
    }
}

namespace
{
    std::string getGpuString(std::string const& gpu)
    {
        if (gpu.substr(0, 6) == "NVIDIA") {
            return gpu.substr(7);
        }
        return gpu;
    }
}

void BrowserWindow::processUserList()
{
    auto sizeAvailable = ImGui::GetContentRegionAvail();
    if (ImGui::BeginChild("##2", ImVec2(sizeAvailable.x, sizeAvailable.y - scale(BrowserBottomSpace)), false, ImGuiWindowFlags_HorizontalScrollbar)) {


        ImGui::PushID("User list");
        auto& styleRepository = StyleRepository::get();
        static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
            | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

        if (ImGui::BeginTabBar("##Simulators", ImGuiTabBarFlags_FittingPolicyResizeDown)) {
            if (ImGui::BeginTabItem("Simulators", nullptr, ImGuiTabItemFlags_None)) {

                if (ImGui::BeginTable("Browser", 5, flags, ImVec2(-1, -1), 0.0f)) {
                    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed, scale(90.0f));
                    auto isLoggedIn = NetworkService::get().getLoggedInUserName().has_value();
                    ImGui::TableSetupColumn(
                        isLoggedIn ? "GPU model" : "GPU (visible if logged in)",
                        ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed,
                        styleRepository.scale(200.0f));
                    ImGui::TableSetupColumn("Time spent", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(80.0f));
                    ImGui::TableSetupColumn(
                        "Reactions received",
                        ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending,
                        scale(120.0f));
                    ImGui::TableSetupColumn(
                        "Reactions given", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(100.0f));
                    ImGui::TableSetupScrollFreeze(0, 1);
                    ImGui::TableHeadersRow();
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

                    ImGuiListClipper clipper;
                    clipper.Begin(_userTOs.size());
                    while (clipper.Step()) {
                        for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                            auto item = &_userTOs[row];

                            ImGui::PushID(row);
                            ImGui::TableNextRow(0, scale(RowHeight));

                            ImGui::TableNextColumn();
                            auto isBoldFont = isLoggedIn && *NetworkService::get().getLoggedInUserName() == item->userName;

                            if (item->online) {
                                AlienImGui::OnlineSymbol();
                                ImGui::SameLine();
                            } else if (item->lastDayOnline) {
                                AlienImGui::LastDayOnlineSymbol();
                                ImGui::SameLine();
                            }
                            processShortenedText(item->userName, isBoldFont);

                            ImGui::TableNextColumn();
                            if (isLoggedIn && LoginController::get().shareGpuInfo()) {
                                processShortenedText(getGpuString(item->gpu), isBoldFont);
                            }

                            ImGui::TableNextColumn();
                            if (item->timeSpent > 0) {
                                processShortenedText(StringHelper::format(item->timeSpent) + "h", isBoldFont);
                            }

                            ImGui::TableNextColumn();
                            processShortenedText(std::to_string(item->starsReceived), isBoldFont);

                            ImGui::TableNextColumn();
                            processShortenedText(std::to_string(item->starsGiven), isBoldFont);

                            ImGui::PopID();
                        }
                    }
                    ImGui::EndTable();
                }
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::PopID();
    }
    ImGui::EndChild();
}

void BrowserWindow::processStatusBar()
{
    std::vector<std::string> statusItems;

    std::unordered_set<NetworkResourceRawTO> simulations;
    std::unordered_set<NetworkResourceRawTO> genomes;
    for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
        auto const& simWorkspace = _workspaces.at(WorkspaceId{NetworkResourceType_Simulation, workspaceType});
        auto const& genomeWorkspace = _workspaces.at(WorkspaceId{NetworkResourceType_Genome, workspaceType});
        simulations.insert(simWorkspace.rawTOs.begin(), simWorkspace.rawTOs.end());
        genomes.insert(genomeWorkspace.rawTOs.begin(), genomeWorkspace.rawTOs.end());
    }

    auto numSimulations = toInt(simulations.size());
    auto numGenomes = toInt(genomes.size());

    statusItems.emplace_back(std::to_string(numSimulations) + " simulations found");
    statusItems.emplace_back(std::to_string(numGenomes) + " genomes found");
    statusItems.emplace_back(std::to_string(_userTOs.size()) + " simulators found");

    if (auto userName = NetworkService::get().getLoggedInUserName()) {
        statusItems.emplace_back("Logged in as " + *userName + " @ " + NetworkService::get().getServerAddress());
    } else {
        statusItems.emplace_back("Not logged in to " + NetworkService::get().getServerAddress());
    }

    if (!NetworkService::get().getLoggedInUserName()) {
        statusItems.emplace_back("In order to share and upvote simulations you need to log in.");
    }

    AlienImGui::StatusBar(statusItems);
}

void BrowserWindow::processSimulationList()
{
    ImGui::PushID("SimulationList");
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable
        | ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV
        | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTable("Browser", 11, flags, ImVec2(-1, -scale(WorkspaceBottomSpace)), 0.0f)) {
        ImGui::TableSetupColumn("Simulation", ImGuiTableColumnFlags_WidthFixed, scale(210.0f), NetworkResourceColumnId_SimulationName);
        ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthFixed, scale(200.0f), NetworkResourceColumnId_Description);
        ImGui::TableSetupColumn("Reactions", ImGuiTableColumnFlags_WidthFixed, scale(140.0f), NetworkResourceColumnId_Likes);
        ImGui::TableSetupColumn(
            "Timestamp",
            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending,
            scale(135.0f),
            NetworkResourceColumnId_Timestamp);
        ImGui::TableSetupColumn("User name", ImGuiTableColumnFlags_WidthFixed, scale(120.0f), NetworkResourceColumnId_UserName);
        ImGui::TableSetupColumn("Downloads", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_NumDownloads);
        ImGui::TableSetupColumn("Width", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_Width);
        ImGui::TableSetupColumn("Height", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_Height);
        ImGui::TableSetupColumn("Objects", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_Particles);
        ImGui::TableSetupColumn("File size", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_FileSize);
        ImGui::TableSetupColumn("Version", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_Version);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

        //create treeTOs if sorting changed
        if (auto sortSpecs = ImGui::TableGetSortSpecs()) {
            if (sortSpecs->SpecsDirty) {
                for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
                    auto& workspace = _workspaces.at(WorkspaceId{NetworkResourceType_Simulation, workspaceType});
                    workspace.sortSpecs.clear();
                    for (int i = 0; i < sortSpecs->SpecsCount; ++i) {
                        workspace.sortSpecs.emplace_back(sortSpecs->Specs[i]);
                    }
                    createTreeTOs(workspace);
                }
                sortSpecs->SpecsDirty = false;
            }
        }

        //process treeTOs
        auto& workspace = _workspaces.at(_currentWorkspace);
        auto scheduleRecreateTreeTOs = false;

        ImGuiListClipper clipper;
        clipper.Begin(workspace.treeTOs.size());
        while (clipper.Step())
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto treeTO = workspace.treeTOs.at(row);
                if (treeTO->isLeaf()) {
                    _lastSessionData.registrate(treeTO->getLeaf().rawTO);
                }

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(RowHeight));
                ImGui::TableNextColumn();

                auto selected = _selectedTreeTO == treeTO;
                if (ImGui::Selectable(
                        "",
                        &selected,
                        ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
                        ImVec2(0, scale(RowHeight) - ImGui::GetStyle().FramePadding.y))) {
                    _selectedTreeTO = selected ? treeTO : nullptr;
                }
                ImGui::SameLine();

                pushTextColor(treeTO);

                if (processResourceNameField(treeTO, workspace.collapsedFolderNames)) {
                    scheduleRecreateTreeTOs = true;
                }
                ImGui::TableNextColumn();
                processDescriptionField(treeTO);
                ImGui::TableNextColumn();
                processReactionList(treeTO);
                ImGui::TableNextColumn();
                processTimestampField(treeTO);
                ImGui::TableNextColumn();
                processUserNameField(treeTO);
                ImGui::TableNextColumn();
                processNumDownloadsField(treeTO);
                ImGui::TableNextColumn();
                processWidthField(treeTO);
                ImGui::TableNextColumn();
                processHeightField(treeTO);
                ImGui::TableNextColumn();
                processNumObjectsField(treeTO, true);
                ImGui::TableNextColumn();
                processSizeField(treeTO, true);
                ImGui::TableNextColumn();
                processVersionField(treeTO);

                popTextColor();

                ImGui::PopID();
            }
        ImGui::EndTable();

        if (scheduleRecreateTreeTOs) {
            createTreeTOs(workspace);
        }
    }
    ImGui::PopID();
}

void BrowserWindow::processGenomeList()
{
    ImGui::PushID("GenomeList");
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable
        | ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody
        | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTable("Browser", 9, flags, ImVec2(0, -scale(WorkspaceBottomSpace)), 0.0f)) {
        ImGui::TableSetupColumn("Genome", ImGuiTableColumnFlags_WidthFixed, scale(210.0f), NetworkResourceColumnId_SimulationName);
        ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthFixed, scale(200.0f), NetworkResourceColumnId_Description);
        ImGui::TableSetupColumn("Reactions", ImGuiTableColumnFlags_WidthFixed, scale(140.0f), NetworkResourceColumnId_Likes);
        ImGui::TableSetupColumn(
            "Timestamp",
            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending,
            scale(135.0f),
            NetworkResourceColumnId_Timestamp);
        ImGui::TableSetupColumn("User name", ImGuiTableColumnFlags_WidthFixed, scale(120.0f), NetworkResourceColumnId_UserName);
        ImGui::TableSetupColumn("Downloads", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_NumDownloads);
        ImGui::TableSetupColumn("Cells", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_Particles);
        ImGui::TableSetupColumn("File size", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_FileSize);
        ImGui::TableSetupColumn("Version", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_Version);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

        //create treeTOs if sorting changed
        if (auto sortSpecs = ImGui::TableGetSortSpecs()) {
            if (sortSpecs->SpecsDirty) {
                for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
                    auto& workspace = _workspaces.at(WorkspaceId{NetworkResourceType_Genome, workspaceType});
                    workspace.sortSpecs.clear();
                    for (int i = 0; i < sortSpecs->SpecsCount; ++i) {
                        workspace.sortSpecs.emplace_back(sortSpecs->Specs[i]);
                    }
                    createTreeTOs(workspace);
                }
                sortSpecs->SpecsDirty = false;
            }
        }

        //process treeTOs
        auto& workspace = _workspaces.at(_currentWorkspace);
        auto scheduleRecreateTreeTOs = false;
        ImGuiListClipper clipper;
        clipper.Begin(workspace.treeTOs.size());
        while (clipper.Step())
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {

                auto& treeTO = workspace.treeTOs.at(row);
                if (treeTO->isLeaf()) {
                    _lastSessionData.registrate(treeTO->getLeaf().rawTO);
                }

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(RowHeight));
                ImGui::TableNextColumn();

                auto selected = _selectedTreeTO == treeTO;
                if (ImGui::Selectable(
                        "",
                        &selected,
                        ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
                        ImVec2(0, scale(RowHeight) - ImGui::GetStyle().FramePadding.y))) {
                    _selectedTreeTO = selected ? treeTO : nullptr;
                }
                ImGui::SameLine();

                pushTextColor(treeTO);

                if (processResourceNameField(treeTO, workspace.collapsedFolderNames)) {
                    scheduleRecreateTreeTOs = true;
                }
                ImGui::TableNextColumn();
                processDescriptionField(treeTO);
                ImGui::TableNextColumn();
                processReactionList(treeTO);
                ImGui::TableNextColumn();
                processTimestampField(treeTO);
                ImGui::TableNextColumn();
                processUserNameField(treeTO);
                ImGui::TableNextColumn();
                processNumDownloadsField(treeTO);
                ImGui::TableNextColumn();
                processNumObjectsField(treeTO, false);
                ImGui::TableNextColumn();
                processSizeField(treeTO, false);
                ImGui::TableNextColumn();
                processVersionField(treeTO);

                popTextColor();

                ImGui::PopID();
            }
        ImGui::EndTable();

        if (scheduleRecreateTreeTOs) {
            createTreeTOs(workspace);
        }
    }
    ImGui::PopID();
}

bool BrowserWindow::processResourceNameField(NetworkResourceTreeTO const& treeTO, std::set<std::vector<std::string>>& collapsedFolderNames)
{
    auto result = false;

    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();

        result |= processFolderTreeSymbols(treeTO, collapsedFolderNames);
        processDownloadButton(leaf);
        ImGui::SameLine();
        if (_currentWorkspace.workspaceType == WorkspaceType_Private && leaf.rawTO->workspaceType == WorkspaceType_Public) {
            AlienImGui::Text(ICON_FA_SHARE_ALT);
            AlienImGui::Tooltip("Visible in the public workspace");
        }
        ImGui::SameLine();

        if (!isOwner(treeTO) && _lastSessionData.isNew(leaf.rawTO)) {
            auto font = StyleRepository::get().getSmallBoldFont();
            auto origSize = font->Scale;
            font->Scale *= 0.65f;
            ImGui::PushFont(font);
            ImGui::PushStyleColor(ImGuiCol_Text, Const::BrowserResourceNewTextColor.Value);
            AlienImGui::Text("NEW");
            ImGui::PopStyleColor();
            font->Scale = origSize;
            ImGui::PopFont();

            ImGui::SameLine();
        }

        processShortenedText(leaf.leafName, true);
    } else {
        auto& folder = treeTO->getFolder();

        result |= processFolderTreeSymbols(treeTO, collapsedFolderNames);
        processShortenedText(treeTO->folderNames.back());
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextLightDecentColor);
        std::string resourceTypeString = [&] {
            if (treeTO->type == NetworkResourceType_Simulation) {
                return folder.numLeafs == 1 ? "sim" : "sims";
            } else {
                return folder.numLeafs == 1 ? "genome" : "genomes";
            }
        }();
        AlienImGui::Text("(" + std::to_string(folder.numLeafs) + " " + resourceTypeString + ")");
        ImGui::PopStyleColor();
    }
    return result;
}

void BrowserWindow::processDescriptionField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        processShortenedText(leaf.rawTO->description);
    }
}

void BrowserWindow::processReactionList(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();

        auto isAddReaction = AlienImGui::ActionButton(AlienImGui::ActionButtonParameters().buttonText(ICON_FA_PLUS));
        AlienImGui::Tooltip("Add a reaction", false);
        if (isAddReaction) {
            _activateEmojiPopup = true;
            _emojiPopupTO = treeTO;
        }

        //calc remap which allows to show most frequent like type first
        std::map<int, int> remap;
        std::set<int> processedEmojiTypes;

        int index = 0;
        while (processedEmojiTypes.size() < leaf.rawTO->numLikesByEmojiType.size()) {
            int maxLikes = 0;
            std::optional<int> maxEmojiType;
            for (auto const& [emojiType, numLikes] : leaf.rawTO->numLikesByEmojiType) {
                if (!processedEmojiTypes.contains(emojiType) && numLikes > maxLikes) {
                    maxLikes = numLikes;
                    maxEmojiType = emojiType;
                }
            }
            processedEmojiTypes.insert(*maxEmojiType);
            remap.emplace(index, *maxEmojiType);
            ++index;
        }

        //show like types with count
        int counter = 0;
        std::optional<int> toggleEmojiType;
        for (auto const& emojiType : remap | std::views::values) {
            auto numLikes = leaf.rawTO->numLikesByEmojiType.at(emojiType);

            ImGui::SameLine();
            AlienImGui::Text(std::to_string(numLikes));
            if (emojiType < _emojis.size()) {
                ImGui::SameLine();
                auto const& emoji = _emojis.at(emojiType);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() - scale(7.0f));
                ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(Const::ToolbarButtonBackgroundColor));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, static_cast<ImVec4>(Const::ToolbarButtonHoveredColor));
                auto cursorPos = ImGui::GetCursorScreenPos();
                auto emojiWidth = scale(toFloat(emoji.width) / 2.5f);
                auto emojiHeight = scale(toFloat(emoji.height) / 2.5f);
                if (ImGui::ImageButton((void*)(intptr_t)emoji.textureId, {emojiWidth, emojiHeight}, ImVec2(0, 0), ImVec2(1, 1), 0)) {
                    toggleEmojiType = emojiType;
                }
                bool isLiked = _ownEmojiTypeBySimId.contains(leaf.rawTO->id) && _ownEmojiTypeBySimId.at(leaf.rawTO->id) == emojiType;
                if (isLiked) {
                    ImGui::GetWindowDrawList()->AddRect(
                        ImVec2(cursorPos.x, cursorPos.y),
                        ImVec2(cursorPos.x + emojiWidth, cursorPos.y + emojiHeight),
                        (ImU32)ImColor::HSV(0, 0, 1, 0.5f),
                        1.0f);
                }
                ImGui::PopStyleColor(2);
                AlienImGui::Tooltip([=, this] { return getUserNamesToEmojiType(leaf.rawTO->id, emojiType); }, false);
            }

            //separator except for last element
            if (++counter < leaf.rawTO->numLikesByEmojiType.size()) {
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() - scale(4.0f));
            }
        }
        if (toggleEmojiType) {
            onToggleLike(treeTO, *toggleEmojiType);
        }
    } else {
        auto& folder = treeTO->getFolder();

        auto pos = ImGui::GetCursorScreenPos();
        ImGui::SetCursorScreenPos({pos.x + scale(3.0f), pos.y});
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextLightDecentColor);
        AlienImGui::Text("(" + std::to_string(folder.numReactions) + ")");
        ImGui::PopStyleColor();
    }
}

void BrowserWindow::processTimestampField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienImGui::Text(leaf.rawTO->timestamp);
    }
}

void BrowserWindow::processUserNameField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        processShortenedText(leaf.rawTO->userName);
    }
}

void BrowserWindow::processNumDownloadsField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienImGui::Text(std::to_string(leaf.rawTO->numDownloads));
    }
}

void BrowserWindow::processWidthField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienImGui::Text(std::to_string(leaf.rawTO->width));
    }
}

void BrowserWindow::processHeightField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienImGui::Text(std::to_string(leaf.rawTO->height));
    }
}

void BrowserWindow::processNumObjectsField(NetworkResourceTreeTO const& treeTO, bool kobjects)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        if (kobjects) {
            AlienImGui::Text(StringHelper::format(leaf.rawTO->particles / 1000) + " K");
        } else {
            AlienImGui::Text(StringHelper::format(leaf.rawTO->particles));
        }
    }
}

void BrowserWindow::processSizeField(NetworkResourceTreeTO const& treeTO, bool kbyte)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        if (kbyte) {
            AlienImGui::Text(StringHelper::format(leaf.rawTO->contentSize / 1024) + " KB");
        } else {
            AlienImGui::Text(StringHelper::format(leaf.rawTO->contentSize) + " Bytes");
        }
    }
}

void BrowserWindow::processVersionField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienImGui::Text(leaf.rawTO->version);
    }
}

bool BrowserWindow::processFolderTreeSymbols(NetworkResourceTreeTO const& treeTO, std::set<std::vector<std::string>>& collapsedFolderNames)
{
    auto result = false;
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserResourceSymbolColor);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0, 0, 0, 0));
    auto const& treeSymbols = treeTO->treeSymbols;
    for (auto const& folderLine : treeSymbols) {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImGuiStyle& style = ImGui::GetStyle();
        switch (folderLine) {
        case FolderTreeSymbols::Expanded: {
            if (AlienImGui::Button(ICON_FA_MINUS_SQUARE, 20.0f)) {
                collapsedFolderNames.insert(treeTO->folderNames);
                result = true;
            }
        } break;
        case FolderTreeSymbols::Collapsed: {
            if (AlienImGui::Button(ICON_FA_PLUS_SQUARE, 20.0f)) {
                collapsedFolderNames.erase(treeTO->folderNames);
                result = true;
            }
        } break;
        case FolderTreeSymbols::Continue: {
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(6.0f), pos.y),
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(RowHeight) + style.FramePadding.y),
                Const::BrowserResourceLineColor);
            ImGui::Dummy({scale(20.0f), 0});
        } break;
        case FolderTreeSymbols::Branch: {
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(6.0f), pos.y),
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(RowHeight) + style.FramePadding.y),
                Const::BrowserResourceLineColor);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y),
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y + scale(1.5f)),
                Const::BrowserResourceLineColor);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f - 0.5f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y - scale(0.5f)),
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f + 2.0f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y + scale(2.0f)),
                Const::BrowserResourceLineColor);
            ImGui::Dummy({scale(20.0f), 0});
        } break;
        case FolderTreeSymbols::End: {
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(6.0f), pos.y),
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y + scale(1.5f)),
                Const::BrowserResourceLineColor);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y),
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y + scale(1.5f)),
                Const::BrowserResourceLineColor);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f - 0.5f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y - scale(0.5f)),
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f + 2.0f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y + scale(2.0f)),
                Const::BrowserResourceLineColor);
            ImGui::Dummy({scale(20.0f), 0});
        } break;
        case FolderTreeSymbols::None: {
            ImGui::Dummy({scale(20.0f), 0});
        } break;
        default: {
        } break;
        }
        ImGui::SameLine();
    }
    ImGui::PopStyleColor(2);
    return result;
}

void BrowserWindow::processEmojiWindow()
{
    if (_activateEmojiPopup) {
        ImGui::OpenPopup("emoji");
        _activateEmojiPopup = false;
    }
    if (ImGui::BeginPopup("emoji")) {
        ImGui::Text("Choose a reaction");
        ImGui::Spacing();
        ImGui::Spacing();
        if (_showAllEmojis) {
            if (ImGui::BeginChild("##reactionchild", ImVec2(scale(335), scale(300)), false)) {
                int offset = 0;
                for (int i = 0; i < NumEmojiBlocks; ++i) {
                    for (int j = 0; j < NumEmojisPerBlock[i]; ++j) {
                        if (j % NumEmojisPerRow != 0) {
                            ImGui::SameLine();
                        }
                        processEmojiButton(offset + j);
                    }
                    AlienImGui::Separator();
                    offset += NumEmojisPerBlock[i];
                }
            }
            ImGui::EndChild();
        } else {
            if (ImGui::BeginChild("##reactionchild", ImVec2(scale(335), scale(90)), false)) {
                for (int i = 0; i < NumEmojisPerRow; ++i) {
                    if (i % NumEmojisPerRow != 0) {
                        ImGui::SameLine();
                    }
                    processEmojiButton(i);
                }
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + scale(8.0f));

                if (AlienImGui::Button("More", ImGui::GetContentRegionAvail().x)) {
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
    auto const& emoji = _emojis.at(emojiType);
    ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(Const::ToolbarButtonBackgroundColor));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, static_cast<ImVec4>(Const::ToolbarButtonHoveredColor));
    auto cursorPos = ImGui::GetCursorScreenPos();
    auto emojiWidth = scale(toFloat(emoji.width));
    auto emojiHeight = scale(toFloat(emoji.height));
    auto leaf = _emojiPopupTO->getLeaf();
    if (ImGui::ImageButton((void*)(intptr_t)emoji.textureId, {emojiWidth, emojiHeight}, {0, 0}, {1.0f, 1.0f})) {
        onToggleLike(_emojiPopupTO, toInt(emojiType));
        ImGui::CloseCurrentPopup();
    }
    ImGui::PopStyleColor(2);

    bool isLiked = _ownEmojiTypeBySimId.contains(leaf.rawTO->id) && _ownEmojiTypeBySimId.at(leaf.rawTO->id) == emojiType;
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

void BrowserWindow::processDownloadButton(BrowserLeaf const& leaf)
{
    auto isDownload = AlienImGui::ActionButton(AlienImGui::ActionButtonParameters().buttonText(ICON_FA_DOWNLOAD));
    AlienImGui::Tooltip("Download", false);
    if (isDownload) {
        onDownloadResource(leaf);
    }
}

namespace
{
    std::vector<std::string> splitString(const std::string& str)
    {
        std::vector<std::string> tokens; 
        boost::algorithm::split_regex(tokens, str, boost::regex("(\n)+"));
        return tokens;
    }
}

void BrowserWindow::processShortenedText(std::string const& text, bool bold) {
    auto substrings = splitString(text);
    if (substrings.empty()) {
        return;
    }
    auto& styleRepository = StyleRepository::get();
    auto textSize = ImGui::CalcTextSize(substrings.at(0).c_str());
    auto needDetailButton = textSize.x > ImGui::GetContentRegionAvail().x || substrings.size() > 1;
    auto cursorPos = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - styleRepository.scale(15.0f);
    if (bold) {
        ImGui::PushFont(styleRepository.getSmallBoldFont());
    }
    AlienImGui::Text(substrings.at(0));
    if (bold) {
        ImGui::PopFont();
    }
    if (needDetailButton) {
        ImGui::SameLine();
        ImGui::SetCursorPosX(cursorPos);

        processDetailButton();
        AlienImGui::Tooltip(text.c_str(), false);
    }
}

bool BrowserWindow::processActionButton(std::string const& text)
{
    ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(Const::ToolbarButtonBackgroundColor));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImU32)Const::ToolbarButtonHoveredColor);
    auto result = ImGui::Button(text.c_str());
    ImGui::PopStyleColor(2);
   
    return result;
}

bool BrowserWindow::processDetailButton()
{
    auto color = Const::DetailButtonColor;
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(color.Value.x, color.Value.y, color.Value.z, h, s, v);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(h, s, v * 0.3f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(h, s, v * 0.4f));
    auto detailClicked = AlienImGui::Button("...");
    ImGui::PopStyleColor(2);
    return detailClicked;
}

void BrowserWindow::processRefreshingScreen(RealVector2D const& startPos)
{
    if (_refreshProcessor->pendingTasks()) {
        auto color = ImColor(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
        color.Value.w = 0.5f;
        auto size = ImGui::GetItemRectSize();
        auto afterTablePos = ImGui::GetCursorScreenPos();

        ImGui::SetCursorScreenPos({startPos.x, startPos.y});
        if (ImGui::BeginChild("##overlay", {size.x, size.y}, 0, ImGuiWindowFlags_NoScrollbar)) {
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            drawList->AddRectFilled({startPos.x, startPos.y}, {startPos.x + size.x, startPos.y + size.y}, color);
            AlienImGui::Spinner(AlienImGui::SpinnerParameters().pos({startPos.x + size.x / 2, startPos.y + size.y / 2}));
        }
        ImGui::EndChild();
        ImGui::SetCursorScreenPos(afterTablePos);
    }
}

void BrowserWindow::processActivated()
{
    onRefresh();
}

void BrowserWindow::processPendingRequestIds()
{
    _refreshProcessor->process();
    _emojiUserNameProcessor->process();
    _reactionProcessor->process();
}

void BrowserWindow::createTreeTOs(Workspace& workspace)
{
    //sorting
    if (workspace.rawTOs.size() > 1) {
        std::sort(workspace.rawTOs.begin(), workspace.rawTOs.end(), [&](auto const& left, auto const& right) {
            return _NetworkResourceRawTO::compare(left, right, workspace.sortSpecs) < 0;
        });
    }

    //filtering
    std::vector<NetworkResourceRawTO> filteredRawTOs;
    for (auto const& rawTO : workspace.rawTOs) {
        if (rawTO->matchWithFilter(_filter)) {
            filteredRawTOs.emplace_back(rawTO);
        }
    }

    //create treeTOs
    workspace.treeTOs = NetworkResourceService::get().createTreeTOs(filteredRawTOs, workspace.collapsedFolderNames);
    _selectedTreeTO = nullptr;
}

void BrowserWindow::sortUserList()
{
    std::sort(_userTOs.begin(), _userTOs.end(), [&](auto const& left, auto const& right) { return UserTO::compareOnlineAndTimestamp(left, right) > 0; });
}

void BrowserWindow::onDownloadResource(BrowserLeaf const& leaf)
{
    ++leaf.rawTO->numDownloads;

    NetworkTransferController::get().onDownload(DownloadNetworkResourceRequestData{
        .resourceId = leaf.rawTO->id,
        .resourceName = leaf.rawTO->resourceName,
        .resourceVersion = leaf.rawTO->version,
        .resourceType = _currentWorkspace.resourceType,
        .downloadCache = _downloadCache});
}

void BrowserWindow::onReplaceResource(BrowserLeaf const& leaf)
{
    auto func = [&] {
        auto data = [&]() -> std::variant<ReplaceNetworkResourceRequestData::SimulationData, ReplaceNetworkResourceRequestData::GenomeData> {
            if (_currentWorkspace.resourceType == NetworkResourceType_Simulation) {
                return ReplaceNetworkResourceRequestData::SimulationData{.zoom = Viewport::get().getZoomFactor(), .center = Viewport::get().getCenterInWorldPos()};
            } else {
                return ReplaceNetworkResourceRequestData::GenomeData{.description = GenomeEditorWindow::get().getCurrentGenome()};
            }
        }();
        NetworkTransferController::get().onReplace(ReplaceNetworkResourceRequestData{
            .resourceId = leaf.rawTO->id,
            .workspaceType = leaf.rawTO->workspaceType,
            .downloadCache = getSimulationCache(),
            .data = data});
    };
    GenericMessageDialog::get().yesNo("Delete", "Do you really want to replace the content of the selected item?", func);
}

void BrowserWindow::onEditResource(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        EditSimulationDialog::get().openForLeaf(treeTO);
    } else {
        auto rawTOs = NetworkResourceService::get().getMatchingRawTOs(treeTO, _workspaces.at(_currentWorkspace).rawTOs);
        EditSimulationDialog::get().openForFolder(treeTO, rawTOs);
    }
}

void BrowserWindow::onMoveResource(NetworkResourceTreeTO const& treeTO)
{
    auto& source = _workspaces.at(_currentWorkspace);
    auto rawTOs = NetworkResourceService::get().getMatchingRawTOs(treeTO, source.rawTOs);

    for (auto const& rawTO : rawTOs) {
        switch (rawTO->workspaceType) {
        case WorkspaceType_Private: {
            rawTO->workspaceType = WorkspaceType_Public;
            auto& publicWorkspace = _workspaces.at(WorkspaceId{_currentWorkspace.resourceType, WorkspaceType_Public});
            publicWorkspace.rawTOs.emplace_back(rawTO);
        } break;
        case WorkspaceType_Public: {
            rawTO->workspaceType = WorkspaceType_Private;
            auto& publicWorkspace = _workspaces.at(WorkspaceId{_currentWorkspace.resourceType, WorkspaceType_Public});
            auto findResult = std::ranges::find_if(publicWorkspace.rawTOs, [&](NetworkResourceRawTO const& otherRawTO) { return otherRawTO->id == rawTO->id; });
            if (findResult != publicWorkspace.rawTOs.end()) {
                publicWorkspace.rawTOs.erase(findResult);
            }
        } break;
        default:
            break;
        }
    }
    for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
        createTreeTOs(_workspaces.at(WorkspaceId{_currentWorkspace.resourceType, workspaceType}));
    }

    //apply changes to server
    MoveNetworkResourceRequestData requestData;
    for (auto const& rawTO : rawTOs) {
        requestData.entries.emplace_back(rawTO->id, rawTO->workspaceType);
    }
    NetworkTransferController::get().onMove(requestData);
}

void BrowserWindow::onDeleteResource(NetworkResourceTreeTO const& treeTO)
{
    auto& currentWorkspace = _workspaces.at(_currentWorkspace);
    auto rawTOs = NetworkResourceService::get().getMatchingRawTOs(treeTO, currentWorkspace.rawTOs);

    auto message = treeTO->isLeaf() ? "Do you really want to delete the selected item?" : "Do you really want to delete the selected folder?";
    GenericMessageDialog::get().yesNo("Delete", message, [rawTOs = rawTOs, this]() {

        //remove resources form workspace
        for (WorkspaceType workspaceType = 0; workspaceType < WorkspaceType_Count; ++workspaceType) {
            auto& workspace = _workspaces.at(WorkspaceId{_currentWorkspace.resourceType, workspaceType});
            for (auto const& rawTO : rawTOs) {
                auto findResult = std::ranges::find_if(workspace.rawTOs, [&](NetworkResourceRawTO const& otherRawTO) { return otherRawTO->id == rawTO->id; });
                if (findResult != workspace.rawTOs.end()) {
                    workspace.rawTOs.erase(findResult);
                }
            }
            createTreeTOs(workspace);
        }

        //apply changes to server
        DeleteNetworkResourceRequestData requestData;
        for (auto const& rawTO : rawTOs) {
            requestData.entries.emplace_back(rawTO->id);
        }
        NetworkTransferController::get().onDelete(requestData);
    });
}

void BrowserWindow::onToggleLike(NetworkResourceTreeTO const& to, int emojiType)
{
    CHECK(to->isLeaf());
    auto& leaf = to->getLeaf();
    if (NetworkService::get().getLoggedInUserName()) {

        //remove existing like
        auto findResult = _ownEmojiTypeBySimId.find(leaf.rawTO->id);
        auto onlyRemoveLike = false;
        if (findResult != _ownEmojiTypeBySimId.end()) {
            auto origEmojiType = findResult->second;
            if (--leaf.rawTO->numLikesByEmojiType[origEmojiType] == 0) {
                leaf.rawTO->numLikesByEmojiType.erase(origEmojiType);
            }
            _ownEmojiTypeBySimId.erase(findResult);
            _userNamesByEmojiTypeBySimIdCache.erase(std::make_pair(leaf.rawTO->id, origEmojiType));  //invalidate cache entry
            onlyRemoveLike = origEmojiType == emojiType;  //remove like if same like icon has been clicked
        }

        //create new like
        if (!onlyRemoveLike) {
            _ownEmojiTypeBySimId[leaf.rawTO->id] = emojiType;
            if (leaf.rawTO->numLikesByEmojiType.contains(emojiType)) {
                ++leaf.rawTO->numLikesByEmojiType[emojiType];
            } else {
                leaf.rawTO->numLikesByEmojiType[emojiType] = 1;
            }
        }

        _userNamesByEmojiTypeBySimIdCache.erase(std::make_pair(leaf.rawTO->id, emojiType));  //invalidate cache entry

        _reactionProcessor->executeTask(
            [&](auto const& senderId) {
                return _persisterFacade->scheduleToggleReactionNetworkResource(
                    SenderInfo{.senderId = senderId, .wishResultData = false, .wishErrorInfo = false},
                    ToggleReactionNetworkResourceRequestData{.resourceId = leaf.rawTO->id, .emojiType = emojiType});
            },
            [](auto const&) {},
            [&](auto const& errors) {
                onRefresh();
                GenericMessageDialog::get().information("Error", errors);
            });
    } else {
        LoginDialog::get().open();
    }
}

void BrowserWindow::onExpandFolders()
{
    auto& workspace = _workspaces.at(_currentWorkspace);
    workspace.collapsedFolderNames.clear();
    createTreeTOs(workspace);
}

void BrowserWindow::onCollapseFolders()
{
    auto& workspace = _workspaces.at(_currentWorkspace);
    workspace.collapsedFolderNames = NetworkResourceService::get().getFolderNames(workspace.rawTOs, 1);
    createTreeTOs(workspace);
}

void BrowserWindow::openWeblink(std::string const& link)
{
#ifdef _WIN32
    ShellExecute(NULL, "open", link.c_str(), NULL, NULL, SW_SHOWNORMAL);
#endif
}

bool BrowserWindow::isOwner(NetworkResourceTreeTO const& treeTO) const
{
    if (treeTO == nullptr) {
        return false;
    }
    auto const& workspace = _workspaces.at(_currentWorkspace);

    auto rawTOs = NetworkResourceService::get().getMatchingRawTOs(treeTO, workspace.rawTOs);
    auto userName = NetworkService::get().getLoggedInUserName().value_or("");
    return std::ranges::all_of(rawTOs, [&](NetworkResourceRawTO const& rawTO) { return rawTO->userName == userName; });
}

std::string BrowserWindow::getUserNamesToEmojiType(std::string const& resourceId, int emojiType)
{
    std::set<std::string> userNames;

    auto findResult = _userNamesByEmojiTypeBySimIdCache.find(std::make_pair(resourceId, emojiType));
    if (findResult != _userNamesByEmojiTypeBySimIdCache.end()) {
        userNames = findResult->second;
    } else {
        if (!_emojiUserNameProcessor->pendingTasks()) {
            _emojiUserNameProcessor->executeTask(
                [&](auto const& senderId) {
                    return _persisterFacade->scheduleGetUserNamesForReaction(
                        SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = false},
                        GetUserNamesForReactionRequestData{.resourceId = resourceId, .emojiType = emojiType});
                },
                [&](auto const& requestId) {
                    auto data = _persisterFacade->fetchGetUserNamesForReactionData(requestId);
                    _userNamesByEmojiTypeBySimIdCache.emplace(std::make_pair(data.resourceId, data.emojiType), data.userNames);
                },
                [](auto const& errors) { GenericMessageDialog::get().information("Error", errors); });
        }
        return "Loading...";
    }

    return boost::algorithm::join(userNames, ", ");
}

std::unordered_set<NetworkResourceRawTO> BrowserWindow::getAllRawTOs() const
{
    std::unordered_set<NetworkResourceRawTO> result;
    for (auto const& workspace : _workspaces | std::views::values) {
        result.insert(workspace.rawTOs.begin(), workspace.rawTOs.end());
    }
    return result;
}

void BrowserWindow::pushTextColor(NetworkResourceTreeTO const& to)
{
    if (to->isLeaf()) {
        auto const& leaf = to->getLeaf();
        if (VersionParserService::get().isVersionOutdated(leaf.rawTO->version)) {
            ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::BrowserVersionOutdatedTextColor);
        } else if (VersionParserService::get().isVersionNewer(leaf.rawTO->version)) {
            ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::BrowserVersionNewerTextColor);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::BrowserVersionOkTextColor);
        }
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::BrowserResourceTextColor);
    }
}

void BrowserWindow::popTextColor()
{
    ImGui::PopStyleColor();
}
