#include "BrowserWindow.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <ranges>

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/GlobalSettings.h"
#include "Base/Resources.h"
#include "Base/StringHelper.h"
#include "Base/VersionChecker.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationController.h"
#include "Network/NetworkResourceService.h"
#include "Network/NetworkService.h"
#include "Network/NetworkResourceParserService.h"
#include "Network/NetworkResourceTreeTO.h"

#include "AlienImGui.h"
#include "StyleRepository.h"
#include "StatisticsWindow.h"
#include "Viewport.h"
#include "TemporalControlWindow.h"
#include "MessageDialog.h"
#include "LoginDialog.h"
#include "UploadSimulationDialog.h"
#include "DelayedExecutionController.h"
#include "EditorController.h"
#include "OpenGLHelper.h"
#include "OverlayMessageController.h"
#include "GenomeEditorWindow.h"

namespace
{
    auto constexpr RefreshInterval = 20;  //in minutes

    auto constexpr UserTableWidth = 300.0f;
    auto constexpr BrowserBottomHeight = 68.0f;
    auto constexpr RowHeight = 25.0f;

    auto constexpr NumEmojiBlocks = 4;
    int const NumEmojisPerBlock[] = {19, 14, 10, 6};
    auto constexpr NumEmojisPerRow = 5;
}

_BrowserWindow::_BrowserWindow(
    SimulationController const& simController,
    StatisticsWindow const& statisticsWindow,
    Viewport const& viewport,
    TemporalControlWindow const& temporalControlWindow,
    EditorController const& editorController)
    : _AlienWindow("Browser", "windows.browser", true)
    , _simController(simController)
    , _statisticsWindow(statisticsWindow)
    , _viewport(viewport)
    , _temporalControlWindow(temporalControlWindow)
    , _editorController(editorController)
{
    _showCommunityCreations = GlobalSettings::getInstance().getBoolState("windows.browser.show community creations", _showCommunityCreations);
    _userTableWidth = GlobalSettings::getInstance().getFloatState("windows.browser.user table width", scale(UserTableWidth));

    int numEmojis = 0;
    for (int i = 0; i < NumEmojiBlocks; ++i) {
        numEmojis += NumEmojisPerBlock[i];
    }
    for (int i = 1; i <= numEmojis; ++i) {
        _emojis.emplace_back(OpenGLHelper::loadTexture(Const::BasePath + "emoji" + std::to_string(i) + ".png"));
    }
}

_BrowserWindow::~_BrowserWindow()
{
    GlobalSettings::getInstance().setBoolState("windows.browser.show community creations", _showCommunityCreations);
    GlobalSettings::getInstance().setBoolState("windows.browser.first start", false);
    GlobalSettings::getInstance().setFloatState("windows.browser.user table width", _userTableWidth);
    GlobalSettings::getInstance().setStringState(
        "windows.browser.simulations.collapsed folders", NetworkResourceService::convertFolderNamesToSettings(_simulations.collapsedFolderNames));
    GlobalSettings::getInstance().setStringState(
        "windows.browser.genomes.collapsed folders", NetworkResourceService::convertFolderNamesToSettings(_genomes.collapsedFolderNames));
    NetworkService::getInstance().shutdown();
}

void _BrowserWindow::registerCyclicReferences(LoginDialogWeakPtr const& loginDialog, UploadSimulationDialogWeakPtr const& uploadSimulationDialog)
{
    _loginDialog = loginDialog;
    _uploadSimulationDialog = uploadSimulationDialog;

    auto firstStart = GlobalSettings::getInstance().getBoolState("windows.browser.first start", true);
    refreshIntern(firstStart);

    auto initialCollapsedSimulationFolders = NetworkResourceService::convertFolderNamesToSettings(NetworkResourceService::getAllFolderNames(_simulations.rawTOs));
    auto collapsedSimulationFolders = GlobalSettings::getInstance().getStringState("windows.browser.simulations.collapsed folders", initialCollapsedSimulationFolders);
    _simulations.collapsedFolderNames = NetworkResourceService::convertSettingsToFolderNames(collapsedSimulationFolders);

    auto initialCollapsedGenomeFolders =
        NetworkResourceService::convertFolderNamesToSettings(NetworkResourceService::getAllFolderNames(_simulations.rawTOs));
    auto collapsedGenomeFolders =
        GlobalSettings::getInstance().getStringState("windows.browser.genomes.collapsed folders", initialCollapsedGenomeFolders);
    _genomes.collapsedFolderNames = NetworkResourceService::convertSettingsToFolderNames(collapsedGenomeFolders);
}

void _BrowserWindow::onRefresh()
{
    refreshIntern(true);
}

void _BrowserWindow::refreshIntern(bool withRetry)
{
    try {
        auto& networkService = NetworkService::getInstance();
        networkService.refreshLogin();

        bool success = networkService.getRemoteSimulationList(_allRawTOs, withRetry);
        success &= networkService.getUserList(_userTOs, withRetry);

        if (!success) {
            if (withRetry) {
                MessageDialog::getInstance().information("Error", "Failed to retrieve browser data. Please try again.");
            }
        } else {
            _simulations.numResources = 0;
            _genomes.numResources = 0;
            for (auto const& entry : _allRawTOs) {
                if (entry->type == NetworkResourceType_Simulation) {
                    ++_simulations.numResources;
                } else {
                    ++_genomes.numResources;
                }
            }
        }
        filterRawTOs();
        scheduleCreateTreeTOs();

        if (networkService.getLoggedInUserName()) {
            if (!networkService.getEmojiTypeBySimId(_ownEmojiTypeBySimId)) {
                MessageDialog::getInstance().information("Error", "Failed to retrieve browser data. Please try again.");
            }
        } else {
            _ownEmojiTypeBySimId.clear();
        }
        sortUserList();
    } catch (std::exception const& e) {
        if (withRetry) {
            MessageDialog::getInstance().information("Error", e.what());
        }
    }
}

void _BrowserWindow::processIntern()
{
    processToolbar();

    {
        auto sizeAvailable = ImGui::GetContentRegionAvail();
        if (ImGui::BeginChild(
                "##1",
                ImVec2(sizeAvailable.x - scale(_userTableWidth), sizeAvailable.y - scale(BrowserBottomHeight)),
                false,
                ImGuiWindowFlags_HorizontalScrollbar)) {
            if (ImGui::BeginTabBar("##Type", ImGuiTabBarFlags_FittingPolicyResizeDown)) {
                if (ImGui::BeginTabItem("Simulations", nullptr, ImGuiTabItemFlags_None)) {
                    processSimulationList();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Genomes", nullptr, ImGuiTabItemFlags_None)) {
                    processGenomeList();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        }
        ImGui::EndChild();
    }
    ImGui::SameLine();

    {
        auto sizeAvailable = ImGui::GetContentRegionAvail();
        ImGui::Button("", ImVec2(scale(5.0f), sizeAvailable.y - scale(BrowserBottomHeight)));
        if (ImGui::IsItemActive()) {
            _userTableWidth -= ImGui::GetIO().MouseDelta.x;
        }
    }

    ImGui::SameLine();
    {
        auto sizeAvailable = ImGui::GetContentRegionAvail();
        if (ImGui::BeginChild(
                "##2", ImVec2(sizeAvailable.x, sizeAvailable.y - scale(BrowserBottomHeight)), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            processUserList();
        }
        ImGui::EndChild();
    }

    processStatus();
    processFilter();
    processEmojiWindow();
}

void _BrowserWindow::processBackground()
{
    auto now = std::chrono::steady_clock::now();
    if (!_lastRefreshTime) {
        _lastRefreshTime = now;
    }
    if (std::chrono::duration_cast<std::chrono::minutes>(now - *_lastRefreshTime).count() >= RefreshInterval) {
        _lastRefreshTime = now;
        refreshIntern(false);
    }

    if (_scheduleRefresh) {
        onRefresh();
        _scheduleRefresh = false;
    }
}

void _BrowserWindow::processToolbar()
{
    auto& networkService = NetworkService::getInstance();
    std::string resourceTypeString = _visibleResourceType == NetworkResourceType_Simulation ? "simulation" : "genome";

    //refresh button
    if (AlienImGui::ToolbarButton(ICON_FA_SYNC)) {
        onRefresh();
    }
    AlienImGui::Tooltip("Refresh");

    //login button
    ImGui::SameLine();
    ImGui::BeginDisabled(networkService.getLoggedInUserName().has_value());
    if (AlienImGui::ToolbarButton(ICON_FA_SIGN_IN_ALT)) {
        if (auto loginDialog = _loginDialog.lock()) {
            loginDialog->open();
        }
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Login or register");

    //logout button
    ImGui::SameLine();
    ImGui::BeginDisabled(!networkService.getLoggedInUserName());
    if (AlienImGui::ToolbarButton(ICON_FA_SIGN_OUT_ALT)) {
        if (auto loginDialog = _loginDialog.lock()) {
            networkService.logout();
            onRefresh();
        }
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Logout");

    //separator
    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    //upload button
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_UPLOAD)) {
        std::string prefix = [&] {
            if (_selectedResource == nullptr || _selectedResource->isLeaf()) {
                return std::string();
            }
            return NetworkResourceService::concatenateFolderNames(_selectedResource->folderNames, true);
        }();
        _uploadSimulationDialog.lock()->open(_visibleResourceType, prefix);
    }
    AlienImGui::Tooltip(
        "Share your current " + resourceTypeString + " with other users:\nThe " + resourceTypeString
        + " will be uploaded to the server and made visible in the browser.\nIf you have already selected a folder, your " + resourceTypeString
        + " will be uploaded there.");

    //delete button
    ImGui::SameLine();
    ImGui::BeginDisabled(
        _selectedResource == nullptr || !_selectedResource->isLeaf()
        || _selectedResource->getLeaf().rawTO->userName != networkService.getLoggedInUserName().value_or(""));
    if (AlienImGui::ToolbarButton(ICON_FA_TRASH)) {
        onDeleteItem(_selectedResource->getLeaf());
        _selectedResource = nullptr;
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Delete selected " + resourceTypeString);

    //separator
    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    //expand button
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_EXPAND_ARROWS_ALT)) {
        onExpandFolders();
    }
    AlienImGui::Tooltip("Expand all folders");

    //collapse button
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_COMPRESS_ARROWS_ALT)) {
        onCollapseFolders();
    }
    AlienImGui::Tooltip("Collapse all folders");

#ifdef _WIN32
    //separator
    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    //Discord button
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_COMMENTS)) {
        openWeblink(Const::DiscordLink);
    }
    AlienImGui::Tooltip("Open ALIEN Discord server");
#endif

    AlienImGui::Separator();
}

void _BrowserWindow::processSimulationList()
{
    ImGui::PushID("SimulationList");
    _visibleResourceType = NetworkResourceType_Simulation;
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable
        | ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody
        | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTable("Browser", 11, flags, ImVec2(0, 0), 0.0f)) {
        ImGui::TableSetupColumn("Simulation", ImGuiTableColumnFlags_WidthFixed, scale(210.0f), NetworkResourceColumnId_SimulationName);
        ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthFixed, scale(200.0f), NetworkResourceColumnId_Description);
        ImGui::TableSetupColumn("Reactions", ImGuiTableColumnFlags_WidthFixed, scale(120.0f), NetworkResourceColumnId_Likes);
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

        //create table data if necessary
        if (ImGuiTableSortSpecs* sortSpecs = ImGui::TableGetSortSpecs()) {
            if (sortSpecs->SpecsDirty || _scheduleCreateSimulationTreeTOs) {
                sortRawTOs(_simulations.rawTOs, sortSpecs);
                sortSpecs->SpecsDirty = false;
                _scheduleCreateSimulationTreeTOs = false;

                _simulations.treeTOs = NetworkResourceService::createTreeTOs(_simulations.rawTOs, _simulations.collapsedFolderNames);
            }
        }
        ImGuiListClipper clipper;
        clipper.Begin(_simulations.treeTOs.size());
        while (clipper.Step())
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto treeTO = _simulations.treeTOs[row];

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(RowHeight));
                ImGui::TableNextColumn();

                auto selected = _selectedResource == treeTO;
                if (ImGui::Selectable(
                        "",
                        &selected,
                        ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
                        ImVec2(0, scale(RowHeight) - ImGui::GetStyle().FramePadding.y))) {
                    _selectedResource = selected ? treeTO : nullptr;
                }
                ImGui::SameLine();

                pushTextColor(treeTO);

                processResourceNameField(treeTO, _simulations.collapsedFolderNames);
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
                processNumParticlesField(treeTO);
                ImGui::TableNextColumn();
                processSizeField(treeTO, true);
                ImGui::TableNextColumn();
                processVersionField(treeTO);

                popTextColor();

                ImGui::PopID();
            }
        ImGui::EndTable();
    }
    ImGui::PopID();
}

void _BrowserWindow::processGenomeList()
{
    ImGui::PushID("GenomeList");
    _visibleResourceType = NetworkResourceType_Genome;
    auto& styleRepository = StyleRepository::getInstance();
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable
        | ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody
        | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTable("Browser", 9, flags, ImVec2(0, 0), 0.0f)) {
        ImGui::TableSetupColumn("Genome", ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(210.0f), NetworkResourceColumnId_SimulationName);
        ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(200.0f), NetworkResourceColumnId_Description);
        ImGui::TableSetupColumn("Reactions", ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(120.0f), NetworkResourceColumnId_Likes);
        ImGui::TableSetupColumn(
            "Timestamp",
            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending,
            scale(135.0f),
            NetworkResourceColumnId_Timestamp);
        ImGui::TableSetupColumn("User name", ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(120.0f), NetworkResourceColumnId_UserName);
        ImGui::TableSetupColumn("Downloads", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_NumDownloads);
        ImGui::TableSetupColumn("Cells", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_Particles);
        ImGui::TableSetupColumn("File size", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_FileSize);
        ImGui::TableSetupColumn("Version", ImGuiTableColumnFlags_WidthFixed, 0.0f, NetworkResourceColumnId_Version);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        //create table data if necessary
        if (ImGuiTableSortSpecs* sortSpecs = ImGui::TableGetSortSpecs()) {
            if (sortSpecs->SpecsDirty || _scheduleCreateGenomeTreeTOs) {
                sortRawTOs(_genomes.rawTOs, sortSpecs);
                sortSpecs->SpecsDirty = false;
                _scheduleCreateGenomeTreeTOs = false;

                _genomes.treeTOs = NetworkResourceService::createTreeTOs(_genomes.rawTOs, _genomes.collapsedFolderNames);
            }
        }
        ImGuiListClipper clipper;
        clipper.Begin(_genomes.treeTOs.size());
        while (clipper.Step())
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {

                auto& treeTO = _genomes.treeTOs[row];

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(RowHeight));
                ImGui::TableNextColumn();

                auto selected = _selectedResource == treeTO;
                if (ImGui::Selectable(
                        "",
                        &selected,
                        ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
                        ImVec2(0, scale(RowHeight) - ImGui::GetStyle().FramePadding.y))) {
                    _selectedResource = selected ? treeTO : nullptr;
                }
                ImGui::SameLine();

                pushTextColor(treeTO);

                processResourceNameField(treeTO, _genomes.collapsedFolderNames);
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
                processNumParticlesField(treeTO);
                ImGui::TableNextColumn();
                processSizeField(treeTO, false);
                ImGui::TableNextColumn();
                processVersionField(treeTO);

                popTextColor();

                ImGui::PopID();
            }
        ImGui::EndTable();
    }
    ImGui::PopID();
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

void _BrowserWindow::processUserList()
{
    auto& networkService = NetworkService::getInstance();

    ImGui::PushID("User list");
    auto& styleRepository = StyleRepository::getInstance();
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable
         | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody
        | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    AlienImGui::Group("Simulators");
    if (ImGui::BeginTable("Browser", 5, flags, ImVec2(0, 0), 0.0f)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed, scale(90.0f));
        auto isLoggedIn = networkService.getLoggedInUserName().has_value();
        ImGui::TableSetupColumn(
            isLoggedIn ? "GPU model" : "GPU (visible if logged in)",
            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed,
            styleRepository.scale(200.0f));
        ImGui::TableSetupColumn("Time spent", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(80.0f));
        ImGui::TableSetupColumn(
            "Reactions received", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, scale(120.0f));
        ImGui::TableSetupColumn("Reactions given", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(100.0f));
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        ImGuiListClipper clipper;
        clipper.Begin(_userTOs.size());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto item = &_userTOs[row];

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(RowHeight));

                ImGui::TableNextColumn();
                auto isBoldFont = isLoggedIn && *networkService.getLoggedInUserName() == item->userName;

                if (item->online) {
                    AlienImGui::OnlineSymbol();
                    ImGui::SameLine();
                } else if (item->lastDayOnline) {
                    AlienImGui::LastDayOnlineSymbol();
                    ImGui::SameLine();
                }
                processShortenedText(item->userName, isBoldFont);

                ImGui::TableNextColumn();
                if (isLoggedIn && _loginDialog.lock()->isShareGpuInfo()) {
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
    ImGui::PopID();
}

void _BrowserWindow::processStatus()
{
    auto& styleRepository = StyleRepository::getInstance();
    auto& networkService = NetworkService::getInstance();

    if (ImGui::BeginChild("##", ImVec2(0, styleRepository.scale(33.0f)), true)) {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::MonospaceColor);
        std::string statusText;
        statusText += std::string(" " ICON_FA_INFO_CIRCLE " ");
        statusText += std::to_string(_simulations.numResources) + " simulations found";

        statusText += std::string(" " ICON_FA_INFO_CIRCLE " ");
        statusText += std::to_string(_genomes.numResources) + " genomes found";

        statusText += std::string(" " ICON_FA_INFO_CIRCLE " ");
        statusText += std::to_string(_userTOs.size()) + " simulators found";

        statusText += std::string("  " ICON_FA_INFO_CIRCLE " ");
        if (auto userName = networkService.getLoggedInUserName()) {
            statusText += "Logged in as " + *userName + " @ " + networkService.getServerAddress();// + ": ";
        } else {
            statusText += "Not logged in to " + networkService.getServerAddress();// + ": ";
        }

        if (!networkService.getLoggedInUserName()) {
            statusText += std::string("   " ICON_FA_INFO_CIRCLE " ");
            statusText += "In order to share and upvote simulations you need to log in.";
        }
        AlienImGui::Text(statusText);
        ImGui::PopStyleColor();
    }
    ImGui::EndChild();
}

void _BrowserWindow::processFilter()
{
    ImGui::Spacing();
    if (AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Community creations"), _showCommunityCreations)) {
        filterRawTOs();
        scheduleCreateTreeTOs();
    }
    ImGui::SameLine();
    if (AlienImGui::InputText(AlienImGui::InputTextParameters().name("Filter"), _filter)) {
        filterRawTOs();
        scheduleCreateTreeTOs();
    }
}

void _BrowserWindow::processResourceNameField(NetworkResourceTreeTO const& treeTO, std::set<std::vector<std::string>>& collapsedFolderNames)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();

        processFolderTreeSymbols(treeTO, collapsedFolderNames);
        processDownloadButton(leaf);
        ImGui::SameLine();
        processShortenedText(leaf.leafName, true);
    } else {
        auto& folder = treeTO->getFolder();

        processFolderTreeSymbols(treeTO, collapsedFolderNames);
        processShortenedText(treeTO->folderNames.back());
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserFolderPropertiesTextColor);
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
}

void _BrowserWindow::processDescriptionField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        processShortenedText(leaf.rawTO->description);
    }
}

void _BrowserWindow::processReactionList(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserDownloadButtonTextColor);
        auto isAddReaction = processActionButton(ICON_FA_PLUS);
        ImGui::PopStyleColor();
        AlienImGui::Tooltip("Add a reaction");
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
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserFolderPropertiesTextColor);
        AlienImGui::Text("(" + std::to_string(folder.numReactions) + ")");
        ImGui::PopStyleColor();
    }
}

void _BrowserWindow::processTimestampField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienImGui::Text(leaf.rawTO->timestamp);
    }
}

void _BrowserWindow::processUserNameField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        processShortenedText(leaf.rawTO->userName);
    }
}

void _BrowserWindow::processNumDownloadsField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienImGui::Text(std::to_string(leaf.rawTO->numDownloads));
    }
}

void _BrowserWindow::processWidthField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienImGui::Text(std::to_string(leaf.rawTO->width));
    }
}

void _BrowserWindow::processHeightField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienImGui::Text(std::to_string(leaf.rawTO->height));
    }
}

void _BrowserWindow::processNumParticlesField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienImGui::Text(StringHelper::format(leaf.rawTO->particles / 1000) + " K");
    }
}

void _BrowserWindow::processSizeField(NetworkResourceTreeTO const& treeTO, bool kbyte)
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

void _BrowserWindow::processVersionField(NetworkResourceTreeTO const& treeTO)
{
    if (treeTO->isLeaf()) {
        auto& leaf = treeTO->getLeaf();
        AlienImGui::Text(leaf.rawTO->version);
    }
}

void _BrowserWindow::processFolderTreeSymbols(NetworkResourceTreeTO const& treeTO, std::set<std::vector<std::string>>& collapsedFolderNames)
{
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserFolderSymbolColor);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0, 0, 0, 0));
    auto const& treeSymbols = treeTO->treeSymbols;
    for (auto const& folderLine : treeSymbols) {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImGuiStyle& style = ImGui::GetStyle();
        switch (folderLine) {
        case FolderTreeSymbols::Expanded: {
            if (AlienImGui::Button(ICON_FA_MINUS_SQUARE, 20.0f)) {
                collapsedFolderNames.insert(treeTO->folderNames);
                scheduleCreateTreeTOs();
            }
        } break;
        case FolderTreeSymbols::Collapsed: {
            if (AlienImGui::Button(ICON_FA_PLUS_SQUARE, 20.0f)) {
                collapsedFolderNames.erase(treeTO->folderNames);
                scheduleCreateTreeTOs();
            }
        } break;
        case FolderTreeSymbols::Continue: {
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(6.0f), pos.y),
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(RowHeight) + style.FramePadding.y),
                Const::BrowserFolderLineColor);
            ImGui::Dummy({scale(20.0f), 0});
        } break;
        case FolderTreeSymbols::Branch: {
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(6.0f), pos.y),
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(RowHeight) + style.FramePadding.y),
                Const::BrowserFolderLineColor);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y),
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y + scale(1.5f)),
                Const::BrowserFolderLineColor);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f - 0.5f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y - scale(0.5f)),
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f + 2.0f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y + scale(2.0f)),
                Const::BrowserFolderLineColor);
            ImGui::Dummy({scale(20.0f), 0});
        } break;
        case FolderTreeSymbols::End: {
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(6.0f), pos.y),
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y + scale(1.5f)),
                Const::BrowserFolderLineColor);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(7.5f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y),
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y + scale(1.5f)),
                Const::BrowserFolderLineColor);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f - 0.5f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y - scale(0.5f)),
                ImVec2(pos.x + style.FramePadding.x + scale(20.0f + 2.0f), pos.y + scale(RowHeight) / 2 - style.FramePadding.y + scale(2.0f)),
                Const::BrowserFolderLineColor);
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
}

void _BrowserWindow::processEmojiWindow()
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

                if (AlienImGui::Button("More", ImGui::GetContentRegionAvailWidth())) {
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

void _BrowserWindow::processEmojiButton(int emojiType)
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

void _BrowserWindow::processDownloadButton(BrowserLeaf const& leaf)
{
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::BrowserDownloadButtonTextColor);
    auto downloadButtonResult = processActionButton(ICON_FA_DOWNLOAD);
    ImGui::PopStyleColor();
    if (downloadButtonResult) {
        onDownloadItem(leaf);
    }
    AlienImGui::Tooltip("Download");
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

void _BrowserWindow::processShortenedText(std::string const& text, bool bold) {
    auto substrings = splitString(text);
    if (substrings.empty()) {
        return;
    }
    auto& styleRepository = StyleRepository::getInstance();
    auto textSize = ImGui::CalcTextSize(substrings.at(0).c_str());
    auto needDetailButton = textSize.x > ImGui::GetContentRegionAvailWidth() || substrings.size() > 1;
    auto cursorPos = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvailWidth() - styleRepository.scale(15.0f);
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

bool _BrowserWindow::processActionButton(std::string const& text)
{
    ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(Const::ToolbarButtonBackgroundColor));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImU32)Const::ToolbarButtonHoveredColor);
    auto result = ImGui::Button(text.c_str());
    ImGui::PopStyleColor(2);
   
    return result;
}

bool _BrowserWindow::processDetailButton()
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

void _BrowserWindow::processActivated()
{
    onRefresh();
}

void _BrowserWindow::scheduleCreateTreeTOs()
{
    _scheduleCreateSimulationTreeTOs = true;
    _scheduleCreateGenomeTreeTOs = true;
}

void _BrowserWindow::sortRawTOs(std::vector<NetworkResourceRawTO>& tos, ImGuiTableSortSpecs* sortSpecs)
{
    if (tos.size() > 1) {
        std::sort(tos.begin(), tos.end(), [&](auto const& left, auto const& right) {
            return _NetworkResourceRawTO::compare(left, right, sortSpecs) < 0;
        });
    }
}

void _BrowserWindow::sortUserList()
{
    std::sort(_userTOs.begin(), _userTOs.end(), [&](auto const& left, auto const& right) { return UserTO::compareOnlineAndTimestamp(left, right) > 0; });
}

void _BrowserWindow::filterRawTOs()
{
    _simulations.rawTOs.clear();
    _simulations.rawTOs.reserve(_allRawTOs.size());
    _genomes.rawTOs.clear();
    _genomes.rawTOs.reserve(_allRawTOs.size());
    for (auto const& to : _allRawTOs) {
        if (to->matchWithFilter(_filter) && _showCommunityCreations != to->fromRelease) {
            if (to->type == NetworkResourceType_Simulation) {
                _simulations.rawTOs.emplace_back(to);
            } else {
                _genomes.rawTOs.emplace_back(to);
            }
        }
    }
}

void _BrowserWindow::onDownloadItem(BrowserLeaf const& leaf)
{
    printOverlayMessage("Downloading ...");
    ++leaf.rawTO->numDownloads;

    delayedExecution([=, this] {
        auto& networkService = NetworkService::getInstance();
        std::string dataTypeString = _visibleResourceType == NetworkResourceType_Simulation ? "simulation" : "genome";
        SerializedSimulation serializedSim;
        if (!networkService.downloadSimulation(serializedSim.mainData, serializedSim.auxiliaryData, serializedSim.statistics, leaf.rawTO->id)) {
            MessageDialog::getInstance().information("Error", "Failed to download " + dataTypeString + ".");
            return;
        }

        if (_visibleResourceType == NetworkResourceType_Simulation) {
            DeserializedSimulation deserializedSim;
            if (!SerializerService::deserializeSimulationFromStrings(deserializedSim, serializedSim)) {
                MessageDialog::getInstance().information("Error", "Failed to load simulation. Your program version may not match.");
                return;
            }

            _simController->closeSimulation();

            std::optional<std::string> errorMessage;
            try {
                _simController->newSimulation(
                    deserializedSim.auxiliaryData.timestep, deserializedSim.auxiliaryData.generalSettings, deserializedSim.auxiliaryData.simulationParameters);
                _simController->setClusteredSimulationData(deserializedSim.mainData);
                _simController->setStatisticsHistory(deserializedSim.statistics);
            } catch (CudaMemoryAllocationException const& exception) {
                errorMessage = exception.what();
            } catch (...) {
                errorMessage = "Failed to load simulation.";
            }

            if (errorMessage) {
                showMessage("Error", *errorMessage);
                _simController->closeSimulation();
                _simController->newSimulation(
                    deserializedSim.auxiliaryData.timestep, deserializedSim.auxiliaryData.generalSettings, deserializedSim.auxiliaryData.simulationParameters);
            }

            _viewport->setCenterInWorldPos(deserializedSim.auxiliaryData.center);
            _viewport->setZoomFactor(deserializedSim.auxiliaryData.zoom);
            _temporalControlWindow->onSnapshot();

        } else {
            std::vector<uint8_t> genome;
            if (!SerializerService::deserializeGenomeFromString(genome, serializedSim.mainData)) {
                MessageDialog::getInstance().information("Error", "Failed to load genome. Your program version may not match.");
                return;
            }
            _editorController->setOn(true);
            _editorController->getGenomeEditorWindow()->openTab(GenomeDescriptionService::convertBytesToDescription(genome));
        }
        if (VersionChecker::isVersionNewer(leaf.rawTO->version)) {
            MessageDialog::getInstance().information(
                "Warning",
                "The download was successful but the " + dataTypeString +" was generated using a more recent\n"
                "version of ALIEN. Consequently, the " + dataTypeString + "might not function as expected.\n"
                "Please visit\n\nhttps://github.com/chrxh/alien\n\nto obtain the latest version.");
        }
    });
}

void _BrowserWindow::onDeleteItem(BrowserLeaf const& leaf)
{
    MessageDialog::getInstance().yesNo("Delete item", "Do you really want to delete the selected item?", [leaf, this]() {
        printOverlayMessage("Deleting ...");

        delayedExecution([leafCopy = leaf, this] {
            auto& networkService = NetworkService::getInstance();
            if (!networkService.deleteSimulation(leafCopy.rawTO->id)) {
                MessageDialog::getInstance().information("Error", "Failed to delete item. Please try again later.");
                return;
            }
            _scheduleRefresh = true;
        });
    });
}

void _BrowserWindow::onToggleLike(NetworkResourceTreeTO const& to, int emojiType)
{
    CHECK(to->isLeaf());
    auto& leaf = to->getLeaf();
    auto& networkService = NetworkService::getInstance();
    if (networkService.getLoggedInUserName()) {

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
        networkService.toggleLikeSimulation(leaf.rawTO->id, emojiType);
    } else {
        _loginDialog.lock()->open();
    }
}

void _BrowserWindow::onExpandFolders()
{
    if (_visibleResourceType == NetworkResourceType_Simulation) {
        _simulations.collapsedFolderNames.clear();
    } else {
        _genomes.collapsedFolderNames.clear();
    }
    scheduleCreateTreeTOs();
}

void _BrowserWindow::onCollapseFolders()
{
    if (_visibleResourceType == NetworkResourceType_Simulation) {
        auto folderNames = NetworkResourceService::getAllFolderNames(_simulations.rawTOs, 1);
        _simulations.collapsedFolderNames.insert(folderNames.begin(), folderNames.end());
    } else {
        auto folderNames = NetworkResourceService::getAllFolderNames(_genomes.rawTOs, 1);
        _genomes.collapsedFolderNames.insert(folderNames.begin(), folderNames.end());
    }
    scheduleCreateTreeTOs();
}

void _BrowserWindow::openWeblink(std::string const& link)
{
#ifdef _WIN32
    ShellExecute(NULL, "open", link.c_str(), NULL, NULL, SW_SHOWNORMAL);
#endif
}

bool _BrowserWindow::isLiked(std::string const& simId)
{
    return _ownEmojiTypeBySimId.contains(simId);
}

std::string _BrowserWindow::getUserNamesToEmojiType(std::string const& simId, int emojiType)
{
    auto& networkService = NetworkService::getInstance();

    std::set<std::string> userNames;

    auto findResult = _userNamesByEmojiTypeBySimIdCache.find(std::make_pair(simId, emojiType));
    if (findResult != _userNamesByEmojiTypeBySimIdCache.end()) {
        userNames = findResult->second;
    } else {
        networkService.getUserNamesForSimulationAndEmojiType(userNames, simId, emojiType);
        _userNamesByEmojiTypeBySimIdCache.emplace(std::make_pair(simId, emojiType), userNames);
    }

    return boost::algorithm::join(userNames, ", ");
}

void _BrowserWindow::pushTextColor(NetworkResourceTreeTO const& to)
{
    if (to->isLeaf()) {
        auto const& leaf = to->getLeaf();
        if (VersionChecker::isVersionOutdated(leaf.rawTO->version)) {
            ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::BrowserVersionOutdatedTextColor);
        } else if (VersionChecker::isVersionNewer(leaf.rawTO->version)) {
            ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::BrowserVersionNewerTextColor);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::BrowserVersionOkTextColor);
        }
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::BrowserFolderTextColor);
    }
}

void _BrowserWindow::popTextColor()
{
    ImGui::PopStyleColor();
}
