#include "BrowserWindow.h"

#include <ranges>

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/Resources.h"
#include "Base/StringHelper.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/SimulationController.h"

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "StyleRepository.h"
#include "NetworkDataParser.h"
#include "NetworkController.h"
#include "StatisticsWindow.h"
#include "Viewport.h"
#include "TemporalControlWindow.h"
#include "MessageDialog.h"
#include "LoginDialog.h"
#include "UploadSimulationDialog.h"
#include "DelayedExecutionController.h"
#include "OpenGLHelper.h"
#include "OverlayMessageController.h"
#include "VersionChecker.h"

namespace
{
    auto constexpr UserTableWidth = 200.0f;
    auto constexpr BrowserBottomHeight = 68.0f;
    auto constexpr RowHeight = 25.0f;

    auto constexpr NumEmojis = 16;
    auto constexpr NumEmojisPerRow = 6;
}

_BrowserWindow::_BrowserWindow(
    SimulationController const& simController,
    NetworkController const& networkController,
    StatisticsWindow const& statisticsWindow,
    Viewport const& viewport,
    TemporalControlWindow const& temporalControlWindow)
    : _AlienWindow("Browser", "windows.browser", true)
    , _simController(simController)
    , _networkController(networkController)
    , _statisticsWindow(statisticsWindow)
    , _viewport(viewport)
    , _temporalControlWindow(temporalControlWindow)
{
    _showCommunityCreations = GlobalSettings::getInstance().getBoolState("windows.browser.show community creations", _showCommunityCreations);
    _userTableWidth = GlobalSettings::getInstance().getFloatState("windows.browser.user table width", scale(UserTableWidth));

    for (int i = 1; i <= NumEmojis; ++i) {
        _emojis.emplace_back(OpenGLHelper::loadTexture(Const::BasePath + "emoji" + std::to_string(i) + ".png"));
    }
}

_BrowserWindow::~_BrowserWindow()
{
    GlobalSettings::getInstance().setBoolState("windows.browser.show community creations", _showCommunityCreations);
    GlobalSettings::getInstance().setBoolState("windows.browser.first start", false);
    GlobalSettings::getInstance().setFloatState("windows.browser.user table width", _userTableWidth);
}

void _BrowserWindow::registerCyclicReferences(LoginDialogWeakPtr const& loginDialog, UploadSimulationDialogWeakPtr const& uploadSimulationDialog)
{
    _loginDialog = loginDialog;
    _uploadSimulationDialog = uploadSimulationDialog;

    auto firstStart = GlobalSettings::getInstance().getBoolState("windows.browser.first start", true);
    refreshIntern(firstStart);
}

void _BrowserWindow::onRefresh()
{
    refreshIntern(true);
}

void _BrowserWindow::refreshIntern(bool withRetry)
{
    try {
        bool success = _networkController->getRemoteSimulationList(_rawRemoteSimulationList, withRetry);
        success &= _networkController->getUserList(_userList, withRetry);

        if (!success) {
            if (withRetry) {
                MessageDialog::getInstance().show("Error", "Failed to retrieve browser data. Please try again.");
            }
        }
        calcFilteredSimulationDatas();

        if (_networkController->getLoggedInUserName()) {
            if (!_networkController->getLikeTypeBySimId(_ownLikeTypeBySimId)) {
                MessageDialog::getInstance().show("Error", "Failed to retrieve browser data. Please try again.");
            }
        } else {
            _ownLikeTypeBySimId.clear();
        }

        sortSimulationList();
        sortUserList();
    } catch (std::exception const& e) {
        if (withRetry) {
            MessageDialog::getInstance().show("Error", e.what());
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
            processSimulationTable();
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
            processUserTable();
        }
        ImGui::EndChild();
    }

    processStatus();
    processFilter();
    processEmojiWindow();

    if(_scheduleRefresh) {
        onRefresh();
        _scheduleRefresh = false;
    }
}

void _BrowserWindow::processToolbar()
{
    if (AlienImGui::ToolbarButton(ICON_FA_SYNC)) {
        onRefresh();
    }
    AlienImGui::Tooltip("Refresh");

    ImGui::SameLine();
    ImGui::BeginDisabled(_networkController->getLoggedInUserName().has_value());
    if (AlienImGui::ToolbarButton(ICON_FA_SIGN_IN_ALT)) {
        if (auto loginDialog = _loginDialog.lock()) {
            loginDialog->open();
        }
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Login or register");

    ImGui::SameLine();
    ImGui::BeginDisabled(!_networkController->getLoggedInUserName());
    if (AlienImGui::ToolbarButton(ICON_FA_SIGN_OUT_ALT)) {
        if (auto loginDialog = _loginDialog.lock()) {
            _networkController->logout();
            onRefresh();
        }
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Logout");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_SHARE_ALT)) {
        if (_networkController->getLoggedInUserName()) {
            if (auto uploadSimulationDialog = _uploadSimulationDialog.lock()) {
                uploadSimulationDialog->open();
            }
        } else {
            _loginDialog.lock()->open();
        }
    }
    AlienImGui::Tooltip("Share your simulation with other users:\nYour current simulation will be uploaded to the server and made visible in the browser.");
    AlienImGui::Separator();
}

void _BrowserWindow::processSimulationTable()
{
    ImGui::PushID("SimTable");
    auto styleRepository = StyleRepository::getInstance();
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable
        | ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody
        | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    AlienImGui::Group("Simulations");
    if (ImGui::BeginTable("Browser", 12, flags, ImVec2(0, 0), 0.0f)) {
        ImGui::TableSetupColumn(
            "Actions",
            ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed,
            scale(90.0f),
            RemoteSimulationDataColumnId_Actions);
        ImGui::TableSetupColumn(
            "Timestamp",
            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending,
            scale(135.0f),
            RemoteSimulationDataColumnId_Timestamp);
        ImGui::TableSetupColumn(
            "User name",
            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed,
            styleRepository.scale(120.0f),
            RemoteSimulationDataColumnId_UserName);
        ImGui::TableSetupColumn(
            "Simulation name",
            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed,
            styleRepository.scale(160.0f),
            RemoteSimulationDataColumnId_SimulationName);
        ImGui::TableSetupColumn(
            "Description",
            ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed,
            styleRepository.scale(120.0f),
            RemoteSimulationDataColumnId_Description);
        ImGui::TableSetupColumn("Reactions", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Likes);
        ImGui::TableSetupColumn("Downloads", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_NumDownloads);
        ImGui::TableSetupColumn("Width", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Width);
        ImGui::TableSetupColumn("Height", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Height);
        ImGui::TableSetupColumn("Objects", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Particles);
        ImGui::TableSetupColumn("File size", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_FileSize);
        ImGui::TableSetupColumn("Version", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Version);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        //sort our data if sort specs have been changed!
        if (ImGuiTableSortSpecs* sortSpecs = ImGui::TableGetSortSpecs()) {
            if (sortSpecs->SpecsDirty || _scheduleSort) {
                if (_filteredRemoteSimulationList.size() > 1) {
                    std::sort(_filteredRemoteSimulationList.begin(), _filteredRemoteSimulationList.end(), [&](auto const& left, auto const& right) {
                        return RemoteSimulationData::compare(&left, &right, sortSpecs) < 0;
                    });
                }
                sortSpecs->SpecsDirty = false;
            }
        }
        ImGuiListClipper clipper;
        clipper.Begin(_filteredRemoteSimulationList.size());
        while (clipper.Step())
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {

                RemoteSimulationData* item = &_filteredRemoteSimulationList[row];

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(RowHeight));

                ImGui::TableNextColumn();
                if (processActionButton(ICON_FA_DOWNLOAD)) {
                    onDownloadSimulation(item);
                }
                AlienImGui::Tooltip("Download");

                ImGui::SameLine();
                auto liked = isLiked(item->id);
                if (liked) {
                    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::LikeTextColor);
                }
                if (processActionButton(ICON_FA_STAR)) {
                    //if (_networkController->getLoggedInUserName()) {
                    //    onToggleLike(*item);
                    //} else {
                    //    _loginDialog.lock()->open();
                    //}
                    _activateEmojiPopup = true;
                    _simIndexOfEmojiPopup = row;
                }
                AlienImGui::Tooltip("Give a star");

                if (liked) {
                    ImGui::PopStyleColor(1);
                }
                ImGui::SameLine();
                ImGui::BeginDisabled(item->userName != _networkController->getLoggedInUserName().value_or(""));
                if (processActionButton(ICON_FA_TRASH)) {
                    onDeleteSimulation(item);
                }
                ImGui::EndDisabled();
                AlienImGui::Tooltip("Delete");

                ImGui::TableNextColumn();

                pushTextColor(*item);

                AlienImGui::Text(item->timestamp);
                ImGui::TableNextColumn();
                processShortenedText(item->userName);
                ImGui::TableNextColumn();
                processShortenedText(item->simName);
                ImGui::TableNextColumn();
                processShortenedText(item->description);
                ImGui::TableNextColumn();
                processEmojiList(item);

                //if (item->getTotalLikes() > 0) {
                //    processDetailButton();
                //    AlienImGui::Tooltip([&] { return getUserLikes(item->id); }, false);
                //}
                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(item->numDownloads));
                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(item->width));
                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(item->height));
                ImGui::TableNextColumn();
                AlienImGui::Text(StringHelper::format(item->particles / 1000) + " K");
                ImGui::TableNextColumn();
                AlienImGui::Text(StringHelper::format(item->contentSize / 1024) + " KB");
                ImGui::TableNextColumn();
                AlienImGui::Text(item->version);

                ImGui::PopStyleColor();
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

void _BrowserWindow::processUserTable()
{
    ImGui::PushID("UserTable");
    auto styleRepository = StyleRepository::getInstance();
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable
         | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody
        | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    AlienImGui::Group("Simulators");
    if (ImGui::BeginTable("Browser", 5, flags, ImVec2(0, 0), 0.0f)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed, scale(90.0f));
        auto isLoggedIn = _networkController->getLoggedInUserName().has_value();
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
        clipper.Begin(_userList.size());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                auto item = &_userList[row];

                ImGui::PushID(row);
                ImGui::TableNextRow(0, scale(RowHeight));

                ImGui::TableNextColumn();
                auto isBoldFont = isLoggedIn && *_networkController->getLoggedInUserName() == item->userName;

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
    auto styleRepository = StyleRepository::getInstance();

    if (ImGui::BeginChild("##", ImVec2(0, styleRepository.scale(33.0f)), true)) {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::MonospaceColor);
        std::string statusText;
        statusText += std::string(" " ICON_FA_INFO_CIRCLE " ");
        statusText += std::to_string(_rawRemoteSimulationList.size()) + " simulations found";

        statusText += std::string(" " ICON_FA_INFO_CIRCLE " ");
        statusText += std::to_string(_userList.size()) + " simulators found";

        statusText += std::string("  " ICON_FA_INFO_CIRCLE " ");
        if (auto userName = _networkController->getLoggedInUserName()) {
            statusText += "Logged in as " + *userName + " @ " + _networkController->getServerAddress();// + ": ";
        } else {
            statusText += "Not logged in to " + _networkController->getServerAddress();// + ": ";
        }

        if (!_networkController->getLoggedInUserName()) {
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
        calcFilteredSimulationDatas();
    }
    ImGui::SameLine();
    if (AlienImGui::InputText(AlienImGui::InputTextParameters().name("Filter"), _filter)) {
        calcFilteredSimulationDatas();
    }
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
        for (auto const& [likeType, emoji] : _emojis | boost::adaptors::indexed(0)) {
            if (likeType % NumEmojisPerRow != 0) {
                ImGui::SameLine();
            }
            ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(Const::ToolbarButtonBackgroundColor));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, static_cast<ImVec4>(Const::ToolbarButtonHoveredColor));
            auto cursorPos = ImGui::GetCursorScreenPos();
            auto emojiWidth = scale(toFloat(emoji.width));
            auto emojiHeight = scale(toFloat(emoji.height));
            auto& sim = _filteredRemoteSimulationList.at(_simIndexOfEmojiPopup);
            if (ImGui::ImageButton((void*)(intptr_t)emoji.textureId, {emojiWidth, emojiHeight}, {0, 0}, {1.0f, 1.0f})) {
                onToggleLike(sim, toInt(likeType));
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleColor(2);

            bool isLiked = _ownLikeTypeBySimId.contains(sim.id) && _ownLikeTypeBySimId.at(sim.id) == likeType;
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
        ImGui::EndPopup();
    }
}

void _BrowserWindow::processEmojiList(RemoteSimulationData* sim)
{
    std::map<int, int> remap;
    std::set<int> processedLikeTypes;

    int index = 0;
    while (processedLikeTypes.size() < sim->numLikesByLikeType.size()) {
        int maxLikes = 0;
        std::optional<int> maxLikeType;
        for (auto const& [likeType, numLikes] : sim->numLikesByLikeType) {
            if (!processedLikeTypes.contains(likeType) && numLikes > maxLikes) {
                maxLikes = numLikes;
                maxLikeType = likeType;
            }
        }
        processedLikeTypes.insert(*maxLikeType);
        remap.emplace(index, *maxLikeType);
        ++index;
    }

    int counter = 0;
    std::optional<int> toggleLikeType;
    for (auto const& likeType : remap | std::views::values) {
        //auto likeType = remap.at(index);
        auto numLikes = sim->numLikesByLikeType.at(likeType);

        AlienImGui::Text(std::to_string(numLikes));
        ImGui::SameLine();
        if (likeType < _emojis.size()) {
            auto const& emoji = _emojis.at(likeType);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() - scale(7.0f));
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + scale(1.0f));
            ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(Const::ToolbarButtonBackgroundColor));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, static_cast<ImVec4>(Const::ToolbarButtonHoveredColor));
            auto cursorPos = ImGui::GetCursorScreenPos();
            auto emojiWidth = scale(toFloat(emoji.width) / 2.5f);
            auto emojiHeight = scale(toFloat(emoji.height) / 2.5f);
            if (ImGui::ImageButton((void*)(intptr_t)emoji.textureId, {emojiWidth, emojiHeight},
                    ImVec2(0, 0),
                    ImVec2(1, 1),
                    0)) {
                toggleLikeType = likeType;
            }
            bool isLiked = _ownLikeTypeBySimId.contains(sim->id) && _ownLikeTypeBySimId.at(sim->id) == likeType;
            if (isLiked) {
                ImDrawList* drawList = ImGui::GetWindowDrawList();
                drawList->AddRect(
                    ImVec2(cursorPos.x, cursorPos.y), ImVec2(cursorPos.x + emojiWidth, cursorPos.y + emojiHeight), (ImU32)ImColor::HSV(0, 0, 1, 0.5f), 1.0f);
            }
            ImGui::PopStyleColor(2);
        }

        //separator except for last element
        if (++counter < sim->numLikesByLikeType.size()) {
            //ImGui::SameLine();
            //AlienImGui::VerticalSeparator(20.0f);
            ImGui::SameLine();
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() - scale(4.0f));
        }
    }
    if (toggleLikeType) {
        onToggleLike(*sim, *toggleLikeType);
    }
}

namespace
{
    std::vector<std::string> splitString(const std::string& str)
    {
        std::vector<std::string> tokens;
        boost::algorithm::split_regex(tokens, str, boost::regex("(\r\n)+"));
        return tokens;
    }
}

void _BrowserWindow::processShortenedText(std::string const& text, bool bold) {
    auto substrings = splitString(text);
    if (substrings.empty()) {
        return;
    }
    auto styleRepository = StyleRepository::getInstance();
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
    auto result = ImGui::Button(text.c_str());
    ImGui::PopStyleColor();
   
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

void _BrowserWindow::sortSimulationList()
{
    _scheduleSort = true;
}

void _BrowserWindow::sortUserList()
{
    std::sort(_userList.begin(), _userList.end(), [&](auto const& left, auto const& right) { return UserData::compareOnlineAndTimestamp(left, right) > 0; });
}

void _BrowserWindow::onDownloadSimulation(RemoteSimulationData* sim)
{
    printOverlayMessage("Downloading ...");

    delayedExecution([=, this] {
        SerializedSimulation serializedSim;
        if (!_networkController->downloadSimulation(serializedSim.mainData, serializedSim.auxiliaryData, sim->id)) {
            MessageDialog::getInstance().show("Error", "Failed to download simulation.");
            return;
        }

        DeserializedSimulation deserializedSim;
        if (!Serializer::deserializeSimulationFromStrings(deserializedSim, serializedSim)) {
            MessageDialog::getInstance().show("Error", "Failed to load simulation. Your program version may not match.");
            return;
        }

        _simController->closeSimulation();
        _statisticsWindow->reset();

        std::optional<std::string> errorMessage;
        try {
            _simController->newSimulation(
                deserializedSim.auxiliaryData.timestep, deserializedSim.auxiliaryData.generalSettings, deserializedSim.auxiliaryData.simulationParameters);
            _simController->setClusteredSimulationData(deserializedSim.mainData);
        } catch (CudaMemoryAllocationException const& exception) {
            errorMessage = exception.what();
        } catch (...) {
            errorMessage = "Failed to load simulation.";
        }

        if (errorMessage) {
            printMessage("Error", *errorMessage);
            _simController->closeSimulation();
            _simController->newSimulation(
                deserializedSim.auxiliaryData.timestep, deserializedSim.auxiliaryData.generalSettings, deserializedSim.auxiliaryData.simulationParameters);
        }

        _viewport->setCenterInWorldPos(deserializedSim.auxiliaryData.center);
        _viewport->setZoomFactor(deserializedSim.auxiliaryData.zoom);
        _temporalControlWindow->onSnapshot();

        if (VersionChecker::isVersionNewer(sim->version)) {
            MessageDialog::getInstance().show(
                "Warning",
                "The download was successful but the simulation was generated using a more recent\n"
                "version of ALIEN. Consequently, the simulation might not function as expected.\n"
                "Please visit\n\nhttps://github.com/chrxh/alien\n\nto obtain the latest version.");
        }
    });
}

void _BrowserWindow::onDeleteSimulation(RemoteSimulationData* sim)
{
    printOverlayMessage("Deleting ...");

    delayedExecution([remoteData = sim, this] {
        if (!_networkController->deleteSimulation(remoteData->id)) {
            MessageDialog::getInstance().show("Error", "Failed to delete simulation. Please try again later.");
            return;
        }
        _scheduleRefresh = true;
    });
}

void _BrowserWindow::onToggleLike(RemoteSimulationData& sim, int likeType)
{
    if (_networkController->getLoggedInUserName()) {

        //remove existing like
        auto findResult = _ownLikeTypeBySimId.find(sim.id);
        auto onlyRemoveLike = false;
        if (findResult != _ownLikeTypeBySimId.end()) {
            auto origLikeType = findResult->second;
            if (--sim.numLikesByLikeType[origLikeType] == 0) {
                sim.numLikesByLikeType.erase(origLikeType);
            }
            _ownLikeTypeBySimId.erase(findResult);
            //_userNamesByLikeTypeBySimIdCache.erase(std::make_pair(entry.id, origLikeType));  //invalidate cache entry
            onlyRemoveLike = origLikeType == likeType;  //remove like if same like icon has been clicked
        }

        //create new like
        if (!onlyRemoveLike) {
            _ownLikeTypeBySimId[sim.id] = likeType;
            if (sim.numLikesByLikeType.contains(likeType)) {
                ++sim.numLikesByLikeType[likeType];
            } else {
                sim.numLikesByLikeType[likeType] = 1;
            }
        }

        //_userNamesByLikeTypeBySimIdCache.erase(std::make_pair(entry.id, likeType));  //invalidate cache entry
        _networkController->toggleLikeSimulation(sim.id, likeType);
        sortSimulationList();
    } else {
        _loginDialog.lock()->open();
    }
}

bool _BrowserWindow::isLiked(std::string const& simId)
{
    return _ownLikeTypeBySimId.contains(simId);
}

std::string _BrowserWindow::getUserNamesToLikeType(std::string const& simId, int likeType)
{
    std::set<std::string> userNames;

    //auto findResult = _userNamesByLikeTypeBySimIdCache.find(std::make_pair(simId, likeType));
    //if (findResult != _userNamesByLikeTypeBySimIdCache.end()) {
    //    userNames = findResult->second;
    //} else {
    //    _networkController->getUserNamesForSimulationAndLikeType(userNames, simId, likeType);
    //    _userNamesByLikeTypeBySimIdCache.emplace(std::make_pair(simId, likeType), userNames);
    //}

    return boost::algorithm::join(userNames, ", ");
}

void _BrowserWindow::pushTextColor(RemoteSimulationData const& entry)
{
    if (VersionChecker::isVersionOutdated(entry.version)) {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::VersionOutdatedColor);
    } else if (VersionChecker::isVersionNewer(entry.version)) {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::VersionNewerColor);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::VersionOkColor);
    }
}

void _BrowserWindow::calcFilteredSimulationDatas()
{
    _filteredRemoteSimulationList.clear();
    _filteredRemoteSimulationList.reserve(_rawRemoteSimulationList.size());
    for (auto const& simData : _rawRemoteSimulationList) {
        if (simData.matchWithFilter(_filter) &&_showCommunityCreations != simData.fromRelease) {
            _filteredRemoteSimulationList.emplace_back(simData);
        }
    }
}
