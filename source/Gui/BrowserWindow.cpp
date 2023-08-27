#include "BrowserWindow.h"

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/regex.hpp>
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
#include "OverlayMessageController.h"
#include "VersionChecker.h"

namespace
{
    auto constexpr UserTableWidth = 200.0f;
    auto constexpr BrowserBottomHeight = 68.0f;
    auto constexpr RowHeight = 25.0f;
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
}

_BrowserWindow::~_BrowserWindow()
{
    GlobalSettings::getInstance().setBoolState("windows.browser.show community creations", _showCommunityCreations);
    GlobalSettings::getInstance().setBoolState("windows.browser.first start", false);
}

void _BrowserWindow::registerCyclicReferences(LoginDialogWeakPtr const& loginDialog, UploadSimulationDialogWeakPtr const& uploadSimulationDialog)
{
    _loginDialog = loginDialog;
    _uploadSimulationDialog = uploadSimulationDialog;

    auto firstStart = GlobalSettings::getInstance().getBoolState("windows.browser.first start", true);
    refreshIntern(firstStart);
    _showCommunityCreations = GlobalSettings::getInstance().getBoolState("windows.browser.show community creations", _showCommunityCreations);
}

void _BrowserWindow::onRefresh()
{
    refreshIntern(true);
}

void _BrowserWindow::refreshIntern(bool withRetry)
{
    try {
        bool success = _networkController->getSimulationDataList(_remoteSimulationList, withRetry);
        success &= _networkController->getUserList(_userList, withRetry);

        if (!success) {
            if (withRetry) {
                MessageDialog::getInstance().show("Error", "Failed to retrieve browser data. Please try again.");
            }
        }
        calcFilteredSimulationDatas();

        if (_networkController->getLoggedInUserName()) {
            std::vector<std::string> likedIds;
            if (!_networkController->getLikedSimulationIdList(likedIds)) {
                MessageDialog::getInstance().show("Error", "Failed to retrieve browser data. Please try again.");
            }
            _likedIds = std::unordered_set<std::string>(likedIds.begin(), likedIds.end());
        } else {
            _likedIds.clear();
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
                ImVec2(sizeAvailable.x - scale(UserTableWidth), sizeAvailable.y - scale(BrowserBottomHeight)),
                false,
                ImGuiWindowFlags_HorizontalScrollbar)) {
            processSimulationTable();
        }
        ImGui::EndChild();
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
        ImGui::TableSetupColumn("Stars", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Likes);
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
                    if (_networkController->getLoggedInUserName()) {
                        onToggleLike(*item);
                    } else {
                        _loginDialog.lock()->open();
                    }
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
                AlienImGui::Text(std::to_string(item->likes));
                if(item->likes > 0) {
                    ImGui::SameLine();
                    processDetailButton();
                    AlienImGui::Tooltip([&] { return getUserLikes(item->id); }, false);
                }
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
        ImGui::TableSetupColumn("Time spent", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(80.0f));
        auto isLoggedIn = _networkController->getLoggedInUserName().has_value();
        ImGui::TableSetupColumn(isLoggedIn ? "GPU model" : "GPU model (visible if logged in)", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(200.0f));
        ImGui::TableSetupColumn(
            "Stars received", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, scale(100.0f));
        ImGui::TableSetupColumn("Stars given", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, styleRepository.scale(100.0f));
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
                }
                processShortenedText(item->userName, isBoldFont);

                ImGui::TableNextColumn();
                if (item->timeSpent > 0) {
                    processShortenedText(StringHelper::format(item->timeSpent) + "h", isBoldFont);
                }

                ImGui::TableNextColumn();
                if (isLoggedIn && _loginDialog.lock()->isShareGpuInfo()) {
                    processShortenedText(item->gpu, isBoldFont);
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
        statusText += std::to_string(_remoteSimulationList.size()) + " simulations found";

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
    std::sort(_userList.begin(), _userList.end(), [&](auto const& left, auto const& right) { return UserData::compare(left, right) > 0; });
}

void _BrowserWindow::onDownloadSimulation(RemoteSimulationData* remoteData)
{
    printOverlayMessage("Downloading ...");

    delayedExecution([=, this] {
        SerializedSimulation serializedSim;
        if (!_networkController->downloadSimulation(serializedSim.mainData, serializedSim.auxiliaryData, remoteData->id)) {
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
            errorMessage = std::string("Failed to load simulation: ") + exception.what();
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

        if (VersionChecker::isVersionNewer(remoteData->version)) {
            MessageDialog::getInstance().show(
                "Warning",
                "The download was successful but the simulation was generated using a more recent\n"
                "version of ALIEN. Consequently, the simulation might not function as expected.\n"
                "Please visit\n\nhttps://github.com/chrxh/alien\n\nto obtain the latest version.");
        }
    });
}

void _BrowserWindow::onDeleteSimulation(RemoteSimulationData* remoteData)
{
    printOverlayMessage("Deleting ...");

    delayedExecution([remoteData = remoteData, this] {
        if (!_networkController->deleteSimulation(remoteData->id)) {
            MessageDialog::getInstance().show("Error", "Failed to delete simulation. Please try again later.");
            return;
        }
        _scheduleRefresh = true;
    });
}

void _BrowserWindow::onToggleLike(RemoteSimulationData& entry)
{
    auto findResult = _likedIds.find(entry.id);
    if (findResult != _likedIds.end()) {
        _likedIds.erase(findResult);
        --entry.likes;
    } else {
        _likedIds.insert(entry.id);
        ++entry.likes;
    }
    _userLikesByIdCache.erase(entry.id); //invalidate cache entry
    _networkController->toggleLikeSimulation(entry.id);
    sortSimulationList();
}

bool _BrowserWindow::isLiked(std::string const& id)
{
    return _likedIds.find(id) != _likedIds.end();
}

std::string _BrowserWindow::getUserLikes(std::string const& id)
{
    std::set<std::string> userLikes;

    auto findResult = _userLikesByIdCache.find(id);
    if (findResult != _userLikesByIdCache.end()) {
        userLikes = findResult->second;
    } else {
        _networkController->getUserLikesForSimulation(userLikes, id);
        _userLikesByIdCache.emplace(id, userLikes);
    }

    return boost::algorithm::join(userLikes, ", ");
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
    _filteredRemoteSimulationList.reserve(_remoteSimulationList.size());
    for (auto const& simData : _remoteSimulationList) {
        if (simData.matchWithFilter(_filter) &&_showCommunityCreations != simData.fromRelease) {
            _filteredRemoteSimulationList.emplace_back(simData);
        }
    }
}
