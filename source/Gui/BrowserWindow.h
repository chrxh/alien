#pragma once

#include "EngineInterface/Definitions.h"

#include "AlienWindow.h"
#include "RemoteSimulationData.h"
#include "UserData.h"
#include "Definitions.h"

class _BrowserWindow : public _AlienWindow
{
public:
    _BrowserWindow(
        SimulationController const& simController,
        NetworkController const& networkController,
        StatisticsWindow const& statisticsWindow,
        Viewport const& viewport,
        TemporalControlWindow const& temporalControlWindow);
    ~_BrowserWindow();

    void registerCyclicReferences(LoginDialogWeakPtr const& loginDialog, UploadSimulationDialogWeakPtr const& uploadSimulationDialog);

    void onRefresh();

private:
    void refreshIntern(bool withRetry);

    void processIntern() override;

    void processSimulationTable();
    void processUserTable();

    void processStatus();
    void processFilter();
    void processToolbar();

    void processEmojiWindow();
    void processEmojiList(RemoteSimulationData* sim);

    void processShortenedText(std::string const& text, bool bold = false);
    bool processActionButton(std::string const& text);
    bool processDetailButton();

    void processActivated() override;

    void sortSimulationList();
    void sortUserList();

    void onDownloadSimulation(RemoteSimulationData* sim);
    void onDeleteSimulation(RemoteSimulationData* sim);
    void onToggleLike(RemoteSimulationData& sim, int likeType);

    bool isLiked(std::string const& simId);
    std::string getUserNamesToLikeType(std::string const& simId, int likeType);

    void pushTextColor(RemoteSimulationData const& entry);
    void calcFilteredSimulationDatas();

    bool _scheduleRefresh = false;
    bool _scheduleSort = false;
    std::string _filter;
    bool _showCommunityCreations = false;
    float _userTableWidth = 0;
    std::unordered_set<std::string> _selectionIds;
    std::unordered_map<std::string, int> _ownLikeTypeBySimId;
//    std::unordered_map<std::pair<std::string, int>, std::set<std::string>> _userNamesByLikeTypeBySimIdCache;
    std::vector<RemoteSimulationData> _rawRemoteSimulationList;
    std::vector<RemoteSimulationData> _filteredRemoteSimulationList;
    std::vector<UserData> _userList;

    std::vector<TextureData> _emojis;

    bool _activateEmojiPopup = false;
    int _simIndexOfEmojiPopup = 0;  //index in _filteredRemoteSimulationList

    SimulationController _simController;
    NetworkController _networkController;
    StatisticsWindow _statisticsWindow;
    Viewport _viewport;
    TemporalControlWindow _temporalControlWindow;
    LoginDialogWeakPtr _loginDialog;
    UploadSimulationDialogWeakPtr _uploadSimulationDialog;
};
