#pragma once

#include <chrono>

#include "Base/Hashes.h"
#include "EngineInterface/Definitions.h"
#include "Network/BrowserDataTO.h"
#include "Network/NetworkDataTO.h"
#include "Network/UserTO.h"

#include "AlienWindow.h"
#include "Definitions.h"

class _BrowserWindow : public _AlienWindow
{
public:
    _BrowserWindow(
        SimulationController const& simController,
        NetworkService const& networkController,
        StatisticsWindow const& statisticsWindow,
        Viewport const& viewport,
        TemporalControlWindow const& temporalControlWindow,
        EditorController const& editorController);
    ~_BrowserWindow();

    void registerCyclicReferences(LoginDialogWeakPtr const& loginDialog, UploadSimulationDialogWeakPtr const& uploadSimulationDialog);

    void onRefresh();

private:
    void refreshIntern(bool withRetry);

    void processIntern() override;
    void processBackground() override;

    void processSimulationList();
    void processGenomeList();
    void processUserList();

    void processStatus();
    void processFilter();
    void processToolbar();

    void processEmojiWindow();
    void processEmojiButton(int emojiType);
    void processEmojiList(BrowserDataTO const& to);

    void processActionButtons(BrowserDataTO const& to);

    void processShortenedText(std::string const& text, bool bold = false);
    bool processActionButton(std::string const& text);
    bool processDetailButton();

    void processActivated() override;

    void sortRemoteSimulationData(std::vector<NetworkDataTO>& remoteData, ImGuiTableSortSpecs* sortSpecs);
    void sortUserList();

    void onDownloadItem(BrowserLeaf const& leaf);
    void onDeleteItem(BrowserLeaf const& leaf);
    void onToggleLike(BrowserDataTO const& to, int emojiType);
    void openWeblink(std::string const& link);

    bool isLiked(std::string const& simId);
    std::string getUserNamesToEmojiType(std::string const& simId, int emojiType);

    void pushTextColor(BrowserDataTO const& to);
    void calcFilteredSimulationAndGenomeLists();

    DataType _selectedDataType = DataType_Simulation; 
    bool _scheduleRefresh = false;
    bool _scheduleCreateBrowserData = false;
    std::string _filter;
    bool _showCommunityCreations = false;
    float _userTableWidth = 0;
    std::unordered_set<std::string> _selectionIds;
    std::unordered_map<std::string, int> _ownEmojiTypeBySimId;
    std::unordered_map<std::pair<std::string, int>, std::set<std::string>> _userNamesByEmojiTypeBySimIdCache;

    int _numSimulations = 0;
    int _numGenomes = 0;
    std::vector<NetworkDataTO> _rawNetworkDataTOs;
    std::vector<NetworkDataTO> _filteredNetworkSimulationTOs;
    std::vector<NetworkDataTO> _filteredNetworkGenomeTOs;

    std::vector<BrowserDataTO> _browserSimulationTOs;
    std::vector<BrowserDataTO> _browserGenomeTOs;

    std::vector<UserTO> _userTOs;

    std::vector<TextureData> _emojis;

    bool _activateEmojiPopup = false;
    bool _showAllEmojis = false;
    BrowserDataTO _emojiPopupTO;

    std::optional<std::chrono::steady_clock::time_point> _lastRefreshTime;

    SimulationController _simController;
    NetworkService _networkService;
    StatisticsWindow _statisticsWindow;
    Viewport _viewport;
    TemporalControlWindow _temporalControlWindow;
    LoginDialogWeakPtr _loginDialog;
    EditorController _editorController;
    UploadSimulationDialogWeakPtr _uploadSimulationDialog;
};
