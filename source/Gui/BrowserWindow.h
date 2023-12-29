#pragma once

#include <chrono>

#include "Base/Hashes.h"
#include "EngineInterface/Definitions.h"
#include "Network/NetworkResourceTreeTO.h"
#include "Network/NetworkResourceRawTO.h"
#include "Network/UserTO.h"

#include "AlienWindow.h"
#include "Definitions.h"

class _BrowserWindow : public _AlienWindow
{
public:
    _BrowserWindow(
        SimulationController const& simController,
        StatisticsWindow const& statisticsWindow,
        Viewport const& viewport,
        TemporalControlWindow const& temporalControlWindow,
        EditorController const& editorController);
    ~_BrowserWindow();

    void registerCyclicReferences(LoginDialogWeakPtr const& loginDialog, UploadSimulationDialogWeakPtr const& uploadSimulationDialog);

    void onRefresh();

private:
    struct ResourceData
    {
        int numResources = 0;
        std::vector<NetworkResourceRawTO> rawTOs;
        std::vector<NetworkResourceTreeTO> treeTOs;
        std::set<std::vector<std::string>> collapsedFolderNames;
        NetworkResourceTreeTO selected;
    };

    void refreshIntern(bool withRetry);

    void processIntern() override;
    void processBackground() override;

    void processSimulationList();
    void processGenomeList();
    void processUserList();

    void processStatus();
    void processFilter();
    void processToolbar();

    void processResourceNameField(NetworkResourceTreeTO const& treeTO, std::set<std::vector<std::string>>& collapsedFolderNames);
    void processDescriptionField(NetworkResourceTreeTO const& treeTO);
    void processReactionList(NetworkResourceTreeTO const& treeTO);
    void processTimestampField(NetworkResourceTreeTO const& treeTO);
    void processUserNameField(NetworkResourceTreeTO const& treeTO);
    void processNumDownloadsField(NetworkResourceTreeTO const& treeTO);
    void processWidthField(NetworkResourceTreeTO const& treeTO);
    void processHeightField(NetworkResourceTreeTO const& treeTO);
    void processNumParticlesField(NetworkResourceTreeTO const& treeTO);
    void processSizeField(NetworkResourceTreeTO const& treeTO, bool kbyte);
    void processVersionField(NetworkResourceTreeTO const& treeTO);

    void processFolderTreeSymbols(NetworkResourceTreeTO const& treeTO, std::set<std::vector<std::string>>& collapsedFolderNames);
    void processEmojiWindow();
    void processEmojiButton(int emojiType);

    void processDownloadButton(BrowserLeaf const& leaf);
    void processActionButtons(NetworkResourceTreeTO const& to);

    void processShortenedText(std::string const& text, bool bold = false);
    bool processActionButton(std::string const& text);
    bool processDetailButton();

    void processActivated() override;

    void sortRawTOs(std::vector<NetworkResourceRawTO>& tos, ImGuiTableSortSpecs* sortSpecs);
    void sortUserList();

    void filterRawTOs();

    void onDownloadItem(BrowserLeaf const& leaf);
    void onDeleteItem(BrowserLeaf const& leaf);
    void onToggleLike(NetworkResourceTreeTO const& to, int emojiType);
    void openWeblink(std::string const& link);

    bool isLiked(std::string const& simId);
    std::string getUserNamesToEmojiType(std::string const& simId, int emojiType);

    void pushTextColor(NetworkResourceTreeTO const& to);
    void popTextColor();

    NetworkResourceType _selectedDataType = NetworkResourceType_Simulation; 
    bool _scheduleRefresh = false;
    bool _scheduleCreateTreeTOs = false;

    std::string _filter;
    bool _showCommunityCreations = false;
    float _userTableWidth = 0;
    std::unordered_set<std::string> _selectionIds;
    std::unordered_map<std::string, int> _ownEmojiTypeBySimId;
    std::unordered_map<std::pair<std::string, int>, std::set<std::string>> _userNamesByEmojiTypeBySimIdCache;

    std::vector<NetworkResourceRawTO> _unfilteredRawTOs;
    ResourceData _genomes;
    ResourceData _simulations;

    std::vector<UserTO> _userTOs;

    std::vector<TextureData> _emojis;

    bool _activateEmojiPopup = false;
    bool _showAllEmojis = false;
    NetworkResourceTreeTO _emojiPopupTO;

    std::optional<std::chrono::steady_clock::time_point> _lastRefreshTime;

    SimulationController _simController;
    StatisticsWindow _statisticsWindow;
    Viewport _viewport;
    TemporalControlWindow _temporalControlWindow;
    LoginDialogWeakPtr _loginDialog;
    EditorController _editorController;
    UploadSimulationDialogWeakPtr _uploadSimulationDialog;
};
