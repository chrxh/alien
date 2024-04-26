#pragma once

#include <chrono>

#include "Base/Hashes.h"
#include "Base/Cache.h"
#include "EngineInterface/Definitions.h"
#include "Network/NetworkResourceTreeTO.h"
#include "Network/NetworkResourceRawTO.h"
#include "Network/UserTO.h"
#include "EngineInterface/SerializerService.h"

#include "AlienWindow.h"
#include "Definitions.h"

struct ImGuiTableColumnSortSpecs;

using BrowserCache = Cache<std::string, DeserializedSimulation, 5>;

class _BrowserWindow : public _AlienWindow
{
public:
    _BrowserWindow(
        SimulationController const& simController,
        StatisticsWindow const& statisticsWindow,
        TemporalControlWindow const& temporalControlWindow,
        EditorController const& editorController);
    ~_BrowserWindow();

    void registerCyclicReferences(
        LoginDialogWeakPtr const& loginDialog,
        UploadSimulationDialogWeakPtr const& uploadSimulationDialog,
        EditSimulationDialogWeakPtr const& editSimulationDialog);

    void onRefresh();
    WorkspaceType getCurrentWorkspaceType() const;

    BrowserCache& getSimulationCache();
    void registerUploadedSimulation(std::string const& id);

private:
    struct WorkspaceId
    {
        NetworkResourceType resourceType;
        WorkspaceType workspaceType;
        auto operator<=>(WorkspaceId const&) const = default;
    };
    struct Workspace
    {
        std::vector<ImGuiTableColumnSortSpecs> sortSpecs;
        std::vector<NetworkResourceRawTO> rawTOs;    //unfiltered, sorted
        std::vector<NetworkResourceTreeTO> treeTOs;  //filtered, sorted
        std::set<std::vector<std::string>> collapsedFolderNames;
    };

    void refreshIntern(bool withRetry);

    void processIntern() override;
    void processBackground() override;

    void processToolbar();
    void processWorkspaceSelectionAndFilter();
    void processWorkspace();
    void processMovableSeparator();
    void processUserList();
    void processStatus();

    void processSimulationList();
    void processGenomeList();

    bool processResourceNameField(NetworkResourceTreeTO const& treeTO, std::set<std::vector<std::string>>& collapsedFolderNames);   //return true if folder symbol clicked
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

    bool processFolderTreeSymbols(NetworkResourceTreeTO const& treeTO, std::set<std::vector<std::string>>& collapsedFolderNames);   //return true if folder symbol clicked
    void processEmojiWindow();
    void processEmojiButton(int emojiType);

    void processDownloadButton(BrowserLeaf const& leaf);

    void processShortenedText(std::string const& text, bool bold = false);
    bool processActionButton(std::string const& text);
    bool processDetailButton();

    void processActivated() override;

    void createTreeTOs(Workspace& workspace);
    void sortUserList();

    void onDownloadResource(BrowserLeaf const& leaf);
    void onEditResource(NetworkResourceTreeTO const& treeTO);
    void onMoveResource(NetworkResourceTreeTO const& treeTO);
    void onDeleteResource(NetworkResourceTreeTO const& treeTO);
    void onToggleLike(NetworkResourceTreeTO const& to, int emojiType);
    void onExpandFolders();
    void onCollapseFolders();
    void openWeblink(std::string const& link);

    bool isOwner(NetworkResourceTreeTO const& treeTO) const;
    std::string getUserNamesToEmojiType(std::string const& resourceId, int emojiType);

    std::vector<std::string> getAllSimulationIds() const;

    void pushTextColor(NetworkResourceTreeTO const& to);
    void popTextColor();

    bool _activateEmojiPopup = false;
    bool _showAllEmojis = false;
    NetworkResourceTreeTO _emojiPopupTO;
    std::optional<std::chrono::steady_clock::time_point> _lastRefreshTime;

    std::vector<UserTO> _userTOs;
    WorkspaceId _currentWorkspace = {NetworkResourceType_Simulation, WorkspaceType_AlienProject};
    std::map<WorkspaceId, Workspace> _workspaces;
    std::unordered_set<std::string> _simIdsFromLastSession;

    NetworkResourceTreeTO _selectedTreeTO;

    std::string _filter;
    float _userTableWidth = 0;
    std::unordered_map<std::string, int> _ownEmojiTypeBySimId;
    std::unordered_map<std::pair<std::string, int>, std::set<std::string>> _userNamesByEmojiTypeBySimIdCache;

    std::vector<TextureData> _emojis;

    BrowserCache _simulationCache;

    SimulationController _simController;
    StatisticsWindow _statisticsWindow;
    TemporalControlWindow _temporalControlWindow;
    LoginDialogWeakPtr _loginDialog;
    EditorController _editorController;
    UploadSimulationDialogWeakPtr _uploadSimulationDialog;
    EditSimulationDialogWeakPtr _editSimulationDialog;
};
