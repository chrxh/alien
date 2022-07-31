#pragma once

#include "EngineInterface/Definitions.h"

#include "AlienWindow.h"
#include "RemoteSimulationData.h"
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
    void refreshIntern(bool firstTimeStartup);

    void processIntern() override;

    void processTable();
    void processStatus();
    void processFilter();
    void processToolbar();
    void processShortenedText(std::string const& text);
    bool processDetailButton();

    void processActivated() override;

    void sortTable();

    void onOpenSimulation(std::string const& id);
    void onDeleteSimulation(std::string const& id);
    void onToggleLike(RemoteSimulationData& entry);

    bool isLiked(std::string const& id);
    std::string getUserLikes(std::string const& id);

    bool _scheduleRefresh = false;
    bool _scheduleSort = false;
    std::string _filter;
    std::unordered_set<std::string> _selectionIds;
    std::unordered_set<std::string> _likedIds;
    std::unordered_map<std::string, std::set<std::string>> _userLikesByIdCache;
    std::vector<RemoteSimulationData> _remoteSimulationDatas;
    std::vector<RemoteSimulationData> _filteredRemoteSimulationDatas;

    SimulationController _simController;
    NetworkController _networkController;
    StatisticsWindow _statisticsWindow;
    Viewport _viewport;
    TemporalControlWindow _temporalControlWindow;
    LoginDialogWeakPtr _loginDialog;
    UploadSimulationDialogWeakPtr _uploadSimulationDialog;
};
