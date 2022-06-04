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

private:
    void processIntern() override;

    void processTable();
    void processStatus();
    void processFilter();
    void processRefresh();

    void processActivated() override;

    void onOpenSimulation(std::string const& id);

    std::string _filter;
    std::set<std::string> _selectionIds;
    std::vector<RemoteSimulationData> _remoteSimulationDatas;
    std::vector<RemoteSimulationData> _filteredRemoteSimulationDatas;

    SimulationController _simController;
    NetworkController _networkController;
    StatisticsWindow _statisticsWindow;
    Viewport _viewport;
    TemporalControlWindow _temporalControlWindow;
};
