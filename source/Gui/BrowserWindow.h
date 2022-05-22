#pragma once

#include "EngineInterface/Definitions.h"

#include "AlienWindow.h"
#include "RemoteSimulationData.h"
#include "Definitions.h"

class _BrowserWindow : public _AlienWindow
{
public:
    _BrowserWindow(SimulationController const& simController, NetworkController const& networkController);
    ~_BrowserWindow();

private:
    void processIntern() override;

    void processTable();
    void processStatus();
    void processFilter();

    void processActivated() override;

    std::string _filter;
    std::set<int> _selection;
    std::vector<RemoteSimulationData> _remoteSimulationDatas;
    std::vector<RemoteSimulationData> _filteredRemoteSimulationDatas;

    SimulationController _simController;
    NetworkController _networkController;
};
