#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _UploadSimulationDialog
{
public:
    _UploadSimulationDialog(
        BrowserWindow const& browserWindow,
        SimulationController const& simController,
        NetworkController const& networkController,
        Viewport const& viewport);
    ~_UploadSimulationDialog();

    void process();

    void show();

private:
    void onUpload();

    BrowserWindow _browserWindow;
    SimulationController _simController; 
    Viewport _viewport;
    NetworkController _networkController;

    bool _show = false;
    std::string _simName;
    std::string _simDescription;

    std::string _origSimName;
    std::string _origSimDescription;
};