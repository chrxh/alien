#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _UploadSimulationDialog
{
public:
    _UploadSimulationDialog(SimulationController const& simController, NetworkController const& networkController);
    ~_UploadSimulationDialog();

    void process();

    void show();

private:
    void onUpload();

    SimulationController _simController; 
    NetworkController _networkController;

    bool _show = false;
    std::string _simName;
    std::string _simDescription;

    std::string _origSimName;
    std::string _origSimDescription;
};