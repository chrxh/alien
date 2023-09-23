#pragma once

#include "AlienDialog.h"
#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _UploadSimulationDialog : public _AlienDialog
{
public:
    _UploadSimulationDialog(
        BrowserWindow const& browserWindow,
        SimulationController const& simController,
        NetworkController const& networkController,
        Viewport const& viewport,
        GenomeEditorWindow const& genomeEditorWindow);
    ~_UploadSimulationDialog();

    void open(DataType dataType);

private:
    void processIntern();
    void openIntern();

    void onUpload();

    std::string _simName;
    std::string _simDescription;

    std::string _origSimName;
    std::string _origSimDescription;

    DataType _dataType = DataType_Simulation;

    BrowserWindow _browserWindow;
    SimulationController _simController;
    Viewport _viewport;
    NetworkController _networkController;
    GenomeEditorWindow _genomeEditorWindow;
};