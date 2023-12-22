#pragma once

#include "EngineInterface/Definitions.h"
#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _UploadSimulationDialog : public _AlienDialog
{
public:
    _UploadSimulationDialog(
        BrowserWindow const& browserWindow,
        LoginDialog const& loginDialog,
        SimulationController const& simController,
        Viewport const& viewport,
        GenomeEditorWindow const& genomeEditorWindow);
    ~_UploadSimulationDialog();

    void open(NetworkResourceType dataType);

private:
    void processIntern();
    void openIntern();

    void onUpload();

    std::string _simName;
    std::string _simDescription;

    std::string _origSimName;
    std::string _origSimDescription;

    NetworkResourceType _dataType = NetworkResourceType_Simulation;

    BrowserWindow _browserWindow;
    LoginDialog _loginDialog;
    SimulationController _simController;
    Viewport _viewport;
    GenomeEditorWindow _genomeEditorWindow;
};