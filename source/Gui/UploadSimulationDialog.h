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

    void open(NetworkResourceType resourceType, std::string const& folder = "");

private:
    void processIntern();
    void openIntern();

    void onUpload();

    std::string _folder;
    std::string _resourceName;
    std::string _resourceDescription;

    std::string _origResourceName;
    std::string _origResourceDescription;

    NetworkResourceType _resourceType = NetworkResourceType_Simulation;
    WorkspaceType _workspaceType = WorkspaceType_Shared;

    BrowserWindow _browserWindow;
    LoginDialog _loginDialog;
    SimulationController _simController;
    Viewport _viewport;
    GenomeEditorWindow _genomeEditorWindow;
};