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

    void onUpload();

    std::string _folder;
    std::unordered_map<std::string, std::string> _resourceNameByFolder;
    std::unordered_map<std::string, std::string> _resourceDescriptionByFolder;

    std::string _resourceName;
    std::string _resourceDescription;

    NetworkResourceType _resourceType = NetworkResourceType_Simulation;
    bool _share = false;

    BrowserWindow _browserWindow;
    LoginDialog _loginDialog;
    SimulationController _simController;
    Viewport _viewport;
    GenomeEditorWindow _genomeEditorWindow;
};