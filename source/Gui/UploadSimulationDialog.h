#pragma once

#include "EngineInterface/Definitions.h"
#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class _UploadSimulationDialog : public AlienDialog
{
public:
    _UploadSimulationDialog(SimulationFacade const& simulationFacade, GenomeEditorWindow const& genomeEditorWindow);
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

    SimulationFacade _simulationFacade;
    GenomeEditorWindow _genomeEditorWindow;
};