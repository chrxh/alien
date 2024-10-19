#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class UploadSimulationDialog : public AlienDialog
{
    MAKE_SINGLETON_CUSTOMIZED(UploadSimulationDialog);

public:
    void init(SimulationFacade const& simulationFacade, GenomeEditorWindow const& genomeEditorWindow);
    void shutdown();

    void open(NetworkResourceType resourceType, std::string const& folder = "");

private:
    UploadSimulationDialog();

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