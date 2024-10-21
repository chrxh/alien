#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "Network/Definitions.h"

#include "AlienDialog.h"
#include "Definitions.h"

class UploadSimulationDialog : public AlienDialog<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(UploadSimulationDialog);

public:
    void open(NetworkResourceType resourceType, std::string const& folder = "");

private:
    UploadSimulationDialog();

    void initIntern(SimulationFacade simulationFacade) override;
    void shutdownIntern() override;
    void processIntern() override;

    void onUpload();

    std::string _folder;
    std::unordered_map<std::string, std::string> _resourceNameByFolder;
    std::unordered_map<std::string, std::string> _resourceDescriptionByFolder;

    std::string _resourceName;
    std::string _resourceDescription;

    NetworkResourceType _resourceType = NetworkResourceType_Simulation;
    bool _share = false;

    SimulationFacade _simulationFacade;
};