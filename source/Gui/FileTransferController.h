#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "PersisterInterface/Definitions.h"
#include "PersisterInterface/PersisterRequestId.h"

#include "Definitions.h"
#include "MainLoopEntity.h"

class FileTransferController : public MainLoopEntity
{
    MAKE_SINGLETON(FileTransferController);

public:
    void init(PersisterFacade const& persisterFacade, SimulationFacade const& simulationFacade);
    
    void onOpenSimulation();
    void onSaveSimulation();

private:
    void process() override;
    void shutdown() override {}

    PersisterFacade _persisterFacade;
    SimulationFacade _simulationFacade;

    TaskProcessor _openSimulationProcessor;

    std::string _referencePath;
};
