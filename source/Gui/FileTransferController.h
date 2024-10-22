#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/SimulationFacade.h"
#include "PersisterInterface/PersisterFacade.h"

#include "Definitions.h"
#include "MainLoopEntity.h"

class FileTransferController : public MainLoopEntity<PersisterFacade, SimulationFacade>
{
    MAKE_SINGLETON(FileTransferController);

public:
    void onOpenSimulation();
    void onSaveSimulation();

private:
    void init(PersisterFacade persisterFacade, SimulationFacade simulationFacade) override;
    void process() override;
    void shutdown() override {}

    PersisterFacade _persisterFacade;
    SimulationFacade _simulationFacade;

    TaskProcessor _openSimulationProcessor;

    std::string _referencePath;
};
