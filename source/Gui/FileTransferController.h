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
    void onOpenSimulationDialog();
    void onOpenSimulation(std::filesystem::path const& filename);

    void onSaveSimulationDialog();

private:
    void init(PersisterFacade persisterFacade, SimulationFacade simulationFacade) override;
    void process() override;
    void shutdown() override;

    PersisterFacade _persisterFacade;
    SimulationFacade _simulationFacade;

    TaskProcessor _openSimulationProcessor;
    TaskProcessor _saveSimulationProcessor;

    std::string _referencePath;
};
