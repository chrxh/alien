#pragma once

#include "EngineInterface/Definitions.h"
#include "PersisterInterface/Definitions.h"
#include "PersisterInterface/PersisterRequestId.h"

#include "Definitions.h"
#include "Base/Singleton.h"

class FileTransferController
{
    MAKE_SINGLETON(FileTransferController);

public:
    void init(PersisterController const& persisterController, SimulationController const& simController, TemporalControlWindow const& temporalControlWindow);
    
    void onOpenSimulation();
    void onSaveSimulation();

    void process();

private:
    PersisterController _persisterController;
    SimulationController _simController;
    TemporalControlWindow _temporalControlWindow;

    TaskProcessor _openSimulationProcessor;

    std::string _referencePath;
};
