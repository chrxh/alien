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
    void init(PersisterFacade const& persisterFacade, SimulationController const& simController, TemporalControlWindow const& temporalControlWindow);
    
    void onOpenSimulation();
    void onSaveSimulation();

    void process();

private:
    PersisterFacade _persisterFacade;
    SimulationController _simController;
    TemporalControlWindow _temporalControlWindow;

    TaskProcessor _openSimulationProcessor;

    std::string _referencePath;
};
