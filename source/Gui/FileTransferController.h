#pragma once

#include "EngineInterface/Definitions.h"
#include "PersisterInterface/Definitions.h"
#include "PersisterInterface/PersisterRequestId.h"

#include "Definitions.h"

class FileTransferController
{
public:
    static FileTransferController& get();

    void init(PersisterController const& persisterController, SimulationController const& simController, TemporalControlWindow const& temporalControlWindow);
    
    void onOpenSimulation();
    void onSaveSimulation();

    void process();

private:
    PersisterController _persisterController;
    SimulationController _simController;
    TemporalControlWindow _temporalControlWindow;

    std::string _startingPath;
    std::vector<PersisterRequestId> _readSimulationRequestIds;
};
