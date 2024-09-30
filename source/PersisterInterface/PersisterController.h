#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "PersisterErrorInfo.h"
#include "PersisterJobState.h"

class _PersisterController
{
public:
    virtual ~_PersisterController() = default;

    virtual void init(SimulationController const& simController) = 0;
    virtual void shutdown() = 0;

    virtual bool isBusy() const = 0;
    virtual PersisterJobState getJobState(PersisterJobId const& id) const = 0;
    virtual PersisterErrorInfo fetchErrorInfo() const = 0;

    virtual PersisterJobId scheduleSaveSimulationToDisc(std::string const& filename, float const& zoom, RealVector2D const& center) = 0;
    struct SavedSimulationData
    {
        std::string name;
        uint64_t timestep = 0;
        std::chrono::milliseconds realtime;
    };
    virtual SavedSimulationData fetchSavedSimulationData(PersisterJobId const& id) = 0;
};
