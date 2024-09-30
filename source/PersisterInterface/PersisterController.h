#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "PersisterJobState.h"

class _PersisterController
{
public:
    virtual ~_PersisterController() = default;

    virtual void init(SimulationController const& simController) = 0;
    virtual void shutdown() = 0;

    virtual PersisterJobState getJobState(PersisterJobId const& id) const = 0;

    virtual PersisterJobId saveSimulationToDisc(std::string const& filename, float const& zoom, RealVector2D const& center) = 0;
};
