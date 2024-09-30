#pragma once

#include <thread>

#include "PersisterInterface/PersisterController.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class _PersisterControllerImpl : public _PersisterController
{
public:
    ~_PersisterControllerImpl();

    void init(SimulationController const& simController) override;
    void shutdown() override;

    PersisterJobState getJobState(PersisterJobId const& id) const override;

    PersisterJobId saveSimulationToDisc(std::string const& filename, float const& zoom, RealVector2D const& center) override;

private:
    PersisterJobId generateNewJobId();

    std::shared_ptr<_PersisterWorker> _worker;
    std::thread* _thread = nullptr;
    PersisterJobId _latestJobId = PersisterJobId(0);
};
