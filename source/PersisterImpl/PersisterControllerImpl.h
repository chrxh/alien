#pragma once

#include <thread>

#include "PersisterInterface/PersisterController.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class _PersisterControllerImpl : public _PersisterController
{
public:
    ~_PersisterControllerImpl() override;

    void init(SimulationController const& simController) override;
    void shutdown() override;

    bool isBusy() const override;
    PersisterJobState getJobState(PersisterJobId const& id) const override;
    std::vector<PersisterErrorInfo> fetchCriticalErrorInfos() override;

    PersisterJobId scheduleSaveSimulationToDisc(std::string const& filename, bool critical, float const& zoom, RealVector2D const& center) override;
    SavedSimulationData fetchSavedSimulationData(PersisterJobId const& id) override;
    PersisterErrorInfo fetchError(PersisterJobId const& id) override;

private:
    static auto constexpr MaxWorkerThreads = 4;

    PersisterJobId generateNewJobId();

    PersisterWorker _worker;
    std::thread* _thread[MaxWorkerThreads];
    int _latestJobId = 0;
};
